// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// Performance test for flagsparseSpMM. Timing follows spec §6.4: >=10 warmup
// iterations, >=100 timed, median reported, JSON on disk.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

BenchReport g_report("spmm");

// random_csr emits rows in order, so expanding it gives a row-sorted COO.
std::vector<int32_t> coo_row_indices(const CsrMatrix& A) {
    std::vector<int32_t> row;
    row.reserve(static_cast<std::size_t>(A.nnz));
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            row.push_back(static_cast<int32_t>(r));
        }
    }
    return row;
}

bool bench_one(flagsparseHandle_t handle, const char* name, int64_t rows, int64_t cols,
               int64_t n, double density,
               flagsparseOrder_t order = FLAGSPARSE_ORDER_ROW,
               flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR) {
    const CsrMatrix A = random_csr(rows, cols, density, 4242);
    const std::vector<float> values(A.values.begin(), A.values.end());
    const std::vector<float> b(static_cast<std::size_t>(cols * n), 1.0f);
    const std::vector<float> c(static_cast<std::size_t>(rows * n), 0.0f);

    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(A.indptr);
    DeviceBuffer d_row = DeviceBuffer::from(coo_row_indices(A));
    DeviceBuffer d_b   = DeviceBuffer::from(b);
    DeviceBuffer d_c   = DeviceBuffer::from(c);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    const flagsparseStatus_t created =
        (format == FLAGSPARSE_FORMAT_COO)
            ? flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, d_row.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F)
            : flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_ptr.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
    if (created != FLAGSPARSE_STATUS_SUCCESS) return false;
    const int64_t ld_b = (order == FLAGSPARSE_ORDER_ROW) ? n : cols;
    const int64_t ld_c = (order == FLAGSPARSE_ORDER_ROW) ? n : rows;
    flagsparseCreateDnMat(&matB, cols, n, ld_b, d_b.get(), FLAGSPARSE_R_32F, order);
    flagsparseCreateDnMat(&matC, rows, n, ld_c, d_c.get(), FLAGSPARSE_R_32F, order);

    const float alpha = 1.0f, beta = 0.0f;
    size_t buffer_size = 0;
    flagsparseSpMM_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                              &beta, matC, FLAGSPARSE_R_32F,
                              FLAGSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
    DeviceBuffer scratch(buffer_size);
    // Preprocess outside the clock: it does the index readback once, which is
    // exactly what cuSPARSE's preprocess step is for. For COO that readback also
    // builds the row-offsets array, so timing it here would charge a per-matrix
    // setup to every iteration and make COO look far worse than it is.
    flagsparseSpMM_preprocess(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                              &beta, matC, FLAGSPARSE_R_32F,
                              FLAGSPARSE_SPMM_ALG_DEFAULT, scratch.get());

    BenchRow row;
    row.name = name;
    row.tag("format", (format == FLAGSPARSE_FORMAT_COO) ? "coo" : "csr")
       .tag("order", (order == FLAGSPARSE_ORDER_COL) ? "col" : "row")
       .tag("dtype", "float32")
       .num("rows", static_cast<double>(rows))
       .num("cols", static_cast<double>(cols))
       .num("n", static_cast<double>(n))
       .num("nnz", static_cast<double>(A.nnz))
       .num("density", density);

    const bool measured = g_report.measure(
        row,
        [&] {
            return flagsparseSpMM(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                  FLAGSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                                  &beta, matC, FLAGSPARSE_R_32F,
                                  FLAGSPARSE_SPMM_ALG_DEFAULT, scratch.get());
        },
        // 2 flops per nonzero per dense column (one multiply, one add).
        2.0 * static_cast<double>(A.nnz) * static_cast<double>(n));

    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
    return measured;
}

class SpMMBenchmark : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

// Spec §6.4 asks for small / medium / large and a density sweep.
TEST_F(SpMMBenchmark, SizeSweep) {
    EXPECT_TRUE(bench_one(handle.h, "small",  512,   512,   32, 0.01));
    EXPECT_TRUE(bench_one(handle.h, "medium", 8192,  8192,  32, 0.001));
    EXPECT_TRUE(bench_one(handle.h, "large",  32768, 32768, 32, 0.0001));
}

TEST_F(SpMMBenchmark, DensitySweep) {
    for (double d : {0.0001, 0.001, 0.01, 0.1}) {
        std::ostringstream name;
        name << "density_" << d;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 4096, 4096, 32, d));
    }
}

// The dense width drives BLOCK_N, BLOCK_NNZ and the warp count, so it is the
// axis most likely to expose a bad launch choice.
TEST_F(SpMMBenchmark, DenseWidthSweep) {
    for (int64_t n : {4, 16, 32, 64, 128, 256}) {
        std::ostringstream name;
        name << "n_" << n;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 4096, 4096, n, 0.001));
    }
}

// Column-major costs nothing but a stride swap in the dispatch; whether it
// costs anything on the device is a measurement, not an assumption.
TEST_F(SpMMBenchmark, ColumnMajorDense) {
    EXPECT_TRUE(bench_one(handle.h, "n_32_col", 4096, 4096, 32, 0.001,
                          FLAGSPARSE_ORDER_COL));
}

// The same shapes in COO, so the two formats can be read off the same JSON.
// Measured on a 5090, COO comes out FASTER than CSR here -- 0.0608 vs 0.0853 ms
// at 4096^2/density 0.1, 0.0116 vs 0.0167 at n=128 -- which is the opposite of
// what carrying a row index per nonzero would suggest. The two kernels are
// structurally the same row loop; what differs is the launch. COO uses the
// operator package's swept BLOCK_NNZ=4 and derives num_warps from the device
// warp size, while the CSR path takes BLOCK_NNZ from the warp/factor heuristic.
// Read that as "the CSR launch config has room", not as "COO is the fast format".
TEST_F(SpMMBenchmark, CooSizeAndWidthSweep) {
    EXPECT_TRUE(bench_one(handle.h, "coo_small",  512,  512,  32, 0.01,
                          FLAGSPARSE_ORDER_ROW, FLAGSPARSE_FORMAT_COO));
    EXPECT_TRUE(bench_one(handle.h, "coo_medium", 8192, 8192, 32, 0.001,
                          FLAGSPARSE_ORDER_ROW, FLAGSPARSE_FORMAT_COO));
    for (double d : {0.0001, 0.001, 0.01, 0.1}) {
        std::ostringstream name;
        name << "coo_density_" << d;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 4096, 4096, 32, d,
                              FLAGSPARSE_ORDER_ROW, FLAGSPARSE_FORMAT_COO));
    }
    for (int64_t n : {4, 32, 128}) {
        std::ostringstream name;
        name << "coo_n_" << n;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 4096, 4096, n, 0.001,
                              FLAGSPARSE_ORDER_ROW, FLAGSPARSE_FORMAT_COO));
    }
    g_report.write();
}

}  // namespace
