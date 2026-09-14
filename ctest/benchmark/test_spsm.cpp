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


// Performance test for flagsparseSpSM, through the shared harness.
//
// Same dependency chain as SpSV, but amortised over many right-hand sides: the
// chain is walked once no matter how wide B is, so the interesting axis here is
// n_rhs. It is also the axis that picks the RHS tile (next power of two, capped
// at 1024) and with it the warp count, so a regression lands on one side of a
// power of two and not the other.
//
// Two costs are deliberately inside the clock and one is outside. Analysis is
// outside -- it extracts the diagonal, which depends only on the matrix. The
// two strided copies that move B into the packed work array and the result back
// out are INSIDE, because a caller pays them on every solve; hiding them would
// report a solve this API cannot actually deliver.

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

BenchReport g_report("spsm");

bool bench_one(flagsparseHandle_t handle, const char* name, int64_t n, int64_t n_rhs,
               double density, flagsparseDataType_t dtype,
               flagsparseOrder_t order = FLAGSPARSE_ORDER_ROW,
               flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR) {
    const TriMatrix T = random_triangular(n, density, true, false, 4242);
    BenchRow row;
    row.name = name;
    row.tag("format", format == FLAGSPARSE_FORMAT_COO ? "coo" : "csr")
       .tag("order", order == FLAGSPARSE_ORDER_COL ? "col" : "row")
       .tag("dtype", dtype == FLAGSPARSE_R_64F ? "float64" : "float32")
       .num("n", static_cast<double>(n))
       .num("n_rhs", static_cast<double>(n_rhs))
       .num("nnz", static_cast<double>(T.nnz()))
       .num("density", density);

    const std::size_t esize = (dtype == FLAGSPARSE_R_64F) ? sizeof(double) : sizeof(float);
    DeviceBuffer d_val(static_cast<std::size_t>(T.nnz()) * esize);
    if (dtype == FLAGSPARSE_R_64F) {
        to_device(d_val.get(), T.values.data(), d_val.size());
    } else {
        const std::vector<float> v(T.values.begin(), T.values.end());
        to_device(d_val.get(), v.data(), d_val.size());
    }
    DeviceBuffer d_col = DeviceBuffer::from(T.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(T.indptr);
    DeviceBuffer d_row = DeviceBuffer::from(coo_row_indices_of(T));
    DeviceBuffer d_b(static_cast<std::size_t>(n * n_rhs) * esize);
    DeviceBuffer d_c(static_cast<std::size_t>(n * n_rhs) * esize);
    {
        const std::vector<double> ones64(static_cast<std::size_t>(n * n_rhs), 1.0);
        if (dtype == FLAGSPARSE_R_64F) {
            to_device(d_b.get(), ones64.data(), d_b.size());
        } else {
            const std::vector<float> ones(ones64.begin(), ones64.end());
            to_device(d_b.get(), ones.data(), d_b.size());
        }
    }

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    flagsparseSpSMDescr_t descr = nullptr;
    const flagsparseStatus_t created =
        (format == FLAGSPARSE_FORMAT_COO)
            ? flagsparseCreateCoo(&matA, n, n, T.nnz(), d_row.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype)
            : flagsparseCreateCsr(&matA, n, n, T.nnz(), d_ptr.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                  dtype);
    if (created != FLAGSPARSE_STATUS_SUCCESS) {
        g_report.skip(row, "failed", status_name(created));
        return false;
    }
    const flagsparseFillMode_t fill = FLAGSPARSE_FILL_MODE_LOWER;
    const flagsparseDiagType_t diag = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
    const int64_t ld = (order == FLAGSPARSE_ORDER_ROW) ? n_rhs : n;
    flagsparseCreateDnMat(&matB, n, n_rhs, ld, d_b.get(), dtype, order);
    flagsparseCreateDnMat(&matC, n, n_rhs, ld, d_c.get(), dtype, order);
    flagsparseSpSM_createDescr(&descr);

    const double alpha_d = 1.0;
    const float alpha_f = 1.0f;
    const void* alpha = (dtype == FLAGSPARSE_R_64F) ? static_cast<const void*>(&alpha_d)
                                                    : static_cast<const void*>(&alpha_f);
    const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    const auto ALG = FLAGSPARSE_SPSM_ALG_DEFAULT;

    bool measured = false;
    size_t bufsz = 0;
    const flagsparseStatus_t sized =
        flagsparseSpSM_bufferSize(handle, NT, NT, alpha, matA, matB, matC, dtype, ALG,
                                  descr, &bufsz);
    if (sized != FLAGSPARSE_STATUS_SUCCESS) {
        g_report.skip(row, sized == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported"
                                                                   : "failed",
                      status_name(sized));
        measured = (sized == FLAGSPARSE_STATUS_NOT_SUPPORTED);
    } else {
        DeviceBuffer scratch(bufsz);
        row.num("scratch_bytes", static_cast<double>(bufsz));
        const flagsparseStatus_t analysed =
            flagsparseSpSM_analysis(handle, NT, NT, alpha, matA, matB, matC, dtype, ALG,
                                    descr, scratch.get());
        if (analysed != FLAGSPARSE_STATUS_SUCCESS) {
            g_report.skip(row, analysed == FLAGSPARSE_STATUS_NOT_SUPPORTED
                                   ? "not_supported" : "failed",
                          std::string("analysis: ") + status_name(analysed));
            measured = (analysed == FLAGSPARSE_STATUS_NOT_SUPPORTED);
        } else {
            measured = g_report.measure(
                row,
                [&] {
                    return flagsparseSpSM_solve(handle, NT, NT, alpha, matA, matB, matC,
                                                dtype, ALG, descr);
                },
                2.0 * static_cast<double>(T.nnz()) * static_cast<double>(n_rhs));
        }
    }

    flagsparseSpSM_destroyDescr(descr);
    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
    return measured;
}

class SpSMBenchmark : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

// The chain is walked once whatever n_rhs is, so this is where SpSM earns its
// keep over repeated SpSV -- and where the RHS tile threshold shows up.
TEST_F(SpSMBenchmark, RhsWidthSweep) {
    for (int64_t k : {1, 8, 32, 64, 128, 512, 1024, 2048}) {
        std::ostringstream name;
        name << "nrhs_" << k;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 4096, k, 0.002,
                              FLAGSPARSE_R_64F));
    }
}

TEST_F(SpSMBenchmark, DepthSweep) {
    for (int64_t n : {1024, 4096, 16384}) {
        std::ostringstream name;
        name << "n_" << n;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), n, 32, 0.002,
                              FLAGSPARSE_R_64F));
    }
}

// Column-major B and C go through the same strided copy as row-major, so any
// difference here is the copy's coalescing, not the solve's.
TEST_F(SpSMBenchmark, LayoutFormatAndDtype) {
    EXPECT_TRUE(bench_one(handle.h, "row_major", 4096, 32, 0.002, FLAGSPARSE_R_64F,
                          FLAGSPARSE_ORDER_ROW));
    EXPECT_TRUE(bench_one(handle.h, "col_major", 4096, 32, 0.002, FLAGSPARSE_R_64F,
                          FLAGSPARSE_ORDER_COL));
    EXPECT_TRUE(bench_one(handle.h, "fp32", 4096, 32, 0.002, FLAGSPARSE_R_32F));
    EXPECT_TRUE(bench_one(handle.h, "coo", 4096, 32, 0.002, FLAGSPARSE_R_64F,
                          FLAGSPARSE_ORDER_ROW, FLAGSPARSE_FORMAT_COO));
    g_report.write();
}

}  // namespace
