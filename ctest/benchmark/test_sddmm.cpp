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


// Performance test for flagsparseSDDMM. Timing follows spec §6.4 through the
// shared harness: >=10 warmup, >=100 timed, median, JSON on disk carrying the
// backend and a per-row status.
//
// SDDMM is nnz-parallel with a reduction over k, so the two axes that decide
// the launch are k (which picks BLOCK_K) and the mean row length (which picks
// BLOCK_P between 64 and 512). Both are swept, because a launch-config
// regression shows up on one side of a threshold and not the other.

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

BenchReport g_report("sddmm");

bool bench_one(flagsparseHandle_t handle, const char* name, int64_t rows, int64_t cols,
               int64_t k, double density, flagsparseDataType_t dtype) {
    const CsrMatrix C = random_csr(rows, cols, density, 4242);
    BenchRow row;
    row.name = name;
    row.tag("dtype", dtype == FLAGSPARSE_R_64F ? "float64" : "float32")
       .num("rows", static_cast<double>(rows))
       .num("cols", static_cast<double>(cols))
       .num("k", static_cast<double>(k))
       .num("nnz", static_cast<double>(C.nnz))
       .num("density", density)
       .num("mean_row_len", rows ? static_cast<double>(C.nnz) / static_cast<double>(rows) : 0.0);

    if (C.nnz == 0) {
        g_report.skip(row, "not_supported", "empty pattern: nothing to measure");
        return true;
    }
    const std::size_t esize = (dtype == FLAGSPARSE_R_64F) ? sizeof(double) : sizeof(float);
    const std::vector<double> a64(static_cast<std::size_t>(rows * k), 0.5);
    const std::vector<double> b64(static_cast<std::size_t>(k * cols), 0.25);

    DeviceBuffer d_a(static_cast<std::size_t>(rows * k) * esize);
    DeviceBuffer d_b(static_cast<std::size_t>(k * cols) * esize);
    DeviceBuffer d_val(static_cast<std::size_t>(C.nnz) * esize);
    DeviceBuffer d_col = DeviceBuffer::from(C.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(C.indptr);
    if (dtype == FLAGSPARSE_R_64F) {
        to_device(d_a.get(), a64.data(), d_a.size());
        to_device(d_b.get(), b64.data(), d_b.size());
    } else {
        const std::vector<float> a(a64.begin(), a64.end()), b(b64.begin(), b64.end());
        to_device(d_a.get(), a.data(), d_a.size());
        to_device(d_b.get(), b.data(), d_b.size());
    }

    flagsparseDnMatDescr_t matA = nullptr, matB = nullptr;
    flagsparseSpMatDescr_t matC = nullptr;
    flagsparseCreateDnMat(&matA, rows, k, k, d_a.get(), dtype, FLAGSPARSE_ORDER_ROW);
    // op(B) is k x cols, the natural orientation. The kernel indexes y as
    // [column][k], but that is the dispatch layer's stride swap, not something
    // the descriptor has to be turned around for.
    flagsparseCreateDnMat(&matB, k, cols, cols, d_b.get(), dtype, FLAGSPARSE_ORDER_ROW);
    if (flagsparseCreateCsr(&matC, rows, cols, C.nnz, d_ptr.get(), d_col.get(),
                            d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_BASE_ZERO, dtype)
        != FLAGSPARSE_STATUS_SUCCESS) {
        g_report.skip(row, "failed", "could not create the CSR descriptor");
        flagsparseDestroyDnMat(matB); flagsparseDestroyDnMat(matA);
        return false;
    }

    const double alpha_d = 1.0, beta_d = 0.0;
    const float alpha_f = 1.0f, beta_f = 0.0f;
    const void* alpha = (dtype == FLAGSPARSE_R_64F) ? static_cast<const void*>(&alpha_d)
                                                    : static_cast<const void*>(&alpha_f);
    const void* beta = (dtype == FLAGSPARSE_R_64F) ? static_cast<const void*>(&beta_d)
                                                   : static_cast<const void*>(&beta_f);
    const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    const auto ALG = FLAGSPARSE_SDDMM_ALG_DEFAULT;

    size_t bufsz = 0;
    const flagsparseStatus_t sized =
        flagsparseSDDMM_bufferSize(handle, NT, NT, alpha, matA, matB, beta, matC, dtype,
                                   ALG, &bufsz);
    if (sized != FLAGSPARSE_STATUS_SUCCESS) {
        g_report.skip(row, sized == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported"
                                                                   : "failed",
                      status_name(sized));
        flagsparseDestroySpMat(matC);
        flagsparseDestroyDnMat(matB); flagsparseDestroyDnMat(matA);
        return sized == FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    DeviceBuffer scratch(bufsz);
    // Preprocess outside the clock: it expands the pattern to one row id per
    // nonzero, which depends on the sparsity only and survives every change of
    // A, B, alpha and beta.
    flagsparseSDDMM_preprocess(handle, NT, NT, alpha, matA, matB, beta, matC, dtype, ALG,
                               scratch.get());

    const bool measured = g_report.measure(
        row,
        [&] {
            return flagsparseSDDMM(handle, NT, NT, alpha, matA, matB, beta, matC, dtype,
                                   ALG, scratch.get());
        },
        2.0 * static_cast<double>(C.nnz) * static_cast<double>(k));

    flagsparseDestroySpMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroyDnMat(matA);
    return measured;
}

class SDDMMBenchmark : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(SDDMMBenchmark, SizeSweep) {
    EXPECT_TRUE(bench_one(handle.h, "small",  512,   512,   64, 0.01,   FLAGSPARSE_R_32F));
    EXPECT_TRUE(bench_one(handle.h, "medium", 8192,  8192,  64, 0.001,  FLAGSPARSE_R_32F));
    EXPECT_TRUE(bench_one(handle.h, "large",  32768, 32768, 64, 0.0001, FLAGSPARSE_R_32F));
}

// k either side of 32, where BLOCK_K stops growing with it.
TEST_F(SDDMMBenchmark, KSweep) {
    for (int64_t k : {8, 16, 32, 64, 128, 256}) {
        std::ostringstream name;
        name << "k_" << k;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 4096, 4096, k, 0.001,
                              FLAGSPARSE_R_32F));
    }
}

// Mean row length either side of 16, which is where BLOCK_P switches from 64 to
// 512 -- and fp64 always takes the narrow one, so it is swept too.
TEST_F(SDDMMBenchmark, RowLengthAndDtype) {
    for (double d : {0.001, 0.01, 0.05}) {
        std::ostringstream f32, f64;
        f32 << "fp32_density_" << d;
        f64 << "fp64_density_" << d;
        EXPECT_TRUE(bench_one(handle.h, f32.str().c_str(), 4096, 4096, 64, d,
                              FLAGSPARSE_R_32F));
        EXPECT_TRUE(bench_one(handle.h, f64.str().c_str(), 4096, 4096, 64, d,
                              FLAGSPARSE_R_64F));
    }
    g_report.write();
}

}  // namespace
