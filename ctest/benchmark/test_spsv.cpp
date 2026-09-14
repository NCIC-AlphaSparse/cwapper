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


// Performance test for flagsparseSpSV, through the shared harness.
//
// A triangular solve is NOT bound by nonzero count the way SpMV is: row r
// cannot start until every row it depends on has finished, so the wall clock
// follows the DEPENDENCY DEPTH. Two matrices with the same nnz and different
// structure are different work. That is why the sweep here is over n and
// density -- which together set the depth -- rather than over nnz alone, and
// why the GFLOPS column is worth reading with suspicion on this operator.
//
// flagsparseSpSV_analysis is outside the clock: it is the per-matrix step,
// which is exactly what cuSPARSE separates it out for.

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

BenchReport g_report("spsv");

bool bench_one(flagsparseHandle_t handle, const char* name, int64_t n, double density,
               bool lower, flagsparseDataType_t dtype,
               flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR) {
    const TriMatrix T = random_triangular(n, density, lower, false, 4242);
    BenchRow row;
    row.name = name;
    row.tag("format", format == FLAGSPARSE_FORMAT_COO ? "coo" : "csr")
       .tag("fill", lower ? "lower" : "upper")
       .tag("dtype", dtype == FLAGSPARSE_R_64F ? "float64" : "float32")
       .num("n", static_cast<double>(n))
       .num("nnz", static_cast<double>(T.nnz()))
       .num("density", density)
       .num("mean_row_len", n ? static_cast<double>(T.nnz()) / static_cast<double>(n) : 0.0);

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
    DeviceBuffer d_x(static_cast<std::size_t>(n) * esize);
    DeviceBuffer d_y(static_cast<std::size_t>(n) * esize);
    {
        const std::vector<double> ones64(static_cast<std::size_t>(n), 1.0);
        if (dtype == FLAGSPARSE_R_64F) {
            to_device(d_x.get(), ones64.data(), d_x.size());
        } else {
            const std::vector<float> ones(ones64.begin(), ones64.end());
            to_device(d_x.get(), ones.data(), d_x.size());
        }
    }

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseSpSVDescr_t descr = nullptr;
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
    const flagsparseFillMode_t fill =
        lower ? FLAGSPARSE_FILL_MODE_LOWER : FLAGSPARSE_FILL_MODE_UPPER;
    const flagsparseDiagType_t diag = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
    flagsparseCreateDnVec(&vecX, n, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, n, d_y.get(), dtype);
    flagsparseSpSV_createDescr(&descr);

    const double alpha_d = 1.0;
    const float alpha_f = 1.0f;
    const void* alpha = (dtype == FLAGSPARSE_R_64F) ? static_cast<const void*>(&alpha_d)
                                                    : static_cast<const void*>(&alpha_f);
    const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    const auto ALG = FLAGSPARSE_SPSV_ALG_DEFAULT;

    size_t bufsz = 0;
    const flagsparseStatus_t sized =
        flagsparseSpSV_bufferSize(handle, NT, alpha, matA, vecX, vecY, dtype, ALG, descr,
                                  &bufsz);
    bool measured = false;
    if (sized != FLAGSPARSE_STATUS_SUCCESS) {
        g_report.skip(row, sized == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported"
                                                                   : "failed",
                      status_name(sized));
        measured = (sized == FLAGSPARSE_STATUS_NOT_SUPPORTED);
    } else {
        DeviceBuffer scratch(bufsz);
        const flagsparseStatus_t analysed =
            flagsparseSpSV_analysis(handle, NT, alpha, matA, vecX, vecY, dtype, ALG,
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
                    return flagsparseSpSV_solve(handle, NT, alpha, matA, vecX, vecY,
                                                dtype, ALG, descr);
                },
                // One multiply-add per stored nonzero. Read as a rate only with
                // the dependency depth in mind: this operator is latency bound.
                2.0 * static_cast<double>(T.nnz()));
        }
        flagsparseSpSV_destroyDescr(descr);
        flagsparseDestroyDnVec(vecY);
        flagsparseDestroyDnVec(vecX);
        flagsparseDestroySpMat(matA);
        return measured;
    }
    flagsparseSpSV_destroyDescr(descr);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return measured;
}

class SpSVBenchmark : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

// n is the dependency depth, so this is the axis that actually moves the clock.
TEST_F(SpSVBenchmark, DepthSweep) {
    for (int64_t n : {1024, 4096, 16384, 65536}) {
        std::ostringstream name;
        name << "n_" << n;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), n, 0.001, true,
                              FLAGSPARSE_R_64F));
    }
}

// Same depth, more work per row: separates the per-row cost from the chain.
TEST_F(SpSVBenchmark, RowWidthSweep) {
    for (double d : {0.0005, 0.002, 0.01, 0.05}) {
        std::ostringstream name;
        name << "density_" << d;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 8192, d, true,
                              FLAGSPARSE_R_64F));
    }
}

// Upper solves run the chain backwards and scan a row from its far end; same
// work, different traversal, so it is measured rather than assumed symmetric.
TEST_F(SpSVBenchmark, FillModeAndDtype) {
    for (bool lower : {true, false}) {
        for (auto dtype : {FLAGSPARSE_R_32F, FLAGSPARSE_R_64F}) {
            std::ostringstream name;
            name << (lower ? "lower_" : "upper_")
                 << (dtype == FLAGSPARSE_R_64F ? "fp64" : "fp32");
            EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 8192, 0.002, lower,
                                  dtype));
        }
    }
}

// COO runs the same kernel over the row offsets analysis had to build anyway,
// so the gap between these rows is that build, not a different solve.
TEST_F(SpSVBenchmark, CooMatchesCsr) {
    EXPECT_TRUE(bench_one(handle.h, "csr_8k", 8192, 0.002, true, FLAGSPARSE_R_64F,
                          FLAGSPARSE_FORMAT_CSR));
    EXPECT_TRUE(bench_one(handle.h, "coo_8k", 8192, 0.002, true, FLAGSPARSE_R_64F,
                          FLAGSPARSE_FORMAT_COO));
    g_report.write();
}

}  // namespace
