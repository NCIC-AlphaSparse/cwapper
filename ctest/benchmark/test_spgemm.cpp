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

// SpGEMM (C = A*A) over the real-matrix corpus, against the vendor baseline.
//
// THE ORACLE IS INDIRECT, ON PURPOSE. Materialising A*A in fp64 on the host costs
// more memory than the product itself for these matrices. But if C = A*A then
// C*x == A*(A*x) for any x, and both sides of that are host SpMVs over vectors of
// length rows -- cheap, and sensitive to a wrong value anywhere in C that x does
// not happen to annihilate. A fixed non-degenerate x is used for the same reason
// the dense operands elsewhere are fixed.
//
// ONLY `copy` IS TIMED. The flow is stateful: workEstimation and compute discover
// C's size and cannot be replayed independently, so the timed region is the phase
// that materialises the result. The vendor baseline times its own copy phase for
// the same reason, which keeps the ratio like-for-like. An earlier sweep of mine
// reported this column without saying so; it says so now.
//
// REAL DTYPES ONLY, matching the kernel's coverage.

#include <gtest/gtest.h>

#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spgemm");

// Reading C back and multiplying costs memory proportional to its nonzeros. Past
// this the row is measured but recorded accuracy="unchecked", which withholds the
// speedup rather than quietly averaging an unverified ratio into the geomean.
constexpr int64_t kVerifyNnzBudget = 20'000'000;

std::vector<double> host_spmv(const CsrMatrix& A, const std::vector<double>& x) {
    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        double acc = 0.0;
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            acc += A.values[static_cast<std::size_t>(p)] *
                   x[static_cast<std::size_t>(A.indices[static_cast<std::size_t>(p)])];
        }
        y[static_cast<std::size_t>(r)] = acc;
    }
    return y;
}

}  // namespace

TEST(SpgemmBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    const auto declared = variants_of("spgemm");
    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        if (A.rows != A.cols) {
            g_report.skip(BenchRow{}.tag("matrix", entry.name).tag("format", "csr"),
                          "skipped_shape", "A*A needs a square A");
            continue;
        }
        const std::vector<double> x = dense_pattern(static_cast<std::size_t>(A.cols));
        const std::vector<double> ref = host_spmv(A, host_spmv(A, x));  // A*(A*x)

        for (const registry::Variant* v : declared) {
            if (std::string(v->format) != "csr") {
                const std::string why =
                    std::string("benchmark/test_spgemm.cpp has no ") + v->format +
                    " operand builder yet";
                report_unimplemented(g_report, *v, why.c_str());
                continue;
            }
            const auto dt = v->dt;
            BenchRow row;
            row.name = std::string("spgemm_csr_") + v->dtype + "_" + entry.name;
            row.tag("operator", v->op)
               .tag("matrix", entry.name).tag("format", "csr").tag("dtype", v->dtype)
               .tag("corpus", corpus_tag())
               .num("rows", static_cast<double>(A.rows))
               .num("nnz", static_cast<double>(A.nnz));
            trace("spgemm", entry.name, v->dtype, A);

            DeviceBuffer indptr = DeviceBuffer::from(A.indptr);
            DeviceBuffer indices = DeviceBuffer::from(A.indices);
            DeviceBuffer values = upload_as(A.values, dt);
            DeviceBuffer c_ptr(static_cast<std::size_t>(A.rows + 1) * sizeof(int32_t));
            if (!indptr.get() || !indices.get() || !values.get() || !c_ptr.get()) {
                g_report.skip(std::move(row), "skipped_memory",
                              "device allocation failed for this matrix");
                continue;
            }

            const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
            flagsparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;
            flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, indptr.get(),
                                indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dt);
            flagsparseCreateCsr(&matB, A.rows, A.cols, A.nnz, indptr.get(),
                                indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dt);
            flagsparseCreateCsr(&matC, A.rows, A.cols, 0, c_ptr.get(), nullptr, nullptr,
                                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                FLAGSPARSE_INDEX_BASE_ZERO, dt);
            flagsparseSpGEMMDescr_t descr = nullptr;
            flagsparseSpGEMM_createDescr(&descr);
            auto teardown = [&]() {
                flagsparseSpGEMM_destroyDescr(descr);
                flagsparseDestroySpMat(matA); flagsparseDestroySpMat(matB);
                flagsparseDestroySpMat(matC);
            };

            std::size_t b1 = 0, b2 = 0;
            flagsparseSpGEMM_workEstimation(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                            sc.beta(dt), matC, dt,
                                            FLAGSPARSE_SPGEMM_DEFAULT, descr, &b1,
                                            nullptr);
            DeviceBuffer s1(b1 ? b1 : 1);
            flagsparseStatus_t st = flagsparseSpGEMM_workEstimation(
                handle.h, NT, NT, sc.alpha(dt), matA, matB, sc.beta(dt), matC, dt,
                FLAGSPARSE_SPGEMM_DEFAULT, descr, &b1, s1.get());
            if (st == FLAGSPARSE_STATUS_SUCCESS) {
                flagsparseSpGEMM_compute(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                         sc.beta(dt), matC, dt,
                                         FLAGSPARSE_SPGEMM_DEFAULT, descr, &b2, nullptr);
            }
            DeviceBuffer s2(b2 ? b2 : 1);
            if (st == FLAGSPARSE_STATUS_SUCCESS) {
                st = flagsparseSpGEMM_compute(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                              sc.beta(dt), matC, dt,
                                              FLAGSPARSE_SPGEMM_DEFAULT, descr, &b2,
                                              s2.get());
            }
            if (st != FLAGSPARSE_STATUS_SUCCESS) {
                teardown();
                // The hash-table overflow path lands here. It is a capability
                // limit of the C wrapper (the Python side falls back to a chunked
                // ESC that the C API does not reach), so it is recorded with that
                // reason and the sweep continues.
                g_report.skip(std::move(row),
                              st == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported"
                                                                    : "failed",
                              "SpGEMM compute declined this matrix");
                continue;
            }

            int64_t cr = 0, cc = 0, cnnz = 0;
            flagsparseSpMatGetSize(matC, &cr, &cc, &cnnz);
            DeviceBuffer c_ind(static_cast<std::size_t>(cnnz) * sizeof(int32_t));
            DeviceBuffer c_val(static_cast<std::size_t>(cnnz) * elem_bytes(dt));
            if (cnnz > 0 && (!c_ind.get() || !c_val.get())) {
                teardown();
                g_report.skip(std::move(row), "skipped_memory",
                              "C allocation failed (" + std::to_string(cnnz) +
                                  " nonzeros)");
                continue;
            }
            flagsparseCsrSetPointers(matC, c_ptr.get(), c_ind.get(), c_val.get());
            row.num("c_nnz", static_cast<double>(cnnz));

            baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                   A.rows, A.cols, A.nnz, dt};
            g_report.measure_vs_baseline(
                std::move(row),
                [&]() {
                    return flagsparseSpGEMM_copy(handle.h, NT, NT, sc.alpha(dt), matA,
                                                 matB, sc.beta(dt), matC, dt,
                                                 FLAGSPARSE_SPGEMM_DEFAULT, descr);
                },
                [&]() -> double {
                    if (cnnz > kVerifyNnzBudget) return -1.0;  // unchecked
                    CsrMatrix C;
                    C.rows = cr; C.cols = cc; C.nnz = cnnz;
                    C.indptr.resize(static_cast<std::size_t>(cr) + 1);
                    C.indices.resize(static_cast<std::size_t>(cnnz));
                    if (to_host(C.indptr.data(), c_ptr.get(),
                                C.indptr.size() * sizeof(int32_t)) !=
                        FLAGSPARSE_STATUS_SUCCESS) return 1e30;
                    if (cnnz > 0 &&
                        to_host(C.indices.data(), c_ind.get(),
                                C.indices.size() * sizeof(int32_t)) !=
                            FLAGSPARSE_STATUS_SUCCESS) return 1e30;
                    C.values = read_back(c_val.get(), static_cast<std::size_t>(cnnz), dt);
                    if (cnnz > 0 && C.values.empty()) return 1e30;
                    return max_error_ratio(host_spmv(C, x), ref, default_tolerance(dt));
                },
                [&](baseline::Timing* t) {
                    return baseline::spgemm_csr(bA, sc.alpha(dt), sc.beta(dt),
                                                BenchReport::kWarmup,
                                                BenchReport::kIters, t);
                },
                0.0);
            teardown();
        }
    }
    EXPECT_GT(g_report.size(), 0u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    g_report.write();
    return rc;
}
