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


// Performance test for SpGEMM, through the shared harness.
//
// This operator is reported as THREE rows per case, not one, because its three
// phases are different kinds of work and averaging them hides the interesting
// part:
//
//   work_estimation  builds the per-row product counts. Host-side, O(nnz_A).
//   compute          the counting kernel, then a host readback and prefix scan
//                    that discovers C's nnz. The readback is a device sync by
//                    construction -- the size is not knowable without it.
//   copy             the fill kernel, then a per-row sort on the HOST to honour
//                    cuSPARSE's sorted-CSR guarantee.
//
// That last one is a known cost, and splitting the phases is what makes it
// visible rather than smeared into a single number. A device-side segmented sort
// is the obvious improvement, and this is the row that would show it landing.
//
// Each phase is re-timed from a clean descriptor: the flow is stateful, so
// running `copy` a hundred times without redoing `compute` would measure a
// repeat of the last phase rather than the phase itself.

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

BenchReport g_report("spgemm");

struct DeviceCsr {
    DeviceBuffer ptr, col, val;
    flagsparseSpMatDescr_t descr = nullptr;
    ~DeviceCsr() { if (descr) flagsparseDestroySpMat(descr); }
};

bool upload(const CsrMatrix& M, flagsparseDataType_t dtype, DeviceCsr* out) {
    const std::vector<float> values(M.values.begin(), M.values.end());
    out->ptr = DeviceBuffer::from(M.indptr);
    out->col = DeviceBuffer::from(M.indices);
    out->val = DeviceBuffer::from(values);
    return flagsparseCreateCsr(&out->descr, M.rows, M.cols, M.nnz, out->ptr.get(),
                               out->col.get(), out->val.get(), FLAGSPARSE_INDEX_32I,
                               FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dtype)
           == FLAGSPARSE_STATUS_SUCCESS;
}

// Scalar products the product would evaluate -- the honest denominator for a
// GFLOPS column here, since nnz_C counts only what survived accumulation.
double product_count(const CsrMatrix& A, const CsrMatrix& B) {
    double total = 0.0;
    for (int32_t k : A.indices) {
        total += B.indptr[static_cast<std::size_t>(k) + 1] -
                 B.indptr[static_cast<std::size_t>(k)];
    }
    return total;
}

void bench_one(flagsparseHandle_t handle, const char* name, int64_t m, int64_t k,
               int64_t n, double density) {
    const CsrMatrix A = random_csr(m, k, density, 11);
    const CsrMatrix B = random_csr(k, n, density, 13);
    const double products = product_count(A, B);
    const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    const auto ALG = FLAGSPARSE_SPGEMM_DEFAULT;
    const float one = 1.0f, zero = 0.0f;

    const auto base_row = [&](const char* phase) {
        BenchRow row;
        row.name = std::string(name) + "_" + phase;
        row.tag("phase", phase)
           .tag("dtype", "float32")
           .num("m", static_cast<double>(m))
           .num("k", static_cast<double>(k))
           .num("n", static_cast<double>(n))
           .num("nnz_a", static_cast<double>(A.nnz))
           .num("nnz_b", static_cast<double>(B.nnz))
           .num("products", products)
           .num("density", density);
        return row;
    };

    DeviceCsr da, db;
    if (!upload(A, FLAGSPARSE_R_32F, &da) || !upload(B, FLAGSPARSE_R_32F, &db)) {
        g_report.skip(base_row("work_estimation"), "failed", "operand upload");
        return;
    }

    // One full pass first: it decides whether this case runs at all, sizes the
    // buffers, and tells us C's nnz for the report.
    DeviceBuffer c_ptr_probe(static_cast<size_t>(m + 1) * sizeof(int32_t));
    flagsparseSpMatDescr_t matC = nullptr;
    if (flagsparseCreateCsr(&matC, m, n, 0, c_ptr_probe.get(), nullptr, nullptr,
                            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F)
        != FLAGSPARSE_STATUS_SUCCESS) {
        g_report.skip(base_row("work_estimation"), "failed", "C descriptor");
        return;
    }
    flagsparseSpGEMMDescr_t probe = nullptr;
    flagsparseSpGEMM_createDescr(&probe);
    size_t size1 = 0, size2 = 0;
    flagsparseSpGEMM_workEstimation(handle, NT, NT, &one, da.descr, db.descr, &zero,
                                    matC, FLAGSPARSE_R_32F, ALG, probe, &size1, nullptr);
    DeviceBuffer buf1(size1);
    flagsparseStatus_t st = flagsparseSpGEMM_workEstimation(
        handle, NT, NT, &one, da.descr, db.descr, &zero, matC, FLAGSPARSE_R_32F, ALG,
        probe, &size1, buf1.get());
    if (st == FLAGSPARSE_STATUS_SUCCESS) {
        flagsparseSpGEMM_compute(handle, NT, NT, &one, da.descr, db.descr, &zero, matC,
                                 FLAGSPARSE_R_32F, ALG, probe, &size2, nullptr);
    }
    DeviceBuffer buf2(size2);
    if (st == FLAGSPARSE_STATUS_SUCCESS) {
        st = flagsparseSpGEMM_compute(handle, NT, NT, &one, da.descr, db.descr, &zero,
                                      matC, FLAGSPARSE_R_32F, ALG, probe, &size2,
                                      buf2.get());
    }
    if (st != FLAGSPARSE_STATUS_SUCCESS) {
        const char* detail = "";
        flagsparseGetLastErrorString(handle, &detail);
        const char* status = (st == FLAGSPARSE_STATUS_NOT_SUPPORTED) ? "not_supported"
                                                                     : "failed";
        for (const char* phase : {"work_estimation", "compute", "copy"}) {
            g_report.skip(base_row(phase), status,
                          std::string(status_name(st)) + ": " + detail);
        }
        flagsparseSpGEMM_destroyDescr(probe);
        flagsparseDestroySpMat(matC);
        return;
    }
    int64_t c_rows = 0, c_cols = 0, c_nnz = 0;
    flagsparseSpMatGetSize(matC, &c_rows, &c_cols, &c_nnz);
    flagsparseSpGEMM_destroyDescr(probe);
    flagsparseDestroySpMat(matC);

    const auto with_nnz = [&](const char* phase) {
        BenchRow row = base_row(phase);
        row.num("nnz_c", static_cast<double>(c_nnz))
           .num("buffer1_bytes", static_cast<double>(size1))
           .num("buffer2_bytes", static_cast<double>(size2));
        return row;
    };

    // Phase 1: work estimation, from a fresh descriptor each time.
    g_report.measure(with_nnz("work_estimation"), [&]() {
        flagsparseSpGEMMDescr_t d = nullptr;
        flagsparseSpGEMM_createDescr(&d);
        flagsparseSpMatDescr_t c = nullptr;
        flagsparseCreateCsr(&c, m, n, 0, c_ptr_probe.get(), nullptr, nullptr,
                            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
        size_t s1 = 0;
        const flagsparseStatus_t r = flagsparseSpGEMM_workEstimation(
            handle, NT, NT, &one, da.descr, db.descr, &zero, c, FLAGSPARSE_R_32F, ALG,
            d, &s1, buf1.get());
        flagsparseSpGEMM_destroyDescr(d);
        flagsparseDestroySpMat(c);
        return r;
    }, 0.0);

    // Phase 2: the counting kernel plus the readback that discovers nnz.
    g_report.measure(with_nnz("compute"), [&]() {
        flagsparseSpGEMMDescr_t d = nullptr;
        flagsparseSpGEMM_createDescr(&d);
        flagsparseSpMatDescr_t c = nullptr;
        flagsparseCreateCsr(&c, m, n, 0, c_ptr_probe.get(), nullptr, nullptr,
                            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
        size_t s1 = 0, s2 = 0;
        flagsparseSpGEMM_workEstimation(handle, NT, NT, &one, da.descr, db.descr, &zero,
                                        c, FLAGSPARSE_R_32F, ALG, d, &s1, buf1.get());
        const flagsparseStatus_t r = flagsparseSpGEMM_compute(
            handle, NT, NT, &one, da.descr, db.descr, &zero, c, FLAGSPARSE_R_32F, ALG,
            d, &s2, buf2.get());
        flagsparseSpGEMM_destroyDescr(d);
        flagsparseDestroySpMat(c);
        return r;
    }, products);

    // Phase 3: the fill kernel plus the host-side per-row sort.
    DeviceBuffer c_ptr(static_cast<size_t>(m + 1) * sizeof(int32_t));
    DeviceBuffer c_col(static_cast<size_t>(std::max<int64_t>(c_nnz, 1)) * sizeof(int32_t));
    DeviceBuffer c_val(static_cast<size_t>(std::max<int64_t>(c_nnz, 1)) * sizeof(float));
    g_report.measure(with_nnz("copy"), [&]() {
        flagsparseSpGEMMDescr_t d = nullptr;
        flagsparseSpGEMM_createDescr(&d);
        flagsparseSpMatDescr_t c = nullptr;
        flagsparseCreateCsr(&c, m, n, 0, c_ptr_probe.get(), nullptr, nullptr,
                            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
        size_t s1 = 0, s2 = 0;
        flagsparseSpGEMM_workEstimation(handle, NT, NT, &one, da.descr, db.descr, &zero,
                                        c, FLAGSPARSE_R_32F, ALG, d, &s1, buf1.get());
        flagsparseSpGEMM_compute(handle, NT, NT, &one, da.descr, db.descr, &zero, c,
                                 FLAGSPARSE_R_32F, ALG, d, &s2, buf2.get());
        flagsparseCsrSetPointers(c, c_ptr.get(), c_col.get(), c_val.get());
        const flagsparseStatus_t r = flagsparseSpGEMM_copy(
            handle, NT, NT, &one, da.descr, db.descr, &zero, c, FLAGSPARSE_R_32F, ALG, d);
        flagsparseSpGEMM_destroyDescr(d);
        flagsparseDestroySpMat(c);
        return r;
    }, 0.0);
}

class SpGEMMBenchmark : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(SpGEMMBenchmark, SizeSweep) {
    bench_one(handle.h, "small",  512,  512,  512,  0.02);
    bench_one(handle.h, "medium", 2048, 2048, 2048, 0.005);
    bench_one(handle.h, "large",  4096, 4096, 4096, 0.002);
}

// Density decides how many products land in one row's hash table, and past
// 0.75 * 8192 a row cannot be served at all -- the sweep is where that ceiling
// turns from a number in the source into a measured NOT_SUPPORTED row.
TEST_F(SpGEMMBenchmark, DensitySweep) {
    for (double d : {0.002, 0.01, 0.05, 0.2}) {
        std::ostringstream name;
        name << "density_" << d;
        bench_one(handle.h, name.str().c_str(), 1024, 1024, 1024, d);
    }
    g_report.write();
}

}  // namespace
