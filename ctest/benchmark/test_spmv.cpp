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


// Performance test for flagsparseSpMV across all four sparse formats. Timing
// follows spec §6.4: >=10 warmup iterations, >=100 timed, median reported, JSON
// on disk.
//
// The formats are NOT interchangeable measurements: CSR and COO are row-parallel
// and deterministic, CSC's non-transposed direction and both BSR directions
// scatter with atomics and pay a beta prologue on top. The JSON carries the
// format and direction so a row is never read as a like-for-like comparison
// when it is not one.

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

BenchReport g_report("spmv");

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

struct CscArrays { std::vector<int32_t> colptr, rowind; std::vector<double> values; };

CscArrays csr_to_csc(const CsrMatrix& A) {
    CscArrays C;
    C.colptr.assign(static_cast<std::size_t>(A.cols) + 1, 0);
    for (int32_t c : A.indices) C.colptr[static_cast<std::size_t>(c) + 1]++;
    for (int64_t c = 0; c < A.cols; ++c) {
        C.colptr[static_cast<std::size_t>(c) + 1] += C.colptr[static_cast<std::size_t>(c)];
    }
    C.rowind.assign(static_cast<std::size_t>(A.nnz), 0);
    C.values.assign(static_cast<std::size_t>(A.nnz), 0.0);
    std::vector<int32_t> cursor(C.colptr.begin(), C.colptr.end() - 1);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const std::size_t slot =
                static_cast<std::size_t>(cursor[static_cast<std::size_t>(A.indices[p])]++);
            C.rowind[slot] = static_cast<int32_t>(r);
            C.values[slot] = A.values[static_cast<std::size_t>(p)];
        }
    }
    return C;
}

const char* format_name(flagsparseFormat_t f) {
    switch (f) {
        case FLAGSPARSE_FORMAT_COO: return "coo";
        case FLAGSPARSE_FORMAT_CSC: return "csc";
        case FLAGSPARSE_FORMAT_BSR: return "bsr";
        default:                    return "csr";
    }
}

const char* op_name(flagsparseOperation_t op) {
    return (op == FLAGSPARSE_OPERATION_NON_TRANSPOSE) ? "non" : "trans";
}

// One configuration, timed. Returns false when the operator refuses it, which
// is reported rather than silently recorded as a zero.
bool bench_one(flagsparseHandle_t handle, const char* name, int64_t rows,
               int64_t cols, double density,
               flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR,
               flagsparseOperation_t opA = FLAGSPARSE_OPERATION_NON_TRANSPOSE,
               flagsparseSpMVAlg_t alg = FLAGSPARSE_SPMV_ALG_DEFAULT) {
    const CsrMatrix A = random_csr(rows, cols, density, 4242);
    const bool trans = (opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t n_x = trans ? rows : cols;
    const int64_t n_y = trans ? cols : rows;

    const CscArrays csc = csr_to_csc(A);
    const std::vector<double>& vals64 =
        (format == FLAGSPARSE_FORMAT_CSC) ? csc.values : A.values;
    const std::vector<float> values(vals64.begin(), vals64.end());
    const std::vector<float> x(static_cast<std::size_t>(n_x), 1.0f);
    const std::vector<float> y(static_cast<std::size_t>(n_y), 0.0f);

    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_ptr = DeviceBuffer::from(
        format == FLAGSPARSE_FORMAT_CSC ? csc.colptr : A.indptr);
    DeviceBuffer d_idx = DeviceBuffer::from(
        format == FLAGSPARSE_FORMAT_CSC ? csc.rowind : A.indices);
    DeviceBuffer d_row = DeviceBuffer::from(coo_row_indices(A));
    DeviceBuffer d_x   = DeviceBuffer::from(x);
    DeviceBuffer d_y   = DeviceBuffer::from(y);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseStatus_t created;
    switch (format) {
        case FLAGSPARSE_FORMAT_COO:
            created = flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, d_row.get(),
                                          d_idx.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
            break;
        case FLAGSPARSE_FORMAT_CSC:
            created = flagsparseCreateCsc(&matA, A.rows, A.cols, A.nnz, d_ptr.get(),
                                          d_idx.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
            break;
        default:
            created = flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_ptr.get(),
                                          d_idx.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
            break;
    }
    if (created != FLAGSPARSE_STATUS_SUCCESS) return false;
    flagsparseCreateDnVec(&vecX, n_x, d_x.get(), FLAGSPARSE_R_32F);
    flagsparseCreateDnVec(&vecY, n_y, d_y.get(), FLAGSPARSE_R_32F);

    const float alpha = 1.0f, beta = 0.0f;
    size_t buffer_size = 0;
    flagsparseSpMV_bufferSize(handle, opA, &alpha, matA, vecX, &beta, vecY,
                              FLAGSPARSE_R_32F, alg, &buffer_size);
    DeviceBuffer scratch(buffer_size);
    // Preprocess outside the clock: it does the index readback once, which is
    // exactly what cuSPARSE's preprocess step is for. For COO that readback also
    // builds the row-offsets array, so timing it would charge a per-matrix setup
    // to every iteration and make COO look far worse than it is.
    flagsparseSpMV_preprocess(handle, opA, &alpha, matA, vecX, &beta, vecY,
                              FLAGSPARSE_R_32F, alg, scratch.get());

    BenchRow row;
    row.name = name;
    row.tag("format", format_name(format))
       .tag("op", op_name(opA))
       .tag("dtype", "float32")
       .num("rows", static_cast<double>(rows))
       .num("cols", static_cast<double>(cols))
       .num("nnz", static_cast<double>(A.nnz))
       .num("density", density);

    const bool measured = g_report.measure(
        row,
        [&] {
            return flagsparseSpMV(handle, opA, &alpha, matA, vecX, &beta, vecY,
                                  FLAGSPARSE_R_32F, alg, scratch.get());
        },
        // 2 flops per nonzero (one multiply, one add).
        2.0 * static_cast<double>(A.nnz));

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return measured;
}

class SpMVBenchmark : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

// Spec §6.4 asks for small / medium / large and a density sweep.
TEST_F(SpMVBenchmark, SizeSweep) {
    EXPECT_TRUE(bench_one(handle.h, "small",  512,   512,   0.01));
    EXPECT_TRUE(bench_one(handle.h, "medium", 8192,  8192,  0.001));
    EXPECT_TRUE(bench_one(handle.h, "large",  32768, 32768, 0.0001));
}

TEST_F(SpMVBenchmark, DensitySweep) {
    for (double d : {0.0001, 0.001, 0.01, 0.1}) {
        std::ostringstream name;
        name << "density_" << d;
        EXPECT_TRUE(bench_one(handle.h, name.str().c_str(), 4096, 4096, d));
    }
}

// The same shapes in COO, so the two row-parallel formats sit next to each other
// in one JSON. COO also carries a second route: COO_ALG2 runs the CSR kernel
// over the offsets it had to build anyway, so the gap between the two rows is
// the segment kernel's own cost, not a format difference.
TEST_F(SpMVBenchmark, CooRoutes) {
    for (double d : {0.001, 0.01, 0.1}) {
        std::ostringstream seg, tocsr;
        seg << "coo_seg_density_" << d;
        tocsr << "coo_tocsr_density_" << d;
        EXPECT_TRUE(bench_one(handle.h, seg.str().c_str(), 4096, 4096, d,
                              FLAGSPARSE_FORMAT_COO,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                              FLAGSPARSE_SPMV_COO_ALG1));
        EXPECT_TRUE(bench_one(handle.h, tocsr.str().c_str(), 4096, 4096, d,
                              FLAGSPARSE_FORMAT_COO,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                              FLAGSPARSE_SPMV_COO_ALG2));
    }
}

// BSR needs real block structure, so it is built here rather than derived from a
// scalar CSR. Both of its directions are atomic and both pay the beta prologue;
// what the block dimension buys is fewer indices per nonzero, which is the axis
// worth sweeping.
bool bench_bsr(flagsparseHandle_t handle, const char* name, int64_t brows,
               int64_t bcols, int64_t block_dim, double block_density,
               flagsparseOperation_t opA) {
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> value(0.0, 1.0);
    std::vector<int32_t> indptr(static_cast<std::size_t>(brows) + 1, 0), indices;
    std::vector<float> data;
    for (int64_t br = 0; br < brows; ++br) {
        for (int64_t bc = 0; bc < bcols; ++bc) {
            if (unit(rng) >= block_density) continue;
            indices.push_back(static_cast<int32_t>(bc));
            for (int64_t i = 0; i < block_dim * block_dim; ++i) {
                data.push_back(static_cast<float>(value(rng)));
            }
        }
        indptr[static_cast<std::size_t>(br) + 1] = static_cast<int32_t>(indices.size());
    }
    const int64_t bnnz = static_cast<int64_t>(indices.size());
    if (bnnz == 0) return false;
    const bool trans = (opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t n_x = (trans ? brows : bcols) * block_dim;
    const int64_t n_y = (trans ? bcols : brows) * block_dim;

    DeviceBuffer d_val = DeviceBuffer::from(data);
    DeviceBuffer d_ptr = DeviceBuffer::from(indptr);
    DeviceBuffer d_idx = DeviceBuffer::from(indices);
    DeviceBuffer d_x   = DeviceBuffer::from(std::vector<float>(n_x, 1.0f));
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<float>(n_y, 0.0f));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    if (flagsparseCreateBsr(&matA, brows, bcols, bnnz, block_dim, block_dim,
                            d_ptr.get(), d_idx.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                            FLAGSPARSE_R_32F, FLAGSPARSE_ORDER_ROW)
        != FLAGSPARSE_STATUS_SUCCESS) {
        return false;
    }
    flagsparseCreateDnVec(&vecX, n_x, d_x.get(), FLAGSPARSE_R_32F);
    flagsparseCreateDnVec(&vecY, n_y, d_y.get(), FLAGSPARSE_R_32F);

    const float alpha = 1.0f, beta = 0.0f;
    flagsparseSpMV_preprocess(handle, opA, &alpha, matA, vecX, &beta, vecY,
                              FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr);
    BenchRow row;
    row.name = name;
    // Every stored block is dense, so the scalar nonzero count is the block
    // count times the block area -- quoting bnnz would understate the work by
    // block_dim^2 and make BSR look artificially fast.
    const int64_t scalar_nnz = bnnz * block_dim * block_dim;
    row.tag("format", "bsr")
       .tag("op", op_name(opA))
       .tag("dtype", "float32")
       .num("rows", static_cast<double>(brows * block_dim))
       .num("cols", static_cast<double>(bcols * block_dim))
       .num("block_dim", static_cast<double>(block_dim))
       .num("nnz", static_cast<double>(scalar_nnz))
       .num("density", static_cast<double>(scalar_nnz) /
                       static_cast<double>(brows * block_dim * bcols * block_dim));

    const bool measured = g_report.measure(
        row,
        [&] {
            return flagsparseSpMV(handle, opA, &alpha, matA, vecX, &beta, vecY,
                                  FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT,
                                  nullptr);
        },
        2.0 * static_cast<double>(scalar_nnz));

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return measured;
}

TEST_F(SpMVBenchmark, BsrBlockDimSweep) {
    for (int64_t bd : {2, 4, 8, 16}) {
        std::ostringstream non, tr;
        non << "bsr_bd" << bd << "_non";
        tr  << "bsr_bd" << bd << "_trans";
        const int64_t blocks = 4096 / bd;
        EXPECT_TRUE(bench_bsr(handle.h, non.str().c_str(), blocks, blocks, bd, 0.01,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE));
        EXPECT_TRUE(bench_bsr(handle.h, tr.str().c_str(), blocks, blocks, bd, 0.01,
                              FLAGSPARSE_OPERATION_TRANSPOSE));
    }
}

// CSC's two directions are structurally different work, not a symmetric pair:
// trans is one program per column and deterministic, non scatters with atomics
// and pays a beta prologue first. Reading one as a proxy for the other is the
// mistake this pair of rows exists to prevent.
TEST_F(SpMVBenchmark, CscBothDirections) {
    for (double d : {0.001, 0.01}) {
        std::ostringstream non, tr;
        non << "csc_non_density_" << d;
        tr  << "csc_trans_density_" << d;
        EXPECT_TRUE(bench_one(handle.h, non.str().c_str(), 4096, 4096, d,
                              FLAGSPARSE_FORMAT_CSC,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE));
        EXPECT_TRUE(bench_one(handle.h, tr.str().c_str(), 4096, 4096, d,
                              FLAGSPARSE_FORMAT_CSC,
                              FLAGSPARSE_OPERATION_TRANSPOSE));
    }
    g_report.write();
}

}  // namespace
