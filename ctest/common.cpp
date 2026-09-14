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


#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

// The adaptor is internal, but the tests need device memory and the whole point
// of routing through it is that they stay vendor-neutral.
#include "adaptor/adaptor.hpp"

namespace fstest {

namespace ad = flagsparse::adaptor;

flagsparseStatus_t dev_alloc(void** ptr, std::size_t bytes) {
    ad::DevicePtr p = 0;
    const flagsparseStatus_t st = ad::device_malloc(&p, bytes);
    *ptr = reinterpret_cast<void*>(p);
    return st;
}
void dev_free(void* ptr) { ad::device_free(reinterpret_cast<ad::DevicePtr>(ptr)); }
flagsparseStatus_t to_device(void* dst, const void* src, std::size_t bytes) {
    return ad::memcpy_h2d(reinterpret_cast<ad::DevicePtr>(dst), src, bytes);
}
flagsparseStatus_t to_host(void* dst, const void* src, std::size_t bytes) {
    return ad::memcpy_d2h(dst, reinterpret_cast<ad::DevicePtr>(src), bytes);
}
void dev_sync() { ad::synchronize(); }

CsrMatrix random_csr(int64_t rows, int64_t cols, double density, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> value(0.0, 1.0);

    CsrMatrix A;
    A.rows = rows; A.cols = cols;
    A.indptr.assign(static_cast<std::size_t>(rows) + 1, 0);
    for (int64_t r = 0; r < rows; ++r) {
        // Row-to-row variation on purpose: a uniform nnz per row would hide a
        // segment count sized from the average instead of the maximum.
        const double row_density = density * (0.25 + 1.5 * unit(rng));
        for (int64_t c = 0; c < cols; ++c) {
            if (unit(rng) < row_density) {
                A.indices.push_back(static_cast<int32_t>(c));
                A.values.push_back(value(rng));
            }
        }
        A.indptr[static_cast<std::size_t>(r) + 1] = static_cast<int32_t>(A.indices.size());
    }
    A.nnz = static_cast<int64_t>(A.indices.size());
    return A;
}

TriMatrix random_triangular(int64_t n, double density, bool lower, bool unit_diag,
                            uint32_t seed) {
    TriMatrix T;
    T.n = n; T.lower = lower; T.unit_diag = unit_diag;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> value(0.0, 1.0);
    const double expected = std::max(1.0, density * static_cast<double>(n));
    T.indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int64_t r = 0; r < n; ++r) {
        const int64_t begin = lower ? 0 : r;
        const int64_t end   = lower ? r + 1 : n;
        for (int64_t c = begin; c < end; ++c) {
            if (c == r) {
                // A unit diagonal is implicit: cuSPARSE does not store it.
                if (unit_diag) continue;
                T.indices.push_back(static_cast<int32_t>(c));
                T.values.push_back(2.0 + std::abs(value(rng)));
                continue;
            }
            if (unit(rng) >= density) continue;
            T.indices.push_back(static_cast<int32_t>(c));
            T.values.push_back(value(rng) / (4.0 * expected));
        }
        T.indptr[static_cast<std::size_t>(r) + 1] = static_cast<int32_t>(T.indices.size());
    }
    return T;
}

namespace {
template <typename M>
std::vector<int32_t> expand_rows(const M& m, int64_t rows) {
    std::vector<int32_t> row;
    row.reserve(static_cast<std::size_t>(m.indices.size()));
    for (int64_t r = 0; r < rows; ++r) {
        for (int32_t p = m.indptr[static_cast<std::size_t>(r)];
             p < m.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            row.push_back(static_cast<int32_t>(r));
        }
    }
    return row;
}
}  // namespace

std::vector<int32_t> coo_row_indices_of(const CsrMatrix& A) {
    return expand_rows(A, A.rows);
}
std::vector<int32_t> coo_row_indices_of(const TriMatrix& T) {
    return expand_rows(T, T.n);
}

std::vector<double> spmv_reference(const CsrMatrix& A, const std::vector<double>& x,
                                   double alpha, double beta,
                                   const std::vector<double>& y_in) {
    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        double acc = 0.0;
        for (int32_t k = A.indptr[static_cast<std::size_t>(r)];
             k < A.indptr[static_cast<std::size_t>(r) + 1]; ++k) {
            acc += A.values[static_cast<std::size_t>(k)] *
                   x[static_cast<std::size_t>(A.indices[static_cast<std::size_t>(k)])];
        }
        // beta == 0 must ignore y entirely, matching cuSPARSE: an uninitialised
        // y would otherwise turn into NaN through 0 * NaN.
        y[static_cast<std::size_t>(r)] =
            alpha * acc + (beta == 0.0 ? 0.0 : beta * y_in[static_cast<std::size_t>(r)]);
    }
    return y;
}

std::vector<double> spmm_reference(const CsrMatrix& A, const std::vector<double>& B,
                                   int64_t n, double alpha, double beta,
                                   const std::vector<double>& c_in) {
    std::vector<double> C(static_cast<std::size_t>(A.rows * n), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
                 p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
                const int64_t col = A.indices[static_cast<std::size_t>(p)];
                acc += A.values[static_cast<std::size_t>(p)] *
                       B[static_cast<std::size_t>(col * n + j)];
            }
            const std::size_t idx = static_cast<std::size_t>(r * n + j);
            // beta == 0 must ignore C entirely, matching cuSPARSE.
            C[idx] = alpha * acc + (beta == 0.0 ? 0.0 : beta * c_in[idx]);
        }
    }
    return C;
}

Tolerance default_tolerance(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_R_16F:
        case FLAGSPARSE_R_16BF: return {1e-2, 1e-3};
        case FLAGSPARSE_R_32F:
        case FLAGSPARSE_C_32F:  return {1e-5, 1e-6};
        case FLAGSPARSE_R_64F:
        case FLAGSPARSE_C_64F:  return {1e-12, 1e-13};
        default:                return {1e-5, 1e-6};
    }
}

// Spec §6.3.1: fp32 sparse accumulation is order-dependent, so a matrix that
// fails at 1e-5 is not necessarily wrong. The caller retries here and reports
// PASS(relaxed) rather than turning a known numerical property into a failure.
Tolerance relaxed_tolerance(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_R_64F:
        case FLAGSPARSE_C_64F: return {1e-10, 1e-11};
        default:               return {1e-3, 1e-4};
    }
}

double max_error_ratio(const std::vector<double>& actual, const std::vector<double>& ref,
                       Tolerance tol) {
    if (actual.size() != ref.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(actual[i])) return std::numeric_limits<double>::infinity();
        const double allow = tol.atol + tol.rtol * std::abs(ref[i]);
        worst = std::max(worst, std::abs(actual[i] - ref[i]) / allow);
    }
    return worst;
}

void print_backend_banner() {
    std::cout << "[  BACKEND  ] " << flagsparseGetBackendName()
              << "  flagsparse " << FLAGSPARSE_VERSION << std::endl;
}

std::string device_arch() {
    return ad::device_architecture(0);
}

bool BenchReport::measure(BenchRow row, const std::function<flagsparseStatus_t()>& once,
                          double flops) {
    // The first call also pays JIT compilation; it decides the status and stays
    // out of the samples.
    const flagsparseStatus_t first = once();
    if (first == FLAGSPARSE_STATUS_NOT_SUPPORTED) {
        row.status = "not_supported";
        row.detail = "operator declined this configuration on this backend";
        rows_.push_back(std::move(row));
        return false;
    }
    if (first != FLAGSPARSE_STATUS_SUCCESS) {
        row.status = "failed";
        row.detail = status_name(first);
        rows_.push_back(std::move(row));
        return false;
    }
    for (int i = 0; i < kWarmup; ++i) once();
    dev_sync();

    std::vector<double> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        once();
        dev_sync();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    row.status = "ok";
    row.median_ms = samples[samples.size() / 2];
    row.gflops = (flops > 0.0 && row.median_ms > 0.0)
                     ? flops / (row.median_ms * 1e6)
                     : 0.0;
    rows_.push_back(std::move(row));
    return true;
}

void BenchReport::skip(BenchRow row, const std::string& status,
                       const std::string& detail) {
    row.status = status;
    row.detail = detail;
    rows_.push_back(std::move(row));
}

void BenchReport::write() const {
    const char* dir = std::getenv("FLAGSPARSE_BENCH_OUT");
    const std::string path =
        std::string(dir ? dir : ".") + "/" + op_ + "_benchmark.json";
    std::ofstream out(path);
    if (!out) return;
    const std::time_t now = std::time(nullptr);
    char stamp[64];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

    out << "{\n  \"timestamp\": \"" << stamp << "\",\n";
    out << "  \"operator\": \"" << op_ << "\",\n";
    // The backend and the device it ran on: without these a row from one chip
    // is indistinguishable from a row from another.
    out << "  \"env\": {\"backend\": \"" << flagsparseGetBackendName()
        << "\", \"arch\": \"" << device_arch()
        << "\", \"version\": " << FLAGSPARSE_VERSION << "},\n";
    out << "  \"config\": {\"warmup\": " << kWarmup << ", \"iters\": " << kIters
        << ", \"statistic\": \"median\"},\n";
    out << "  \"result\": [\n";
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        const BenchRow& r = rows_[i];
        out << "    {\"name\": \"" << r.name << "\", \"status\": \"" << r.status
            << "\"";
        if (!r.detail.empty()) out << ", \"detail\": \"" << r.detail << "\"";
        for (const auto& kv : r.tags) {
            out << ", \"" << kv.first << "\": \"" << kv.second << "\"";
        }
        for (const auto& kv : r.numbers) {
            out << ", \"" << kv.first << "\": " << std::setprecision(10) << kv.second;
        }
        if (r.status == "ok") {
            out << ", \"median_ms\": " << std::setprecision(6) << r.median_ms
                << ", \"gflops\": " << r.gflops;
        }
        // No vendor sparse baseline is wired up on any backend yet, so these are
        // null rather than 1.0 -- a fabricated baseline is worse than none.
        out << ", \"baseline_ms\": null, \"speedup\": null}";
        out << (i + 1 < rows_.size() ? ",\n" : "\n");
    }
    std::size_t ok = 0, unsupported = 0, failed = 0;
    for (const BenchRow& r : rows_) {
        if (r.status == "ok") ++ok;
        else if (r.status == "not_supported") ++unsupported;
        else ++failed;
    }
    out << "  ],\n  \"summary\": {\"rows\": " << rows_.size()
        << ", \"ok\": " << ok << ", \"not_supported\": " << unsupported
        << ", \"failed\": " << failed
        << ", \"baseline\": \"none (vendor sparse baseline not wired up)\"}\n}\n";
    std::cout << "wrote " << path << "  (" << ok << " ok, " << unsupported
              << " not_supported, " << failed << " failed)" << std::endl;
}

const char* status_name(flagsparseStatus_t st) {
    const char* name = nullptr;
    flagsparseGetErrorName(st, &name);
    return name;
}

}  // namespace fstest
