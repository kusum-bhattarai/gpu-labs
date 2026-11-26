#include "reference.h"

#include <omp.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <random>

float* alloc_matrix(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    bytes = (bytes + 127) / 128 * 128;  // aligned_alloc requires a multiple of alignment
    return static_cast<float*>(std::aligned_alloc(128, bytes));
}

void free_matrix(float* p) { std::free(p); }

void fill_random(float* M, int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (size_t i = 0; i < (size_t)N * N; ++i) M[i] = d(rng);
}

void reference_matmul(const float* A, const float* B, int N,
                      std::vector<double>& ref, std::vector<double>& absref) {
    ref.assign((size_t)N * N, 0.0);
    absref.assign((size_t)N * N, 0.0);
    // Naive i-j-k, rows split across ALL cores (ignoring OMP_NUM_THREADS, which
    // controls the kernel under test) only so N=2048 finishes in seconds.
    #pragma omp parallel for schedule(static) num_threads(omp_get_num_procs())
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double s = 0.0, sa = 0.0;
            for (int k = 0; k < N; ++k) {
                double a = A[(size_t)i * N + k], b = B[(size_t)k * N + j];
                s  += a * b;
                sa += std::fabs(a) * std::fabs(b);
            }
            ref[(size_t)i * N + j] = s;
            absref[(size_t)i * N + j] = sa;
        }
    }
}

CheckResult check(const float* C, const std::vector<double>& ref,
                  const std::vector<double>& absref, int N) {
    CheckResult r{true, 0.0, 0.0, 0, -1, -1};
    const double scale = (double)N * FLT_EPSILON;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            size_t idx = (size_t)i * N + j;
            double err = std::fabs((double)C[idx] - ref[idx]);
            double tol = scale * absref[idx] + 1e-30;
            double ratio = err / tol;
            if (!(err <= tol)) {  // also catches NaN
                if (r.bad_count == 0) { r.first_bad_i = i; r.first_bad_j = j; }
                r.bad_count++;
                r.pass = false;
                if (std::isnan(err)) { r.max_abs_err = NAN; r.worst_ratio = INFINITY; continue; }
            }
            if (!std::isnan(r.max_abs_err) && err > r.max_abs_err) r.max_abs_err = err;
            if (ratio > r.worst_ratio) r.worst_ratio = ratio;
        }
    }
    return r;
}
