#pragma once
#include <cstddef>
#include <vector>

// 128-byte aligned (Apple M-series L1/L2 cache line is 128 B).
float* alloc_matrix(int N);
void free_matrix(float* p);

// Uniform [-1, 1], deterministic for a given seed.
void fill_random(float* M, int N, unsigned seed);

// Naive triple-loop reference, accumulated in double.
// ref[i,j]    = sum_k A[i,k] * B[k,j]
// absref[i,j] = sum_k |A[i,k]| * |B[k,j]|   (used for the error bound)
void reference_matmul(const float* A, const float* B, int N,
                      std::vector<double>& ref, std::vector<double>& absref);

struct CheckResult {
    bool pass;
    double max_abs_err;     // max |C - ref|
    double worst_ratio;     // max |C - ref| / tol  (pass iff < 1)
    long bad_count;         // elements over tolerance (NaN counts as bad)
    int first_bad_i, first_bad_j;
};

// Forward-error bound for a length-N float dot product:
//   |C - ref| <= gamma_N * (|A||B|)_ij,  gamma_N ~= N * u,  u = FLT_EPSILON/2.
// We use N * FLT_EPSILON (2x slack). Any summation order passes; indexing bugs don't.
CheckResult check(const float* C, const std::vector<double>& ref,
                  const std::vector<double>& absref, int N);
