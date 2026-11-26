// Correctness at awkward sizes: tiled/vectorized kernels usually break on
// N that isn't a multiple of the tile or vector width.
#include "matmul.h"
#include "reference.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
    const int sizes[] = {1, 2, 3, 7, 16, 31, 33, 64, 127, 129, 256, 513};
    int failed = 0;
    for (int N : sizes) {
        float* A = alloc_matrix(N);
        float* B = alloc_matrix(N);
        float* C = alloc_matrix(N);
        fill_random(A, N, 42u + N);
        fill_random(B, N, 99u + N);
        std::vector<double> ref, absref;
        reference_matmul(A, B, N, ref, absref);
        std::fill(C, C + (size_t)N * N, std::numeric_limits<float>::quiet_NaN());
        matmul(A, B, C, N);
        CheckResult r = check(C, ref, absref, N);
        std::printf("N=%4d  %s  err/tol=%.4g  bad=%ld\n", N, r.pass ? "PASS" : "FAIL",
                    r.worst_ratio, r.bad_count);
        failed += !r.pass;
        free_matrix(A); free_matrix(B); free_matrix(C);
    }
    std::printf("%s\n", failed ? "FAILED" : "ALL PASS");
    return failed ? 1 : 0;
}
