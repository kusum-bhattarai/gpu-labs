#pragma once

// C = A * B for square N x N matrices, row-major, float32.
// Element (i, j) of a matrix M lives at M[i * N + j].
// C is OUTPUT ONLY: overwrite it, don't accumulate into it
// (the harness fills C with NaN before calling you, so unwritten elements fail).
void matmul(const float* A, const float* B, float* C, int N);
