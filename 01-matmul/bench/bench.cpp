// Benchmark harness for matmul(A, B, C, N).
//
//   ./build/bench --version naive [--sizes 512,1024,2048] [--runs 5] [--csv path]
//
// Per N: correctness check first (also serves as the cold warmup). If it fails,
// nothing is timed. Otherwise 1 more untimed warmup, then --runs timed runs; the
// median is reported. Every N appends one row to results/results.csv.

#include "matmul.h"
#include "reference.h"

#include <omp.h>
#include <pthread.h>
#include <sys/qos.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifndef MATMUL_ROOT
#define MATMUL_ROOT "."
#endif

// ---- Machine model: Apple M3 Pro (5P + 6E). ESTIMATES, verify before trusting. ----
// P-core: ~4.05 GHz, 4 x 128-bit NEON FMA pipes -> 4 pipes * 4 lanes * 2 flop = 32 flop/cycle.
// E-core: ~2.75 GHz, assumed 2 pipes -> 16 flop/cycle.
// Unified memory: ~150 GB/s.
// NOTE: this is the NEON roofline. Accelerate/MLX-cpu use the AMX matrix coprocessor
// and measured ~1.5 TFLOPS here (>100% of this "peak") - they are on a different roofline.
static constexpr int    kPCores           = 5;
static constexpr double kPeakGflopsPCore  = 4.05 * 32;   // ~129.6
static constexpr double kPeakGflopsECore  = 2.75 * 16;   // ~44
static constexpr double kMemBwGBs         = 150.0;

static double peak_gflops_est(int threads) {
    int p = std::min(threads, kPCores);
    int e = std::max(0, threads - kPCores);
    return p * kPeakGflopsPCore + e * kPeakGflopsECore;
}

static std::string run_cmd(const std::string& cmd) {
    std::string out;
    if (FILE* f = popen(cmd.c_str(), "r")) {
        char buf[256];
        while (fgets(buf, sizeof buf, f)) out += buf;
        pclose(f);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

static std::string git_sha() {
    std::string root = MATMUL_ROOT;
    std::string sha = run_cmd("git -C '" + root + "' rev-parse --short HEAD 2>/dev/null");
    if (sha.empty()) return "nogit";
    std::string dirty = run_cmd("git -C '" + root + "' status --porcelain -- . 2>/dev/null");
    return dirty.empty() ? sha : sha + "-dirty";
}

static std::string now_iso() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

static const char* kCsvHeader =
    "timestamp,git_sha,version,N,threads,runs,median_ms,min_ms,gflops,"
    "pct_peak_est,gbps_min_traffic,max_abs_err,err_over_tol,pass\n";

static FILE* open_csv(const std::string& path) {
    bool exists = false;
    if (FILE* f = std::fopen(path.c_str(), "r")) {
        std::fseek(f, 0, SEEK_END);
        exists = std::ftell(f) > 0;
        std::fclose(f);
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) { std::perror(path.c_str()); std::exit(2); }
    if (!exists) std::fputs(kCsvHeader, f);
    return f;
}

static void usage() {
    std::fprintf(stderr,
        "usage: bench --version LABEL [--sizes 512,1024,2048] [--runs 5] [--csv PATH]\n"
        "  threads: set OMP_NUM_THREADS (default: all cores)\n");
    std::exit(2);
}

int main(int argc, char** argv) {
    std::string version;
    std::vector<int> sizes = {512, 1024, 2048};
    int runs = 5;
    std::string csv_path = std::string(MATMUL_ROOT) + "/results/results.csv";

    for (int a = 1; a < argc; ++a) {
        auto next = [&]() -> std::string { if (a + 1 >= argc) usage(); return argv[++a]; };
        if (!std::strcmp(argv[a], "--version")) version = next();
        else if (!std::strcmp(argv[a], "--runs")) runs = std::atoi(next().c_str());
        else if (!std::strcmp(argv[a], "--csv")) csv_path = next();
        else if (!std::strcmp(argv[a], "--sizes")) {
            sizes.clear();
            std::stringstream ss(next());
            for (std::string tok; std::getline(ss, tok, ',');) sizes.push_back(std::atoi(tok.c_str()));
        } else usage();
    }
    if (version.empty() || version.find_first_of(",\"\n") != std::string::npos || runs < 1) usage();

    // macOS won't let us pin to a core; highest QoS makes the scheduler prefer P-cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    const int threads = omp_get_max_threads();
    const std::string sha = git_sha();
    FILE* csv = open_csv(csv_path);

    std::printf("version=%s  sha=%s  threads=%d  runs=%d  peak_est=%.0f GFLOPS\n",
                version.c_str(), sha.c_str(), threads, runs, peak_gflops_est(threads));
    std::printf("%6s %10s %10s %9s %8s %8s %11s  %s\n",
                "N", "median_ms", "min_ms", "GFLOPS", "%peak", "GB/s", "err/tol", "check");

    int failures = 0;
    for (int N : sizes) {
        float* A = alloc_matrix(N);
        float* B = alloc_matrix(N);
        float* C = alloc_matrix(N);
        fill_random(A, N, 1234u + N);
        fill_random(B, N, 5678u + N);

        std::vector<double> ref, absref;
        reference_matmul(A, B, N, ref, absref);

        // Correctness (doubles as cold warmup). NaN-poison C so unwritten elements fail.
        std::fill(C, C + (size_t)N * N, std::numeric_limits<float>::quiet_NaN());
        matmul(A, B, C, N);
        CheckResult cr = check(C, ref, absref, N);

        if (!cr.pass) {
            ++failures;
            std::printf("%6d %10s %10s %9s %8s %8s %11s  FAIL (%ld bad, first at [%d,%d] got %g want %g) - not timed\n",
                        N, "-", "-", "-", "-", "-", "-", cr.bad_count, cr.first_bad_i, cr.first_bad_j,
                        (double)C[(size_t)cr.first_bad_i * N + cr.first_bad_j],
                        ref[(size_t)cr.first_bad_i * N + cr.first_bad_j]);
            std::fprintf(csv, "%s,%s,%s,%d,%d,%d,,,,,,%g,%g,0\n",
                         now_iso().c_str(), sha.c_str(), version.c_str(), N, threads, runs,
                         cr.max_abs_err, cr.worst_ratio);
            std::fflush(csv);
            free_matrix(A); free_matrix(B); free_matrix(C);
            continue;
        }

        matmul(A, B, C, N);  // warm warmup

        std::vector<double> ms(runs);
        for (int r = 0; r < runs; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            matmul(A, B, C, N);
            auto t1 = std::chrono::steady_clock::now();
            ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::vector<double> sorted = ms;
        std::sort(sorted.begin(), sorted.end());
        double median = (runs % 2) ? sorted[runs / 2]
                                   : 0.5 * (sorted[runs / 2 - 1] + sorted[runs / 2]);
        double minv = sorted.front();

        double flops  = 2.0 * N * (double)N * N;
        double gflops = flops / (median * 1e-3) / 1e9;
        double pct    = 100.0 * gflops / peak_gflops_est(threads);
        // Compulsory traffic only: read A, B once, write C once. A lower bound on bytes moved.
        double gbps   = 3.0 * N * (double)N * sizeof(float) / (median * 1e-3) / 1e9;

        std::printf("%6d %10.3f %10.3f %9.2f %7.1f%% %8.2f %11.4f  ok\n",
                    N, median, minv, gflops, pct, gbps, cr.worst_ratio);
        std::fprintf(csv, "%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.3f,%.2f,%.3f,%g,%g,1\n",
                     now_iso().c_str(), sha.c_str(), version.c_str(), N, threads, runs,
                     median, minv, gflops, pct, gbps, cr.max_abs_err, cr.worst_ratio);
        std::fflush(csv);

        free_matrix(A); free_matrix(B); free_matrix(C);
    }
    std::fclose(csv);
    std::printf("roofline refs (est): %.0f GFLOPS/P-core, %.0f GB/s DRAM -> appended to %s\n",
                kPeakGflopsPCore, kMemBwGBs, csv_path.c_str());
    return failures ? 1 : 0;
}
