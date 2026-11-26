#!/usr/bin/env python3
"""MLX float32 matmul baseline, same protocol and CSV schema as ./build/bench.

    .venv/bin/python scripts/mlx_baseline.py [--device gpu|cpu] [--sizes 512,1024,2048] [--runs 5]

MLX is lazy: mx.matmul only builds a graph. mx.eval() is what launches the work
and waits for it, so it must sit inside the timed region.
"""
import argparse
import csv
import datetime as dt
import pathlib
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "results.csv"
HEADER = ["timestamp", "git_sha", "version", "N", "threads", "runs", "median_ms", "min_ms",
          "gflops", "pct_peak_est", "gbps_min_traffic", "max_abs_err", "err_over_tol", "pass"]

# ESTIMATES - verify. M3 Pro 14-core GPU: 14 cores * 128 FP32 ALUs * 2 flop/FMA * ~1.4 GHz.
PEAK_GFLOPS_EST = {"gpu": 14 * 128 * 2 * 1.4, "cpu": None}


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "."],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--sizes", default="512,1024,2048")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--csv", default=str(CSV_PATH))
    args = ap.parse_args()

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    version = f"mlx-{args.device}"
    sha = git_sha()
    peak = PEAK_GFLOPS_EST[args.device]

    path = pathlib.Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        print(f"version={version}  sha={sha}  runs={args.runs}  peak_est={peak and round(peak)} GFLOPS")
        print(f"{'N':>6} {'median_ms':>10} {'min_ms':>10} {'GFLOPS':>9} {'%peak':>8} {'err/tol':>9}  check")

        for N in (int(s) for s in args.sizes.split(",")):
            rng = np.random.default_rng(1234 + N)
            a_np = rng.uniform(-1, 1, (N, N)).astype(np.float32)
            b_np = rng.uniform(-1, 1, (N, N)).astype(np.float32)
            a, b = mx.array(a_np), mx.array(b_np)
            mx.eval(a, b)  # upload outside the timed region

            # Correctness: same bound as the C++ harness, N * FLT_EPS * (|A||B|)_ij.
            c = mx.matmul(a, b)
            mx.eval(c)
            a64, b64 = a_np.astype(np.float64), b_np.astype(np.float64)
            ref = a64 @ b64
            tol = N * np.finfo(np.float32).eps * (np.abs(a64) @ np.abs(b64))
            err = np.abs(np.array(c, dtype=np.float64) - ref)
            ratio = float(np.max(err / tol))
            ok = bool(np.all(err <= tol))

            mx.eval(mx.matmul(a, b))  # warm warmup
            ms = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                mx.eval(mx.matmul(a, b))
                ms.append((time.perf_counter() - t0) * 1e3)
            med, mn = statistics.median(ms), min(ms)
            gflops = 2 * N**3 / (med * 1e-3) / 1e9
            pct = 100 * gflops / peak if peak else float("nan")
            gbps = 3 * N * N * 4 / (med * 1e-3) / 1e9

            print(f"{N:>6} {med:>10.3f} {mn:>10.3f} {gflops:>9.2f} {pct:>7.1f}% {ratio:>9.4f}  "
                  f"{'ok' if ok else 'FAIL'}")
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), sha, version, N, "", args.runs,
                        f"{med:.4f}", f"{mn:.4f}", f"{gflops:.3f}", "" if peak is None else f"{pct:.2f}",
                        f"{gbps:.3f}", f"{float(err.max()):g}", f"{ratio:g}", int(ok)])


if __name__ == "__main__":
    main()
