#!/usr/bin/env python3
"""Show the latest run of <label> next to the MLX baselines and the previous run of <label>.

    python3 scripts/compare.py <label> [--csv results/results.csv]
Stdlib only (no venv needed).
"""
import argparse
import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--csv", default=str(ROOT / "results" / "results.csv"))
    args = ap.parse_args()

    runs = {}  # (version, N) -> list of passing rows, oldest first
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["pass"] == "1" and r["gflops"]:
                runs.setdefault((r["version"], int(r["N"])), []).append(r)

    sizes = sorted(n for (v, n) in runs if v == args.label)
    if not sizes:
        raise SystemExit(f"no passing rows for '{args.label}' in {args.csv}")

    def g(v, n, idx=-1):
        rows = runs.get((v, n), [])
        return float(rows[idx]["gflops"]) if len(rows) >= abs(idx) else None

    def fmt(x, spec):
        if x is None or x != x:  # missing or NaN: keep column width
            return "-".rjust(len(format(0.0, spec)))
        return format(x, spec)
    hdr = (f"{'N':>5} {'ms':>9} {'GFLOPS':>8} {'%peak':>6} {'GB/s':>7} {'thr':>3} | "
           f"{'prev':>8} {'Δ':>7} | {'mlx-gpu':>8} {'x gpu':>7} | {'mlx-cpu':>8} {'x cpu':>7}")
    print(f"version={args.label}  sha={runs[(args.label, sizes[0])][-1]['git_sha']}")
    print(hdr)
    print("-" * len(hdr))
    for n in sizes:
        r = runs[(args.label, n)][-1]
        me, prev = float(r["gflops"]), g(args.label, n, -2)
        gpu, cpu = g("mlx-gpu", n), g("mlx-cpu", n)
        delta = None if prev is None else 100 * (me / prev - 1)
        print(f"{n:>5} {float(r['median_ms']):>9.2f} {me:>8.2f} {fmt(float(r['pct_peak_est'] or 'nan'), '>5.1f')}% "
              f"{float(r['gbps_min_traffic']):>7.2f} {r['threads']:>3} | "
              f"{fmt(prev, '>8.2f')} {fmt(delta, '>+6.1f')}{'%' if delta is not None else ' '} | "
              f"{fmt(gpu, '>8.1f')} {fmt(None if gpu is None else me / gpu, '>7.4f')} | "
              f"{fmt(cpu, '>8.1f')} {fmt(None if cpu is None else me / cpu, '>7.4f')}")
    print("x gpu / x cpu = your GFLOPS / baseline GFLOPS.  %peak is vs the NEON estimate for 'thr' threads.")


if __name__ == "__main__":
    main()
