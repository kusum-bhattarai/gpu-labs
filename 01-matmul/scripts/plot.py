#!/usr/bin/env python3
"""GFLOPS vs N, one line per version (latest passing run of each version/N).

    .venv/bin/python scripts/plot.py [--csv results/results.csv] [--out results/gflops.png]
                                     [--versions naive,tiled,mlx-gpu] [--dark]
"""
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Fixed categorical order (never cycled); a version keeps its color by first appearance.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark":  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
INK = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#8a8984", grid="#e6e5e1"),
    "dark":  dict(surface="#1a1a19", primary="#f0efec", secondary="#c3c2b7", muted="#8a8984", grid="#2e2e2c"),
}
# Same estimates as bench.cpp - verify.
PEAK_1P_EST = 4.05 * 32  # GFLOPS, one P-core, NEON FMA


def load(path, only):
    latest = {}  # (version, N) -> row ; later rows overwrite earlier ones
    order = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["pass"] != "1" or not row["gflops"]:
                continue
            v = row["version"]
            if only and v not in only:
                continue
            if v not in order:
                order.append(v)
            latest[(v, int(row["N"]))] = row
    return order, latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "results.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "gflops.png"))
    ap.add_argument("--versions", default="", help="comma list; default all")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args()

    only = [v for v in args.versions.split(",") if v]
    order, latest = load(args.csv, only)
    if only:
        order = [v for v in only if v in order]
    if not order:
        raise SystemExit(f"no passing rows in {args.csv}")
    if len(order) > 8:
        raise SystemExit("more than 8 versions - pass --versions to pick which to compare")

    mode = "dark" if args.dark else "light"
    ink, colors = INK[mode], SERIES[mode]
    plt.rcParams.update({"font.size": 11, "text.color": ink["primary"],
                         "axes.labelcolor": ink["secondary"], "xtick.color": ink["secondary"],
                         "ytick.color": ink["secondary"]})
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=ink["surface"])
    ax.set_facecolor(ink["surface"])

    all_n = sorted({n for (_, n) in latest})
    ends = []
    for i, v in enumerate(order):
        ns = sorted(n for (vv, n) in latest if vv == v)
        ys = [float(latest[(v, n)]["gflops"]) for n in ns]
        ax.plot(ns, ys, color=colors[i], lw=2, marker="o", ms=7,
                markeredgecolor=ink["surface"], markeredgewidth=2, label=v, zorder=3)
        ends.append((v, ns[-1], ys[-1]))

    ax.axhline(PEAK_1P_EST, color=ink["muted"], lw=1, ls="--", zorder=1)
    ax.annotate(f"1 P-core NEON peak (est) {PEAK_1P_EST:.0f}", (all_n[0], PEAK_1P_EST),
                xytext=(0, 4), textcoords="offset points", color=ink["secondary"], fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(all_n, [str(n) for n in all_n])
    ax.set_xlabel("N")
    ax.set_ylabel("GFLOPS (median, log scale)")
    ax.set_title("matmul float32 — GFLOPS by version", loc="left", color=ink["primary"])
    ax.grid(True, which="major", color=ink["grid"], lw=0.8, zorder=0)
    ax.grid(False, which="minor")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink["grid"])
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}" if y >= 1 else f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(all_n[0] / 1.15, all_n[-1] * 1.6)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    for t in leg.get_texts():
        t.set_color(ink["primary"])
    fig.tight_layout()

    # Direct labels at line ends (<= 4 series), text in ink, nudged apart so they never overlap.
    if len(order) <= 4:
        fig.canvas.draw()
        to_px, min_gap = ax.transData, 15.0
        placed = []  # (label, x_data, y_px)
        for v, x, y in sorted(ends, key=lambda e: e[2]):
            py = to_px.transform((x, y))[1]
            if placed and py - placed[-1][2] < min_gap:
                py = placed[-1][2] + min_gap
            placed.append((f"{v}  {y:,.0f}", x, py, y))
        for text, x, py, y in placed:
            px = to_px.transform((x, y))
            ax.annotate(text, (x, y), xytext=(10, (py - px[1]) * 72 / fig.dpi), textcoords="offset points",
                        va="center", color=ink["primary"], fontsize=10)

    fig.savefig(args.out, dpi=150, facecolor=ink["surface"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
