#!/usr/bin/env bash
# build -> tests -> bench -> compare.   usage: scripts/bench.sh <label> [threads=1]
# Env: BENCH_CSV=path to log somewhere other than results/results.csv
set -euo pipefail
cd "$(dirname "$0")/.."

label="${1:-}"
threads="${2:-1}"
if [[ -z "$label" || "$label" == *[,\"[:space:]]* ]]; then
  echo "usage: /bench <label> [threads]   (label: no spaces, commas or quotes)" >&2; exit 2
fi
[[ "$threads" =~ ^[0-9]+$ ]] || { echo "threads must be an integer" >&2; exit 2; }
csv="${BENCH_CSV:-results/results.csv}"

echo "== build"
[[ -d build ]] || cmake -S . -B build >/dev/null
cmake --build build -j 2>&1 | grep -Ev '^\[ *[0-9]+%\] (Building|Linking|Built)' || true
[[ -x build/bench ]] || { echo "build failed" >&2; exit 1; }

echo "== tests (must pass before any number is reported)"
if ! ./build/test_matmul; then echo "TESTS FAILED - not benchmarking" >&2; exit 1; fi

echo "== bench  (OMP_NUM_THREADS=$threads)"
OMP_NUM_THREADS="$threads" ./build/bench --version "$label" --csv "$csv"

echo "== compare"
python3 scripts/compare.py "$label" --csv "$csv"
