#!/bin/bash
# Demo runs, then the figures. Results go to demo/runs/, figures to demo/figures/.
# About 10 minutes with 3 parallel jobs (JOBS=n to change); the slowest run is F = 0.01.
# One sample (Philox disorder, seed 1, not periodic in u: M = 0), L = 1024, every run until the
# line has advanced by 1200, writing the metastable state every 20 steps:
#  - T -> 0+ at F = 0.3, 0.1, 0.03, 0.01;
#  - finite temperature at F = 0.093, T = 0.25, 0.5, 1, 2, and T -> 0+ with the same lmax = 50.
# F_c of this sample (computed with M = 4096) is 0.927, so F = 0.093 is f = 0.1.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
make -s -C "$ROOT"
BIN="$ROOT"/build/mr_creep
R="$ROOT"/demo/runs
mkdir -p "$R"

{
	for F in 0.3 0.1 0.03 0.01; do
		echo "--f $F --out $R/t0_F$F"
	done
	for T in 0 0.25 0.5 1 2; do
		echo "--f 0.093 --T $T --lmax 50 --out $R/T${T}_F0.093"
	done
} | xargs -P "${JOBS:-3}" -I{} bash -c 'set -- {}; mkdir -p "${!#}"; OMP_NUM_THREADS=1 "$0" --L 1024 --M 0 --seed 1 --absolute --advance 1200 --events 10000000 --conf-every 20 "$@"' "$BIN"

python3 "$ROOT"/demo/plot_demo.py "$R" "$ROOT"/demo/figures
