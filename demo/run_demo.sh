#!/bin/bash
# Demo runs, then the figures. Results go to demo/runs/, figures to demo/figures/.
# About 40 minutes with 3 parallel jobs (JOBS=n to change).
# One sample (Philox disorder, seed 1, not periodic in u: M = 0), L = 1024, every run until the
# line has advanced by 1200, writing the metastable state every 20 steps:
#  - T -> 0+ at F = 0.3, 0.1, 0.03, 0.01;
#  - finite temperature at F = 0.093, T = 0.03, 0.1, 0.3, 1, and T -> 0+ with the same lmax = 50;
#  - velocity at f = 0.03, 0.1, 0.3 and T = 0.02 .. 0.1 (advance 1500, steady state after 300);
#  - persistence and chi4 on a sample of L = 512 (F_c = 0.949) at f = 0.1, 0.3 and T = 0.02 .. 0.1,
#    advance 6000, steady state after 300, persistence up to tau = 300/v, with the moved sites of
#    every event. At low T the time is dominated by a few long waits, and tau_alpha >> 1/v.
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
	for T in 0 0.03 0.1 0.3 1; do
		echo "--f 0.093 --T $T --lmax 50 --out $R/T${T}_F0.093"
	done
	for F in 0.0278 0.0927 0.278; do  # f = 0.03, 0.1, 0.3
		for T in 0.02 0.03 0.05 0.07 0.1; do
			echo "--f $F --T $T --lmax 50 --skip 300 --advance 1500 --out $R/vT_F${F}_T$T"
		done
	done
	for F in 0.0949 0.285; do  # L = 512: f = 0.1, 0.3
		for T in 0.02 0.03 0.05 0.07 0.1; do
			echo "--f $F --T $T --lmax 50 --skip 300 --persistence 300 --advance 6000 --save-moves --L 512 --conf-every 0 --out $R/pT_F${F}_T$T"
		done
	done
} | xargs -P "${JOBS:-3}" -I{} bash -c 'set -- {}; mkdir -p "${!#}"; OMP_NUM_THREADS=1 "$0" --L 1024 --M 0 --seed 1 --absolute --advance 1200 --events 10000000 --conf-every 20 "$@"' "$BIN"

python3 "$ROOT"/demo/plot_demo.py "$R" "$ROOT"/demo/figures
