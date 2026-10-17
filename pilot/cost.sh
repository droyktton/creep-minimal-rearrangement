#!/bin/bash
# Pilot 1: cost per unit advance and window distribution, L = 1024 (F_c = 0.927), T = 0.05
BIN=$(dirname "$0")/../build/mr_creep
for f in 0.01 0.03 0.1; do for lm in 50 100; do
	F=$(python3 -c "print($f*0.927)")
	echo "--f $F --lmax $lm --out cost/f${f}_lmax$lm"
done; done | xargs -P 3 -I{} bash -c 'set -- {}; mkdir -p "${!#}"; s=$(date +%s.%N); OMP_NUM_THREADS=1 nice -n 19 "$0" --L 1024 --M 0 --seed 1 --absolute --T 0.05 --advance 100 --skip 50 --events 100000000 --walltime 600 --checkpoint 60 "$@"; r=$?; echo "${!#} exit $r wall $(echo "$(date +%s.%N) - $s" | bc)"' "$BIN"
