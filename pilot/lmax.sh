#!/bin/bash
# Pilot 2: convergence in lmax at f = 0.01, L = 1024 (F_c = 0.927), T = 0.05
BIN=$(dirname "$0")/../build/mr_creep
for lm in 100 200 300; do echo "--lmax $lm --out lmax/f0.01_lmax$lm"; done |
xargs -P 3 -I{} bash -c 'set -- {}; mkdir -p "${!#}"; s=$(date +%s.%N); OMP_NUM_THREADS=1 nice -n 19 "$0" --L 1024 --M 0 --seed 1 --absolute --f 0.00927 --T 0.05 --advance 300 --skip 50 --events 100000000 --walltime 1800 "$@"; r=$?; echo "${!#} exit $r wall $(echo "$(date +%s.%N) - $s" | bc)"' "$BIN"
