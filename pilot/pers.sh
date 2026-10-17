#!/bin/bash
# Pilot 3: persistence, chi4 and S4 at f = 0.01, 0.03, 0.1, T = 0.05 and 0.1, L = 1024 and 4096
# (seed 1; F = f F_c with F_c = 0.927 of L = 1024), lmax 150, 100, 50 with the fallback to larger
# windows; advance 3000, steady state after 300, persistence up to 1000/v with 100 origins per
# tau_max (v of the transient can be off by a factor of several), moved sites kept (moves.bin)
BIN=$(dirname "$0")/../build/mr_creep
for L in 4096 1024; do for T in 0.05 0.1; do for fl in "0.01 150" "0.03 100" "0.1 50"; do
	set -- $fl
	F=$(python3 -c "print($1*0.927)")
	echo "--L $L --T $T --f $F --lmax $2 --out pers/L${L}_f$1_T$T"
done; done; done |
xargs -P "${JOBS:-3}" -I{} bash -c 'set -- {}; mkdir -p "${!#}"; s=$(date +%s.%N); OMP_NUM_THREADS=1 nice -n 19 "$0" --M 0 --seed 1 --absolute --advance 3000 --skip 300 --persistence 1000 --origins 100 --save-moves --events 1000000000 --walltime 3600 --checkpoint 300 --restart "$@"; r=$?; echo "${!#} exit $r wall $(echo "$(date +%s.%N) - $s" | bc)"' "$BIN"
