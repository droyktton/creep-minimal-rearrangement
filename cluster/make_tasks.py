#!/usr/bin/env python3
"""Task list for run_array.sbatch: one line per run, "OUTDIR ARGS...".

usage: make_tasks.py GRID > tasks.txt     GRID: fc | ell | lowT

fc:    F_c of samples (seeds 1-8) at L = 1024, 4096 (M = 4096; the M = 0 runs see the same
       disorder for 0 <= u < M), to fix F = f F_c.
ell:   ell(f, T) from S4(q, tau_alpha): f = 0.01 .. 0.1, T = 0.05, 0.07, 0.1, L = 4096 and 16384,
       samples = seeds 1-4 (kmc seed = seed).
lowT:  the same at T = 0.02, 0.03 and L = 4096 only: v, tau_alpha and u_c = v tau_alpha over a wider
       range of T (ell is not resolved there at small f).
Edit the lists below to change the grid.
"""
import sys

FC = 0.927            # F_c used for F = f F_c (L = 1024, seed 1; to be replaced by the fc result)
LMAX = {0.01: 200, 0.02: 150, 0.03: 100, 0.05: 80, 0.1: 50}   # >= 2.5 x the 99% window
BASE = "--M 0 --absolute --skip 300 --advance 3000 --persistence 1000 --origins 100 --save-moves --events 4000000000"


def fc():
    for L in (1024, 4096):
        for s in range(1, 9):
            print(f"fc/L{L}_s{s} --L {L} --M 4096 --f 0.5 --seed {s} --events 0")


def ell():
    for L in (4096, 16384):
        for T in (0.05, 0.07, 0.1):
            for f, lm in LMAX.items():
                for s in range(1, 5):
                    print(f"ell/L{L}_f{f}_T{T}_s{s} {BASE} --L {L} --T {T} --f {f * FC:.6g} --lmax {lm} --seed {s} --kmc-seed {s}")


def lowT():
    for T in (0.02, 0.03):
        for f, lm in LMAX.items():
            for s in range(1, 5):
                print(f"ell/L4096_f{f}_T{T}_s{s} {BASE} --L 4096 --T {T} --f {f * FC:.6g} --lmax {lm} --seed {s} --kmc-seed {s}")


{"fc": fc, "ell": ell, "lowT": lowT}[sys.argv[1]]()
