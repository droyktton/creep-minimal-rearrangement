#!/usr/bin/env python3
"""Effective barrier of the velocity, U_eff = -d ln v / d(1/T), at each force, from summary.txt.

usage: arrhenius.py [--split T] RUN_DIR [RUN_DIR ...]

ln v is averaged over the samples of each point (runs with the same parameters except the seeds),
then fitted linearly in 1/T at each (f, L). With --split T, the fit is also done separately on
the temperatures >= T and < T, to see whether the slope changes at low T.
f is the force of params.txt: F with --absolute, F/F_c otherwise.
One line per (f, L): f L temperatures U_eff U_eff*f^(1/4) [U_eff(T >= split) U_eff(T < split)].
"""
import argparse
from collections import defaultdict

import numpy as np

from waits import params, point, summary


def lnv_points(runs):
    """{(f, T, L): ln v of each sample}"""
    groups = defaultdict(list)
    for d in runs:
        try:
            lv = summary(d)["ln_v"]
        except OSError:
            continue
        p = dict(point(params(d)))
        groups[(float(p["f"].split()[0]), float(p["T"]), int(p["L"]))].append(lv)
    return groups


def barrier(T, lv):
    return -np.polyfit(1 / np.asarray(T), lv, 1)[0] if len(T) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=float)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    by_f = defaultdict(list)
    for (f, T, L), lv in lnv_points(a.runs).items():
        by_f[(f, L)].append((T, np.mean(lv)))
    print("# f L temperatures U_eff U_eff*f^(1/4)" + (" U_eff(T>=split) U_eff(T<split)" if a.split else ""))
    for (f, L), pts in sorted(by_f.items()):
        T, lv = np.array(sorted(pts)).T
        u = barrier(T, lv)
        line = f"{f:g} {L} {len(T)} {u:.4g} {u * f ** 0.25:.4g}"
        if a.split:
            hi, lo = T >= a.split, T < a.split
            line += f" {barrier(T[hi], lv[hi]):.4g} {barrier(T[lo], lv[lo]):.4g}"
        print(line)


if __name__ == "__main__":
    main()
