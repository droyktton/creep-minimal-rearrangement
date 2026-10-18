#!/usr/bin/env python3
"""Statistics of the waiting times dt of the steady state, from events.dat.

usage: waits.py [--skip X] [--tail P] RUN_DIR [RUN_DIR ...]

At low T the distribution of dt can have a power-law tail P(dt > t) ~ t^-mu with mu < 1: the
mean wait then diverges, the steady-state time is set by a few waits, and v does not average
within a run. Per run:
  max_frac, top10_frac  fraction of the steady-state time in the longest wait, in the 10 longest;
  mu                    Hill estimate of the tail exponent on the longest fraction P of the waits
                        (default 0.5%, at least 50 waits);
  dlnv                  max - min of ln v over the four quarters of the steady-state advance.
Runs with the same parameters except the seeds are samples of one point. One line per point:
f T L samples mean (sd) of ln v (summary.txt) and of each statistic over the samples.
Only events after an advance X (default: skip of params.txt) are used.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

SEEDS = {"seed", "kmc_seed"}


def params(d):
    p = {}
    for line in open(os.path.join(d, "params.txt")):
        k, _, v = line.strip().partition(" ")
        p[k] = v
    return p


def summary(d):
    return {k: float(v) for k, v in (line.split() for line in open(os.path.join(d, "summary.txt")))}


def point(p):
    return tuple(sorted((k, v) for k, v in p.items() if k not in SEEDS | {"Fc", "F"}))


def run_stats(d, skip=None, tail=0.005):
    """max_frac, top10_frac, mu, dlnv of one run."""
    if skip is None:
        skip = float(params(d).get("skip", 0))
    e = np.loadtxt(os.path.join(d, "events.dat"), usecols=(6, 8))
    keep = e[:, 0] - e[0, 0] >= skip
    u, dt = e[keep, 0], e[keep, 1]
    s = np.sort(dt)[::-1]
    k = max(50, int(tail * len(s)))
    mu = 1 / np.mean(np.log(s[:k] / s[k]))
    edges = np.linspace(u[0], u[-1], 5)
    q = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, 3)
    lnv = [np.log(np.ptp(u[q == i]) / dt[q == i].sum()) for i in range(4)]
    return s[0] / s.sum(), s[:10].sum() / s.sum(), mu, np.ptp(lnv)


def points(runs, skip=None, tail=0.005):
    """{(f, T, L): array of rows ln_v max_frac top10_frac mu dlnv, one per sample}"""
    groups = defaultdict(list)
    for d in runs:
        if not os.path.exists(os.path.join(d, "summary.txt")):
            print(f"{d}: not finished, skipped", file=sys.stderr)
            continue
        p = params(d)
        groups[point(p)].append(d)
    out = {}
    for key, ds in groups.items():
        p = dict(key)
        rows = [(summary(d)["ln_v"],) + run_stats(d, skip, tail) for d in ds]
        out[(float(p["f"].split()[0]), float(p["T"]), int(p["L"]))] = np.array(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=float)
    ap.add_argument("--tail", type=float, default=0.005)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    print("# f T L samples ln_v sd max_frac sd top10_frac sd mu sd dlnv sd")
    for (f, T, L), r in sorted(points(a.runs, a.skip, a.tail).items()):
        sd = r.std(axis=0, ddof=1) if len(r) > 1 else np.full(r.shape[1], np.nan)
        cols = " ".join(f"{m:.4g} {s:.2g}" for m, s in zip(r.mean(axis=0), sd))
        print(f"{f:g} {T:g} {L} {len(r)} {cols}")


if __name__ == "__main__":
    main()
