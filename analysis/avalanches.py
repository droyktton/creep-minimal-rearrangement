#!/usr/bin/env python3
"""Avalanches (clusters of correlated events) and their size L_av, from events.dat.

usage: avalanches.py [--K K] [--skip X] [--out FILE] RUN_DIR [RUN_DIR ...]

An event joins the cluster of an earlier event if its arc [x_left - 1, x_left + extent] touches
the arc of that event, and that event is among the last K L / 1024 events of the whole line
(the sequential criterion of Ferrero et al. 2017 at T -> 0+, with a memory proportional to L since
the number of events per unit time is). Events touching several recent clusters merge them.
The size of a cluster is the number of distinct sites its events moved (their arcs), and its
duration the time from its first to its last event. Only events after an advance X (default:
skip of params.txt) are used.

Prints per run: f T L K clusters <S> <S^2>/<S> S_99 max(S), and with --out the histogram of S
(log bins) for each run as FILE_<run>.dat.
"""
import argparse
import os

import numpy as np


def params(d):
    p = {}
    for line in open(os.path.join(d, "params.txt")):
        k, _, v = line.strip().partition(" ")
        p[k] = v
    return p


def clusters(xl, ext, L, K):
    n = len(xl)
    parent = np.arange(n)

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    last = np.full(L, -10**12, dtype=np.int64)   # last event that covered each site
    owner = np.full(L, -1, dtype=np.int64)       # its cluster (an event index, resolved by find)
    size = np.zeros(n, dtype=np.int64)
    for e in range(n):
        a, m = int(xl[e]), int(ext[e])
        sites = np.arange(a, a + m) % L
        touch = np.arange(a - 1, a + m + 1) % L
        recent = touch[last[touch] >= e - K]
        root = e
        if len(recent):
            roots = {find(int(o)) for o in owner[recent]}
            root = min(roots)
            for r in roots:
                if r != root:
                    parent[r] = root
                    size[root] += size[r]
        # new sites: not covered before by this cluster
        prev = owner[sites]
        new = sum(1 for o in prev if o < 0 or find(int(o)) != root)
        size[root] += new
        last[sites] = e
        owner[sites] = root
    roots = np.array([find(e) for e in range(n)])
    return size[np.unique(roots)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=float, default=1.0)
    ap.add_argument("--skip", type=float)
    ap.add_argument("--out")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    print("# f T L K clusters mean_S S2/S S_99 S_max")
    for d in a.runs:
        p = params(d)
        L = int(p["L"])
        skip = a.skip if a.skip is not None else float(p.get("skip", 0))
        e = np.loadtxt(os.path.join(d, "events.dat"), usecols=(6, 10, 11))
        keep = e[:, 0] - e[0, 0] >= skip
        K = max(1, int(round(a.K * L / 1024)))
        S = clusters(e[keep, 1], e[keep, 2], L, K)
        S = np.minimum(S, L)
        print(f"{p['f'].split()[0]} {p['T']} {L} {K} {len(S)} {S.mean():.4g} {(S**2).mean() / S.mean():.4g} "
              f"{np.percentile(S, 99):.4g} {S.max()}")
        if a.out:
            bins = np.unique(np.round(np.logspace(0, np.log10(L + 1), 40)).astype(int))
            h, _ = np.histogram(S, bins)
            np.savetxt(f"{a.out}_{os.path.basename(os.path.normpath(d))}.dat",
                       np.c_[np.sqrt(bins[:-1] * bins[1:]), h / np.diff(bins) / len(S)], header="S P(S)")


if __name__ == "__main__":
    main()
