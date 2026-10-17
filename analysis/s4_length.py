#!/usr/bin/env python3
"""Correlation length of the persistence from the four-point structure factor S4(q, tau).

usage: s4_length.py [--tau alpha|chi4] [--out FILE] RUN_DIR [RUN_DIR ...]

Each RUN_DIR is the output of mr_creep --persistence (persistence.dat, s4.dat, params.txt).
Runs with the same parameters except the seeds (seed, kmc_seed) are samples of one point: their
pi, chi4 and S4 are averaged with the number of persistence origins as weight (chi4 is then the
mean of the chi4 of each sample, the fluctuations of pi at fixed disorder).

At tau_alpha (pi = 1/2, default) or at the maximum of chi4, S4(q) is fitted to the
Ornstein-Zernike form A / (1 + (q ell)^2) (the correlation of the persistence decaying as
exp(-r/ell)), on the modes with S4 >= S4(q_1) / 20. Also given: ell_half = 1/q_half, with
S4(q_half) = A / 2, the same for the Ornstein-Zernike form, and chi4(tau) with L / ell.
The ratio r12 = S4(q_1)/S4(q_2) tends to 4 when ell >> L/(2 pi) (Ornstein-Zernike): ell is then
not resolved by the system and the fits fail or give only a lower bound (r12 > 3 is flagged).
One line per point: f T L lmax theta samples origins tau chi4 A ell ell_half r12 resolved.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

SEEDS = {"seed", "kmc_seed"}


def params(d):
    p = {}
    for line in open(os.path.join(d, "params.txt")):
        k, _, v = line.strip().partition(" ")
        p[k] = v
    return p


def load(d):
    with open(os.path.join(d, "s4.dat")) as f:
        head = f.readline()
        q = np.array(f.readline().split()[2:], float)
    n = int(re.search(r"(\d+) origins", head).group(1))
    s4 = np.loadtxt(os.path.join(d, "s4.dat"))
    pers = np.loadtxt(os.path.join(d, "persistence.dat"))
    return q, n, pers, s4


def oz(q, A, ell):
    return A / (1 + (q * ell) ** 2)


def fit(q, s):
    use = s >= s[0] / 20
    ell = 1 / q[np.argmin(abs(s - s[0] / 2))]
    try:
        (A, ell), _ = curve_fit(lambda x, a, l: np.log(oz(x, a, l)), q[use], np.log(s[use]),
                                p0=(s[0], ell), bounds=((0, 0), (np.inf, np.inf)), max_nfev=10000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, np.nan
    # q_half: where the data cross A/2, by log-linear interpolation
    k = np.nonzero(s < A / 2)[0]
    if len(k) == 0 or k[0] == 0:
        return A, ell, np.nan
    k = k[0]
    lq = np.interp(np.log(A / 2), np.log(s[[k, k - 1]]), np.log(q[[k, k - 1]]))
    return A, ell, np.exp(-lq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", choices=["alpha", "chi4"], default="alpha")
    ap.add_argument("--out")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()

    groups = defaultdict(list)
    for d in a.runs:
        if not os.path.exists(os.path.join(d, "s4.dat")):
            print(f"{d}: no s4.dat, skipped", file=sys.stderr)
            continue
        p = params(d)
        groups[tuple(sorted((k, v) for k, v in p.items() if k not in SEEDS | {"Fc", "F"}))].append(d)

    out = open(a.out, "w") if a.out else sys.stdout
    print("# f T L lmax theta samples origins tau chi4 A ell ell_half r12 resolved", file=out)
    for key, ds in sorted(groups.items(), key=lambda g: tuple(float(dict(g[0])[k].split()[0]) for k in ("L", "f", "T"))):
        p = dict(key)
        q = None
        W = 0
        P = S = None
        for d in ds:
            qd, n, pers, s4 = load(d)
            if q is None:
                q, P, S = qd, n * pers, n * s4
            elif len(pers) != len(P):
                print(f"{d}: other tau grid, skipped", file=sys.stderr)
                continue
            else:
                P += n * pers
                S += n * s4
            W += n
        P /= W
        S /= W
        # the tau grid is relative to 1/v of each sample: average it too (log)
        tau = P[:, 0]
        pi, chi4 = P[:, 1], P[:, 3]
        if a.tau == "alpha":
            k = np.nonzero(pi <= 0.5)[0]
            if len(k) == 0 or k[0] == 0:
                print(f"{ds[0]}: pi does not cross 1/2", file=sys.stderr)
                continue
            b = k[0] if 0.5 - pi[k[0]] < pi[k[0] - 1] - 0.5 else k[0] - 1
        else:
            b = int(np.argmax(chi4))
        A, ell, ellh = fit(q, S[b, 1:])
        r12 = S[b, 1] / S[b, 2]
        print(f"{p['f'].split()[0]} {p['T']} {p['L']} {p['lmax']} {p['theta']} {len(ds)} {W} "
              f"{tau[b]:.6g} {chi4[b]:.6g} {A:.6g} {ell:.6g} {ellh:.6g} {r12:.3g} {int(r12 <= 3)}", file=out)


if __name__ == "__main__":
    main()
