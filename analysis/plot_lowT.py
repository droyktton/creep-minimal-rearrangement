#!/usr/bin/env python3
"""Figure of the velocity and the waiting times at low T: python3 plot_lowT.py [--Fc FC] FIG RUN_DIR [RUN_DIR ...]

The runs of the ell and lowT grids at one L (cluster/make_tasks.py), seeds 1-4 at each (f, T).
(a) ln v against 1/T, mean and sd over the samples, with the fit of each f;
(b) effective barrier -d ln v/d(1/T) against f, with f^(-1/4);
(c) tail exponent mu of the waiting times, P(dt > t) ~ t^-mu, against T; the mean wait
    diverges for mu < 1;
(d) fraction of the steady-state time in the longest wait of each run;
(e) spread of ln v: between samples (sd) and within a run (max - min over the four quarters
    of the steady-state advance);
(f) persistence pi of each sample against the advance v tau at the smallest f (v of summary.txt).
    The run sets tau_max = X/v with v measured over the second half of the transient: at low T
    that misses the long waits, overestimates v and cuts the window far below an advance X.
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arrhenius import barrier
from waits import params, point, points, summary


def persistence(runs, f0):
    """{T: [(v tau, pi) of each sample]} at f = f0"""
    out = defaultdict(list)
    for d in runs:
        p = dict(point(params(d)))
        if not np.isclose(float(p["f"].split()[0]), f0) or not os.path.exists(os.path.join(d, "persistence.dat")):
            continue
        P = np.loadtxt(os.path.join(d, "persistence.dat"))
        out[float(p["T"])].append((P[:, 0] * summary(d)["v"], P[:, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Fc", type=float, default=0.927)
    ap.add_argument("fig")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    fig_path, runs = a.fig, a.runs
    # {(f, T, L): rows ln_v max_frac top10_frac mu dlnv}, f = F/F_c
    R = {(round(F / a.Fc, 6), T, L): r for (F, T, L), r in points(runs).items()}
    Fs = sorted({f for f, _, _ in R})
    Ts = sorted({T for _, T, _ in R})
    L = sorted({L for _, _, L in R})
    cf = dict(zip(Fs, plt.cm.viridis(np.linspace(0, 0.85, len(Fs)))))
    ct = dict(zip(Ts, plt.cm.plasma(np.linspace(0, 0.85, len(Ts)))))
    m = lambda f, T, c: R[(f, T, L[0])][:, c]

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    U = {}
    for f in Fs:
        T = np.array([T for T in Ts if (f, T, L[0]) in R])
        lv = np.array([m(f, t, 0).mean() for t in T])
        sd = np.array([m(f, t, 0).std(ddof=1) for t in T])
        U[f] = barrier(T, lv)
        ax[0, 0].errorbar(1 / T, lv, sd, fmt="o", ms=4, capsize=2, color=cf[f], label=f"f = {f:g}: {U[f]:.2f}")
        x = np.linspace(1 / T.max(), 1 / T.min())
        ax[0, 0].plot(x, np.polyval(np.polyfit(1 / T, lv, 1), x), lw=0.8, color=cf[f])
        ax[0, 2].semilogy(T, [m(f, t, 1).mean() for t in T], "o-", color=cf[f], label=f"f = {f:g}")
        mu = np.array([m(f, t, 3).mean() for t in T])
        ax[1, 0].plot(T, mu, "o-", color=cf[f], label=f"f = {f:g}: $T_0$ = {np.sum(T * T) / np.sum(T * mu):.3f}")
        ax[1, 1].semilogy(T, sd, "o-", color=cf[f], label=f"f = {f:g}")
        ax[1, 1].semilogy(T, [m(f, t, 4).mean() for t in T], "s--", ms=4, mfc="none", color=cf[f])
    ax[0, 0].set(xlabel="1/T", ylabel="ln v  ($\\nu_0$ = 1)", title="(a) velocity; legend: $U_{eff}$")
    ax[0, 0].legend(fontsize=8)
    f = np.array(Fs)
    ax[0, 1].loglog(f, [U[x] for x in Fs], "ko-", label="$-d \\ln v / d(1/T)$")
    ax[0, 1].loglog(f, U[Fs[0]] * (f / f[0]) ** -0.25, "C3--", label="$f^{-1/4}$")
    ax[0, 1].set(xlabel="$f = F/F_c$", ylabel="effective barrier", title="(b) effective barrier against f")
    ax[0, 1].legend(fontsize=8)
    ax[0, 2].set(xlabel="T", ylabel="longest wait / steady-state time", title="(d) weight of the longest wait")
    ax[0, 2].legend(fontsize=8)
    ax[1, 0].axhline(1, color="k", lw=0.6, ls=":")
    ax[1, 0].set(xlabel="T", ylabel="$\\mu$", title="(c) tail of the waits, $P(dt > t) \\sim t^{-\\mu}$; fit $\\mu = T/T_0$")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].set(xlabel="T", ylabel="spread of ln v",
                 title="(e) ln v: sd between samples (o), range within a run (□)")
    ax[1, 1].legend(fontsize=8)
    P = persistence(runs, Fs[0] * a.Fc)
    for T in sorted(P):
        for k, xy in enumerate(P[T]):
            ax[1, 2].semilogx(*xy, lw=1, color=ct[T], label=f"T = {T:g}" if k == 0 else None)
    ax[1, 2].axhline(0.5, color="k", lw=0.6, ls=":")
    ax[1, 2].set(xlabel="advance $v\\tau$", ylabel="$\\langle\\pi\\rangle$", title=f"(f) persistence at f = {Fs[0]:g}, each sample")
    ax[1, 2].legend(fontsize=8)
    fig.suptitle(f"Velocity and waiting times, L = {L[0]}, 4 samples per point, U(l) = $l^{{1/3}}$; steady state (advance 300 to 3000)")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)


if __name__ == "__main__":
    main()
