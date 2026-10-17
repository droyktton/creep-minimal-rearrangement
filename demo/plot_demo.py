"""Figures of the demo (demo/run_demo.sh). Usage: python3 plot_demo.py RUNS_DIR FIG_DIR [F_c]

t0_creep.png: T -> 0+ at several forces (one sample, L = 1024): activity map and metastable states
  at the lowest force, structure factor with the scaling of the equilibrium-depinning crossover,
  size of the rearrangements against f, and the statistics of the events.
finite_t.png: F = 0.093 at several temperatures: structure factor, window sizes, velocity against
  1/T, activity maps and event areas.

Steady state: the states and events after the line has advanced by SKIP.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SKIP = 300
ZEQ, ZDEP = 2 / 3, 1.25
NUEQ = 1 / (2 - ZEQ)  # 3/4


def load(d):
    par = {}
    for l in open(f"{d}/params.txt"):
        k, v = l.split(None, 1)
        par[k] = v.split()[0]
    L = int(par["L"])
    ev = np.loadtxt(f"{d}/events.dat", ndmin=2)
    c = np.fromfile(f"{d}/configs.bin", dtype=np.int32)
    c = c[:len(c) // L * L].reshape(-1, L)
    adv = ev[:, 6] - ev[0, 6]
    ss = adv > SKIP
    cs = c[c.mean(1) - c[0].mean() > SKIP].astype(float)
    return dict(par=par, L=L, F=float(par["F"]), T=float(par["T"]), ev=ev, ss=ss, c=c, cs=cs, adv=adv)


def structure_factor(cs):
    uq = np.fft.rfft(cs - cs.mean(1, keepdims=True), axis=1)
    L = cs.shape[1]
    return 2 * np.pi * np.arange(1, L // 2 + 1) / L, (np.abs(uq[:, 1:]) ** 2 / L).mean(0)


def logbins(x, n=25):
    x = x[x > 0]
    e = np.unique(np.round(np.logspace(0, np.log10(x.max() + 1), n))) - 0.5
    h, e = np.histogram(x, e, density=True)
    c = np.sqrt(e[1:] * e[:-1])
    return c[h > 0], h[h > 0]


def activity(ax, D, n=1500, title=""):
    """Moved regions of n consecutive steady-state events: [x_left, x_left + L_rel) at height step."""
    ev = D["ev"][D["ss"]][:n]
    L = D["L"]
    xs, ys = [], []
    for k, e in enumerate(ev):
        x0, w = int(e[10]), int(e[4])
        for a, b in ((x0, min(x0 + w, L)), (0, max(0, x0 + w - L))):  # periodic wrap
            if b > a:
                xs += [a, b, None]; ys += [k, k, None]
    ax.plot(xs, ys, lw=0.8, color="k")
    ax.set(xlabel="$x$", ylabel="event", xlim=(0, L), ylim=(0, len(ev)), title=title)


def t0(runs, figs, Fc):
    D = {float(d[len("t0_F"):]): load(f"{runs}/{d}") for d in os.listdir(runs) if d.startswith("t0_F")}
    Fs = sorted(D)
    cols = dict(zip(Fs, plt.cm.viridis(np.linspace(0, 0.85, len(Fs)))))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    lo = D[Fs[0]]
    activity(ax[0, 0], lo, title=f"(a) activity map, f = {Fs[0] / Fc:.2g}\n(moved region of each event)")
    c = lo["cs"]
    for k, x in enumerate(c[::max(1, len(c) // 12)]):
        ax[0, 1].plot(x - c[0].mean(), lw=0.8, color=plt.cm.Greys(0.35 + 0.6 * k / 12))
    ax[0, 1].set(xlabel="$x$", ylabel="$u$", title=f"(b) metastable states, f = {Fs[0] / Fc:.2g}")

    for F in Fs:
        q, S = structure_factor(D[F]["cs"])
        f = F / Fc
        ax[0, 2].loglog(q * f ** -NUEQ, S * f ** (NUEQ * (1 + 2 * ZEQ)), "o", ms=2.5, color=cols[F], label=f"f = {f:.2g}")
    x = np.logspace(-1.5, 1.5, 10)
    ax[0, 2].loglog(x, 3 * x ** (-1 - 2 * ZEQ), "k:", label="$\\zeta_{eq}$ = 2/3")
    ax[0, 2].loglog(x, 3 * x ** (-1 - 2 * ZDEP), "k--", label="$\\zeta_{dep}$ = 5/4")
    ax[0, 2].set(xlabel="$q f^{-\\nu_{eq}}$", ylabel="$S(q) f^{\\nu_{eq}(1+2\\zeta_{eq})}$",
                 title="(c) structure factor, $\\nu_{eq}$ = 3/4", ylim=(1e-4, None))
    ax[0, 2].legend(fontsize=7)

    f = np.array(Fs) / Fc
    for col, lab, m in ((1, "window $l$ of the rearrangement", "o-"), (4, "$L_{rel}$ (sites moved)", "s-")):
        ax[1, 0].loglog(f, [D[F]["ev"][D[F]["ss"], col].mean() for F in Fs], m, label=lab)
    ax[1, 0].loglog(f, 5 * f ** -NUEQ, "k--", label="$f^{-3/4}$")
    ax[1, 0].set(xlabel="$f = F/F_c$", ylabel="mean over events", title="(d) size of the events")
    ax[1, 0].legend(fontsize=8)
    for F in Fs:
        e = D[F]["ev"][D[F]["ss"]]
        c, h = logbins(e[:, 5])
        ax[1, 1].loglog(c, h, "o-", ms=3, color=cols[F], label=f"f = {F / Fc:.2g}")
        ax[1, 2].loglog(e[:, 4], e[:, 5], ".", ms=1, alpha=0.2, color=cols[F])
    ax[1, 1].set(xlabel="area", ylabel="probability density", title="(e) event area")
    ax[1, 1].legend(fontsize=8)
    ll = np.array([3, 1000])
    ax[1, 2].loglog(ll, 0.3 * ll ** (1 + ZDEP), "k--", label="$L^{1+\\zeta_{dep}}$")
    ax[1, 2].set(xlabel="$L_{rel}$", ylabel="area", title="(f) area vs length of the events")
    ax[1, 2].legend(fontsize=8)
    fig.suptitle(f"T $\\to$ 0$^+$, one sample, L = {lo['L']}, $F_c$ = {Fc:.3g}; steady state (advance > {SKIP})")
    fig.tight_layout()
    fig.savefig(f"{figs}/t0_creep.png", dpi=110)
    for F in Fs:
        e = D[F]["ev"][D[F]["ss"]]
        print(f"T = 0, f = {F / Fc:.3g}: {len(e)} events, mean l {e[:, 1].mean():.1f}, "
              f"mean L_rel {e[:, 4].mean():.1f}, mean area {e[:, 5].mean():.1f}")


def finite_t(runs, figs, Fc):
    D = {float(d[1:d.index("_F")]): load(f"{runs}/{d}") for d in os.listdir(runs) if d.startswith("T") and "_F" in d}
    Ts = sorted(D)
    cols = dict(zip(Ts, plt.cm.plasma(np.linspace(0, 0.85, len(Ts)))))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for T in Ts:
        q, S = structure_factor(D[T]["cs"])
        ax[0, 0].loglog(q, S, "o", ms=2.5, color=cols[T], label=f"T = {T:g}")
        c, h = logbins(D[T]["ev"][D[T]["ss"], 1], 30)
        ax[0, 1].semilogy(c, h, "o-", ms=3, color=cols[T], label=f"T = {T:g}")
        c, h = logbins(D[T]["ev"][D[T]["ss"], 5])
        ax[1, 2].loglog(c, h, "o-", ms=3, color=cols[T], label=f"T = {T:g}")
    q = q[q < 1]
    for z, ls in ((ZDEP, "--"), (0.5, ":")):
        ax[0, 0].loglog(q, 0.5 * q ** (-1 - 2 * z), "k" + ls, label=f"$\\zeta$ = {z:g}")
    ax[0, 0].set(xlabel="$q$", ylabel="$S(q)$", title="(a) structure factor")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].set(xlabel="window $l$", ylabel="probability density", title="(b) window of the activated rearrangements")
    ax[0, 1].legend(fontsize=8)
    ax[1, 2].set(xlabel="area", ylabel="probability density", title="(f) event area")
    ax[1, 2].legend(fontsize=8)

    Tp = [T for T in Ts if T > 0]
    U0, th = float(D[Tp[0]]["par"]["U0"]), float(D[Tp[0]]["par"]["theta"])
    # time in units of 1/nu0: sum(dt) exp(U(2)/T); v = advance / time
    lnv = [np.log(D[T]["adv"][-1] - D[T]["adv"][D[T]["ss"]][0]) - np.log(D[T]["ev"][D[T]["ss"], 8].sum())
           - U0 * 2 ** th / T for T in Tp]
    x = 1 / np.array(Tp)
    ax[0, 2].plot(x, lnv, "ko-")
    b = np.polyfit(x, lnv, 1)
    ax[0, 2].plot(x, np.polyval(b, x), "C3--", label=f"slope {-b[0]:.2f} (effective barrier)")
    ax[0, 2].set(xlabel="1/T", ylabel="ln v  ($\\nu_0$ = 1)", title="(c) velocity (Arrhenius)")
    ax[0, 2].legend(fontsize=8)
    activity(ax[1, 0], D[Tp[len(Tp) // 2]], title=f"(d) activity map, T = {Tp[len(Tp) // 2]:g}")
    activity(ax[1, 1], D[Tp[-1]], title=f"(e) activity map, T = {Tp[-1]:g}")
    L, F = D[Ts[0]]["L"], D[Ts[0]]["F"]
    fig.suptitle(f"Finite temperature, one sample, L = {L}, f = {F / Fc:.2g}, $l_{{max}}$ = 50, "
                 f"U(l) = {U0:g} $l^{{{th:.3g}}}$; steady state (advance > {SKIP})")
    fig.tight_layout()
    fig.savefig(f"{figs}/finite_t.png", dpi=110)
    for T, v in zip(Tp, lnv):
        e = D[T]["ev"][D[T]["ss"]]
        print(f"T = {T:g}: {len(e)} events, mean l {e[:, 1].mean():.1f}, mean area {e[:, 5].mean():.1f}, ln v {v:.2f}")
    print(f"Arrhenius slope (effective barrier): {-b[0]:.3f}")


if __name__ == "__main__":
    runs, figs = sys.argv[1], sys.argv[2]
    Fc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.927  # F_c of the demo sample (M = 4096)
    os.makedirs(figs, exist_ok=True)
    t0(runs, figs, Fc)
    finite_t(runs, figs, Fc)
