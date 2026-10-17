"""Figures of the demo (demo/run_demo.sh). Usage: python3 plot_demo.py RUNS_DIR FIG_DIR [F_c]

t0_creep.png: T -> 0+ at several forces (one sample, L = 1024): activity map and metastable states
  at the lowest force, structure factor with the scaling of the equilibrium-depinning crossover,
  size of the rearrangements against f, and the statistics of the events.
finite_t.png: F = 0.093 at several temperatures: structure factor, window sizes, velocity against
  1/T, activity maps and event areas.
velocity.png: f = 0.03, 0.1, 0.3 and T = 0.02 .. 0.1: velocity v(f, T), creep law and effective barrier.
persistence.png (L = 512, F_c = 0.949): persistence maps, persistence pi(tau) and four-point
  susceptibility chi4(tau) (Russo et al.), relaxation time tau_alpha and the maximum of chi4.

Steady state: the states and events after the line has advanced by SKIP.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SKIP = 300
FC512 = 0.9487  # F_c of the L = 512 sample of the persistence runs (M = 4096)
ZEQ, ZDEP = 2 / 3, 1.25
NUEQ = 1 / (2 - ZEQ)  # 3/4


def load(d):
    par = {}
    for l in open(f"{d}/params.txt"):
        k, v = l.split(None, 1)
        par[k] = v.split()[0]
    L = int(par["L"])
    ev = np.loadtxt(f"{d}/events.dat", ndmin=2)
    cf = f"{d}/configs.bin"
    c = np.fromfile(cf, dtype=np.int32) if os.path.exists(cf) else np.zeros(L, np.int32)
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
    """Extent of n consecutive steady-state events: the arc [x_left, x_left + extent) at height step."""
    ev = D["ev"][D["ss"]][:n]
    L = D["L"]
    xs, ys = [], []
    for k, e in enumerate(ev):
        x0, w = int(e[10]), int(e[11])
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
    activity(ax[1, 0], D[Tp[0]], title=f"(d) activity map, T = {Tp[0]:g}")
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


def summary(d):
    out = {}
    for l in open(f"{d}/summary.txt"):
        k, v = l.split()
        out[k] = float(v)
    return out


def load_vt(runs, prefix):
    R = {}
    for d in os.listdir(runs):
        if d.startswith(prefix):
            F, T = d[len(prefix):].split("_T")
            pf = f"{runs}/{d}/persistence.dat"
            R[(float(F), float(T))] = dict(d=f"{runs}/{d}", s=summary(f"{runs}/{d}"),
                                          p=np.loadtxt(pf) if os.path.exists(pf) else None)
    return R


def velocity(runs, figs, Fc):
    R = load_vt(runs, "vT_F")
    Fs = sorted({F for F, _ in R})
    Ts = sorted({T for _, T in R})
    cf = dict(zip(Fs, plt.cm.viridis(np.linspace(0, 0.85, len(Fs)))))
    ct = dict(zip(Ts, plt.cm.plasma(np.linspace(0, 0.85, len(Ts)))))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    Ueff = {}
    for F in Fs:
        T = np.array([T for T in Ts if (F, T) in R])
        lnv = np.array([R[(F, t)]["s"]["ln_v"] for t in T])
        b = np.polyfit(1 / T, lnv, 1)
        Ueff[F] = -b[0]
        ax[0].plot(1 / T, lnv, "o-", color=cf[F], label=f"f = {F / Fc:.2g}: slope {-b[0]:.2f}")
    ax[0].set(xlabel="1/T", ylabel="ln v  ($\\nu_0$ = 1)", title="(a) velocity against 1/T")
    ax[0].legend(fontsize=8)
    for T in Ts:
        F = np.array([F for F in Fs if (F, T) in R])
        ax[1].plot((F / Fc) ** -0.25, [T * R[(f, T)]["s"]["ln_v"] for f in F], "o-", color=ct[T], label=f"T = {T:g}")
    ax[1].set(xlabel="$f^{-1/4}$", ylabel="T ln v", title="(b) creep law: T ln v against $f^{-1/4}$")
    ax[1].legend(fontsize=8)
    f = np.array(Fs) / Fc
    ax[2].loglog(f, [Ueff[F] for F in Fs], "ko-", label="$-d \\ln v / d(1/T)$")
    ax[2].loglog(f, Ueff[Fs[1]] * (f / f[1]) ** -0.25, "C3--", label="$f^{-1/4}$")
    ax[2].set(xlabel="$f = F/F_c$", ylabel="effective barrier", title="(c) effective barrier against f")
    ax[2].legend(fontsize=8)
    fig.suptitle("Mean velocity v(f, T), one sample, L = 1024, $l_{max}$ = 50, U(l) = $l^{1/3}$; steady state (advance 300 to 1500)")
    fig.tight_layout()
    fig.savefig(f"{figs}/velocity.png", dpi=110)
    for F in Fs:
        print(f"f = {F / Fc:.2g}: ln v " + ", ".join(f"T={T:g}: {R[(F, T)]['s']['ln_v']:.2f}" for T in Ts if (F, T) in R)
              + f"; effective barrier {Ueff[F]:.2f}")


def read_moves(d):
    """moves.bin: per event the time and the runs (start, length) of moved sites."""
    raw = open(f"{d}/moves.bin", "rb").read()
    pos, out = 0, []
    while pos < len(raw):
        t = np.frombuffer(raw, np.float64, 1, pos)[0]
        n = int(np.frombuffer(raw, np.int32, 1, pos + 8)[0])
        out.append((t, np.frombuffer(raw, np.int32, 2 * n, pos + 12).reshape(n, 2)))
        pos += 12 + 8 * n
    return out


def persistence_map(ax, r, tau_a, title):
    """Time to the first move of every site after one origin t0 in the steady state, in units of tau_alpha."""
    D = load(r["d"])
    L = D["L"]
    mv = read_moves(r["d"])
    k0 = int(np.argmax(D["adv"] > 600))  # an origin well inside the steady state
    t0 = mv[k0][0]
    first = np.full(L, np.inf)
    for t, runs in mv[k0 + 1:]:
        for a, w in runs:
            seg = first[a:a + w]
            seg[np.isinf(seg)] = t - t0
        if t - t0 > 100 * tau_a:
            break
    y = np.minimum(first / tau_a, 1e3)
    ax.fill_between(np.arange(L), 1e-3, y, step="mid", color="C0", lw=0)
    ax.axhline(1, color="k", lw=0.6)
    ax.set(yscale="log", ylim=(1e-3, 1e2), xlim=(0, L), xlabel="$x$", ylabel="time to the first move / $\\tau_\\alpha$",
           title=title)


def persistence(runs, figs, Fc):
    R = load_vt(runs, "pT_F")
    Fs = sorted({F for F, _ in R})
    Ts = sorted({T for _, T in R})
    cf = dict(zip(Fs, plt.cm.viridis(np.linspace(0, 0.85, len(Fs)))))
    ct = dict(zip(Ts, plt.cm.plasma(np.linspace(0, 0.85, len(Ts)))))
    F1 = min(Fs, key=lambda F: abs(F / Fc - 0.1))
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    for a, T in zip(ax[0], [Ts[0], Ts[len(Ts) // 2], Ts[-1]]):
        persistence_map(a, R[(F1, T)], R[(F1, T)]["s"]["tau_alpha"],
                        f"persistence map, f = {F1 / Fc:.2g}, T = {T:g}\n(sites not moved yet lie under the curve)")
    for T in Ts:
        if (F1, T) not in R:
            continue
        P, ta = R[(F1, T)]["p"], R[(F1, T)]["s"]["tau_alpha"]
        ax[0, 3].semilogx(P[:, 0], P[:, 1], color=ct[T], label=f"T = {T:g}")
        ax[1, 0].semilogx(P[:, 0] / ta, P[:, 1], color=ct[T], label=f"T = {T:g}")
        ax[1, 1].semilogx(P[:, 0] / ta, P[:, 3], color=ct[T], label=f"T = {T:g}")
    ax[0, 3].set(xlabel="$\\tau$  (1/$\\nu_0$)", ylabel="$\\langle\\pi(\\tau)\\rangle$", title=f"persistence, f = {F1 / Fc:.2g}")
    ax[0, 3].legend(fontsize=7)
    ax[1, 0].axhline(0.5, color="k", lw=0.5)
    ax[1, 0].set(xlabel="$\\tau / \\tau_\\alpha$", ylabel="$\\langle\\pi(\\tau)\\rangle$", title=f"persistence against $\\tau/\\tau_\\alpha$, f = {F1 / Fc:.2g}")
    ax[1, 0].legend(fontsize=7)
    ax[1, 1].set(xlabel="$\\tau / \\tau_\\alpha$", ylabel="$\\chi_4(\\tau)$", title=f"four-point susceptibility, f = {F1 / Fc:.2g}")
    ax[1, 1].legend(fontsize=7)
    for F in Fs:
        T = np.array([T for T in Ts if (F, T) in R])
        ta = np.array([R[(F, t)]["s"]["tau_alpha"] for t in T])
        ok = ta > 0
        b = np.polyfit(1 / T[ok], np.log(ta[ok]), 1)
        ax[1, 2].semilogy(1 / T[ok], ta[ok], "o-", color=cf[F], label=f"f = {F / Fc:.2g}: slope {b[0]:.2f}")
        ax[1, 3].loglog(T, [R[(F, t)]["s"]["chi4_max"] for t in T], "o-", color=cf[F], label=f"f = {F / Fc:.2g}")
    ax[1, 2].set(xlabel="1/T", ylabel="$\\tau_\\alpha$  (1/$\\nu_0$)", title="relaxation time, $\\langle\\pi(\\tau_\\alpha)\\rangle$ = 1/2")
    ax[1, 2].legend(fontsize=8)
    T = np.array(Ts)
    c0 = R[(F1, Ts[0])]["s"]["chi4_max"]
    ax[1, 3].loglog(T, c0 * (T / T[0]) ** (-4 / 3), "k--", label="$T^{-\\nu_{dep}}$, $\\nu_{dep}$ = 4/3")
    ax[1, 3].set(xlabel="T", ylabel="$\\chi_4^{max}$", title="maximum of $\\chi_4$")
    ax[1, 3].legend(fontsize=8)
    fig.suptitle(f"Persistence and dynamical heterogeneity (Russo et al.), one sample, L = 512, $F_c$ = {Fc:.3g}, "
                 "$l_{max}$ = 50, U(l) = $l^{1/3}$; steady state (advance 300 to 6000)")
    fig.tight_layout()
    fig.savefig(f"{figs}/persistence.png", dpi=110)
    for F in Fs:
        print(f"f = {F / Fc:.2g}: " + ", ".join(f"T={T:g}: tau_a {R[(F, T)]['s']['tau_alpha']:.3g} chi4max {R[(F, T)]['s']['chi4_max']:.1f}"
                                             for T in Ts if (F, T) in R))


if __name__ == "__main__":
    runs, figs = sys.argv[1], sys.argv[2]
    Fc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.927  # F_c of the demo sample (M = 4096)
    os.makedirs(figs, exist_ok=True)
    t0(runs, figs, Fc)
    finite_t(runs, figs, Fc)
    velocity(runs, figs, Fc)
    persistence(runs, figs, FC512)
