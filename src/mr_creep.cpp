// mr_creep: creep of a directed elastic line by activated rearrangements, at T -> 0+ and at
// finite temperature.
//
// T -> 0+: algorithm of E. E. Ferrero, L. Foini, T. Giamarchi, A. B. Kolton and A. Rosso,
//   PRL 118, 147208 (2017), Supplemental Material: an approximation of the exact transition
//   pathways of A. B. Kolton, A. Rosso, T. Giamarchi and W. Krauth, PRB 79, 184207 (2009).
// T > 0: its extension by G. Russo, E. E. Ferrero, A. B. Kolton, A. Rosso and D. Vandembroucq,
//   arXiv:2604.17600 (2026), a kinetic Monte Carlo over all the energy-lowering rearrangements.
//
// Model: u_i integer, i = 0..L-1 periodic, u_i in a random potential periodic in u with period M
// (M = 0: not periodic),
//   E = sum_i [ V(i, u_i mod M) - F u_i + (c/2) (u_{i+1} - u_i)^2 ],  |u_{i+1} - u_i| <= mcut.
// The Gaussian disorder V(i, u) is generated on demand from the counter (i, u) and the seed with
// the Philox4x32-10 generator of Random123, so it is not stored (default), or stored as a table of
// Mersenne Twister numbers as in krgk_creep (--rng mt): with the same seed both codes then see
// the same sample.
//
// Each creep step goes from a metastable state alpha to the next one, gamma, in two steps:
// (i) activation: the smallest compact rearrangement that lowers the energy. For every window
//     [i, i+l] the end points u_i, u_{i+l} are kept fixed and the optimal path between them is
//     computed (forward: u'_j >= u_j); the rearrangement is the one of the smallest l for which
//     the optimal path has a lower energy than the line. Among windows of that l the one that
//     lowers the energy most is taken.
//     At T > 0 every window [i, i+l], l <= lmax, whose optimal path lowers the energy is a
//     candidate, with the barrier U(l) = U0 l^theta and the rate exp(-U(l)/T); one candidate is
//     drawn with probability proportional to its rate (kinetic Monte Carlo) and the time
//     advances by -ln(r)/R, R the total rate.
//     The windows are independent: they are computed in parallel (OpenMP) and, after a step,
//     only the windows that touch the moved sites are computed again.
// (ii) deterministic relaxation: forward elementary moves (rigid displacement by +1 of a block
//     of consecutive sites), the shortest favourable block first, as in krgk_creep.
// The method gives the sequence of metastable states and the events, not the energy barriers;
// at T > 0 the time follows from the assumed barriers U(l).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <Random123/philox.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static const double BIG = 1e300;
static const double PRECISION_FC = 0.005;  // step in F when locating F_c (as in krgk_creep)
static const double EPS = 1e-9;            // smallest energy gain accepted

struct Params {
	int L = 64, M = 1024, mcut = 1;
	double disorder = 1.0, c = 2.0, f = 0.5;
	bool f_absolute = false;
	unsigned long seed = 12345;
	bool philox = true;       // disorder on demand (Random123 Philox); false: Mersenne Twister table
	int xi_x = 1, xi_u = 1;   // Philox disorder constant on rectangles of xi_x sites by xi_u in u
	long events = 100;
	double advance = 0;       // stop earlier, once the mean position has advanced by this much
	int lmax = 0;             // largest window (0: L - 1 at T = 0, min(50, L - 1) at T > 0)
	double T = 0;             // temperature (0: the T -> 0+ limit)
	double U0 = 1.0, theta = 1.0 / 3.0;  // barrier of a rearrangement of window l: U0 l^theta
	unsigned long kmc_seed = 1;          // random numbers of the kinetic Monte Carlo
	int conf_every = 0;       // write the metastable configuration every n events (0: never)
	bool save_events = false; // write alpha, the activated configuration and gamma of every event
	bool save_moves = false;  // write the time and the moved sites (as runs) of every event to moves.bin
	double skip = 0;          // steady state after this advance: velocity (and persistence) measured there
	double persistence = 0;   // T > 0: persistence and chi4 up to tau = persistence / v (0: off)
	int origins = 10;         // persistence: origins opened per tau_max
	double checkpoint = 0;    // write checkpoint.bin every this many seconds of wall time (0: never)
	double walltime = 0;      // stop after this many seconds with a checkpoint, exit status 3 (0: off)
	bool restart = false;     // resume from checkpoint.bin in --out if present
	std::string out = ".";
};

// Mersenne Twister and Gaussian deviates as in krgk_creep (and the original code):
// genrand_real3 in (0,1) and the polar Box-Muller of Numerical Recipes.
class Random {
	std::mt19937 mt;
	bool have = false;
	double saved = 0.0;
	double real3() { return ((double)mt() + 0.5) * (1.0 / 4294967296.0); }
public:
	explicit Random(unsigned long s) : mt((uint32_t)s) {}
	double gauss()
	{
		if (have) { have = false; return saved; }
		double v1, v2, r2;
		do {
			v1 = 2.0 * real3() - 1.0;
			v2 = 2.0 * real3() - 1.0;
			r2 = v1 * v1 + v2 * v2;
		} while (r2 >= 1.0 || r2 == 0.0);
		double fac = std::sqrt(-2.0 * std::log(r2) / r2);
		saved = v1 * fac; have = true;
		return v2 * fac;
	}
};

class Line {
public:
	int L, M, mcut, lmax;
	double c, F = 0.0;
	double T, U0, theta;
	std::mt19937_64 kmc;
	bool philox;
	int xi_x, xi_u;
	double strength;
	r123::Philox4x32::key_type key;
	std::vector<double> V;  // V[i*M + j], table of the Mersenne Twister disorder (--rng mt)
	// Philox disorder: cache of V(s, x) for x in [base_s, base_s + W) at each site, refilled when
	// the line leaves it (rebase); outside the band V is computed directly
	int W = 0;
	std::vector<double> band;
	std::vector<int> base;
	std::vector<int> u;

	Line(const Params &p) : L(p.L), M(p.M), mcut(p.mcut),
		lmax(p.lmax > 0 ? p.lmax : (p.T > 0 ? std::min(50, p.L - 1) : p.L - 1)), c(p.c),
		T(p.T), U0(p.U0), theta(p.theta), kmc(p.kmc_seed),
		philox(p.philox), xi_x(p.xi_x), xi_u(p.xi_u), strength(p.disorder), key{{(uint32_t)p.seed, (uint32_t)(p.seed >> 32)}},
		V(p.philox ? 0 : (size_t)p.L * p.M), u(p.L, 0), dirty(p.L, 1), ds(p.L), P(2 * p.L + 1), D(2 * p.L + 1),
		cand(p.L), valid(p.L, 0)
	{
		if (philox) {
			int Kb = std::min(lmax * mcut, 64);
			W = 3 * Kb + 4;
			band.resize((size_t)L * W);
			base.assign(L, INT32_MIN / 2);
			return;
		}
		Random rng(p.seed);
		for (auto &v : V) v = rng.gauss();
		double mean = 0.0;
		for (double v : V) mean += v;
		mean /= V.size();
		for (auto &v : V) v = (v - mean) * p.disorder;
	}

	double pot(int i, int j) const
	{
		if (philox) {
			unsigned x = (unsigned)(j - base[i]);
			if (x < (unsigned)W) return band[(size_t)i * W + x];
		}
		return pot_raw(i, j);
	}

	// keep u_s - 1 .. u_s + 2 Kb + 1, the values the windows and the moves read, in the band
	void rebase()
	{
		if (!philox) return;
		int Kb = (W - 4) / 3;
		for (int s = 0; s < L; s++) {
			if (u[s] - 1 >= base[s] && u[s] + 2 * Kb + 2 <= base[s] + W) continue;
			base[s] = u[s] - 1;
			for (int x = 0; x < W; x++) band[(size_t)s * W + x] = pot_raw(s, base[s] + x);
		}
	}

	double pot_raw(int i, int j) const
	{
		int m = j;
		if (M > 0) { m = j % M; if (m < 0) m += M; }
		if (!philox) return V[(size_t)i * M + m];
		// one Gaussian amplitude per rectangle of xi_x sites by xi_u in u (floor division), by
		// Box-Muller from two 64-bit uniforms in (0, 1) made of the four 32-bit outputs
		int bi = i / xi_x, bu = m >= 0 ? m / xi_u : -((-m - 1) / xi_u) - 1;
		r123::Philox4x32::ctr_type ctr = {{(uint32_t)bi, (uint32_t)bu, 0, 0}};
		auto r = r123::Philox4x32()(ctr, key);
		const double k = 1.0 / 9007199254740992.0;  // 2^-53
		double u1 = ((double)((((uint64_t)r[0] << 32) | r[1]) >> 11) + 0.5) * k;
		double u2 = ((double)((((uint64_t)r[2] << 32) | r[3]) >> 11) + 0.5) * k;
		return strength * std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
	}
	double site(int i, int x) const { return pot(i, x) - F * x; }

	double energy(const std::vector<int> &x) const
	{
		double E = 0.0;
		for (int i = 0; i < L; i++) {
			int d = x[i + 1 == L ? 0 : i + 1] - x[i];
			E = E + site(i, x[i]) + c * 0.5 * d * d;
		}
		return E;
	}

	// ---- deterministic relaxation (elementary moves) ----
	// A block move (i, il) displaces sites i+1 .. i+il-1 by dir with sites i and i+il fixed
	// (il = 2..L); il = L+1 is the rigid move of the whole line. Same moves and order as
	// krgk_creep: the first favourable move in the order (il, i) is applied. Block energies come
	// from prefix sums of the site terms; blocks that touch no site changed since the last
	// metastable state (dirty) cannot have become favourable and are skipped.
	std::vector<char> dirty;
	std::vector<double> ds, P;  // site-term change of each site, and its prefix sums (2L)
	std::vector<int> D;         // prefix counts of dirty sites (2L)
	int pdir = 0;

	void touch_all() { std::fill(dirty.begin(), dirty.end(), 1); }

	void prefix(int dir)
	{
		rebase();  // the moves read V at u and u +- 1
		for (int s = 0; s < L; s++) ds[s] = site(s, u[s] + dir) - site(s, u[s]);
		P[0] = 0; D[0] = 0;
		for (int s = 0; s < 2 * L; s++) { P[s + 1] = P[s] + ds[s % L]; D[s + 1] = D[s] + dirty[s % L]; }
		pdir = dir;
	}

	// Lowest energy change over all block moves (i, il), 2 <= il <= L, and the rigid move, in O(L).
	// With e = i + il the energy change is separable, d = A(i) + B(e):
	//   A(i) = c (1 + dir (u_{i+1} - u_i)) - P[i+1],  B(e) = c dir (u_{e-1} - u_e) + P[e],
	// each term defined only if its border slope stays within mcut; the minimum over
	// e in [i+2, i+L] is a sliding-window minimum.
	double lowest_move(int dir) const
	{
		auto at = [&](int s) { return u[s % L]; };
		std::vector<double> B(2 * L);
		for (int e = 2; e < 2 * L; e++)
			B[e] = std::abs(at(e - 1) + dir - at(e)) > mcut ? BIG : c * dir * (at(e - 1) - at(e)) + P[e];
		double best = P[L];
		std::vector<int> q(2 * L);  // deque of indices e with increasing B
		int h = 0, t = 0, next = 2;
		for (int i = 0; i < L; i++) {
			for (; next <= i + L; next++) {
				while (t > h && B[q[t - 1]] >= B[next]) t--;
				q[t++] = next;
			}
			while (q[h] < i + 2) h++;
			if (std::abs(at(i + 1) + dir - u[i]) > mcut || B[q[h]] >= BIG) continue;
			best = std::min(best, c * (1.0 + dir * (at(i + 1) - u[i])) - P[i + 1] + B[q[h]]);
		}
		return best;
	}

	// One elementary move in direction dir. Returns the smallest energy change per moved site
	// among the moves tried (all moves if full); it is negative if and only if a move was made.
	double move(int dir, bool full)
	{
		prefix(dir);
		if (!full && (D[L] == 0 || lowest_move(dir) > 1e-9)) return BIG;  // no favourable move
		double dmin = BIG;
		for (int il = 2; il <= L + 1; il++) {
			int nstart = (il <= L) ? L : 1;
			for (int i = 0; i < nstart; i++) {
				double d;
				if (il <= L) {
					if (!full && D[i + il + 1] - D[i] == 0) continue;  // sites i .. i+il clean
					int istop = (i + il) % L, a = u[(i + 1) % L], b = u[(istop - 1 + L) % L];
					if (std::abs(a + dir - u[i]) > mcut || std::abs(b + dir - u[istop]) > mcut) continue;
					d = c * (1.0 + dir * (a - u[i] + b - u[istop])) + P[i + il] - P[i + 1];
				} else d = P[L];
				dmin = std::min(dmin, d / (il - 1));
				if (d < 0) {
					int n = (il <= L) ? il - 1 : L;
					for (int j = 1; j <= n; j++) { int s = (i + j) % L; u[s] += dir; dirty[s] = 1; }
					if (il <= L) { dirty[i] = 1; dirty[(i + il) % L] = 1; }
					return dmin;
				}
			}
		}
		return dmin;
	}

	void relax_forward()
	{
		while (move(+1, false) < 0.0) {}
		std::fill(dirty.begin(), dirty.end(), 0);  // metastable: no favourable move left
	}

	// Critical force with the procedure of krgk_creep (find_critical of the original): starting
	// from u = 0 and the initial F, relax backward; then advance forward by 2M, raising F by
	// (dmin + PRECISION_FC) whenever no forward move lowers the energy. F_c = F - PRECISION_FC.
	// The critical configuration (crit) is the one that blocked the line at the last raise of F.
	double critical_force(double F0, std::vector<int> &crit)
	{
		F = F0;
		std::fill(u.begin(), u.end(), 0);
		while (move(-1, true) < 0.0) {}
		F -= 5.0 * PRECISION_FC;
		int count = 0;
		while (count == 0) {
			std::vector<int> u0 = u;
			auto advanced = [&]() { int m = u[0] - u0[0]; for (int i = 1; i < L; i++) m = std::min(m, u[i] - u0[i]); return m; };
			while (advanced() < 2 * M) {
				double d = move(+1, true);
				if (d >= 0.0) { F += d + PRECISION_FC; count++; crit = u; }
			}
			if (count == 0) { F -= 5.0 * PRECISION_FC; for (auto &x : u) x -= 2 * M; }
		}
		touch_all();
		return F - PRECISION_FC;
	}

	// ---- activation: minimal rearrangement ----
	// Optimal forward path from site i (u'_i = u_i) by transfer matrix: after j steps, C[k] is the
	// lowest energy of the bonds i .. i+j and of the sites i+1 .. i+j-1, with u'_{i+j} = u_i + k
	// and u'_s >= u_s. The window [i, i+l] improves if C[u_{i+l} - u_i] (end point fixed) is below
	// the energy of the line on the same bonds and sites. Cost O(l^2 mcut^2) up to window l.
	struct Window { int l = 0; double gain = 0; };
	// candidates of the windows starting at i: at T = 0 the smallest improving window (none if no
	// window up to lmax improves), at T > 0 all of them; R is the sum of their rates
	struct Candidates { std::vector<Window> w; double R = 0; };
	std::vector<Candidates> cand;
	std::vector<char> valid;
	double dt = 0;  // time of the last step (T > 0), in units of exp(U(2)/T)/nu0
	int last_i = 0; // start of the window of the last step

	// rate exp(-U(l)/T) relative to that of the smallest window, exp(-U(2)/T)
	double rate(int l) const { return std::exp(-U0 * (std::pow((double)l, theta) - std::pow(2.0, theta)) / T); }

	// smallest improving window starting at i (l = 0: none up to lm, default lmax); with all,
	// every improving window up to lm; with path, the optimal path of window l0
	Window scan(int i, int l0 = 0, std::vector<int> *path = nullptr, std::vector<Window> *all = nullptr, int lm = 0) const
	{
		const int lend = l0 ? l0 : (lm ? lm : lmax);
		const int K = lend * mcut, n = 2 * K + 1, a0 = u[i];
		std::vector<double> C(n, BIG), T(n);
		std::vector<signed char> from(path ? (size_t)lend * n : 0);
		C[K] = 0.0;
		// C is finite only on [clo, chi]: the forward paths stay at or above the line, so the
		// range follows the line from below and widens by mcut per step above. A path at step l
		// is useful only if it can still reach the end point of a later window l' <= lend, which
		// bounds it from above by top[l] = max_{l' >= l} (ke(l') + mcut (l' - l)) (ke: the line)
		std::vector<int> top(lend + 1);
		top[lend] = u[(i + lend) % L] - a0 + K;
		for (int l = lend - 1; l >= 1; l--) top[l] = std::max(u[(i + l) % L] - a0 + K, top[l + 1] + mcut);
		int clo = K, chi = K;
		double cur = 0.0;  // energy of the line on the same bonds and interior sites
		for (int l = 1; l <= lend; l++) {
			int s = (i + l) % L, sp = (i + l - 1) % L;
			if (l > 1) cur += site(sp, u[sp]);
			int db = u[s] - u[sp];
			cur += c * 0.5 * db * db;
			int tlo = std::max(0, clo - mcut), thi = std::min({n - 1, chi + mcut, top[l]});
			for (int k = tlo; k <= thi; k++) {
				double best = BIG; int arg = 0;
				for (int d = -mcut; d <= mcut; d++) {
					int k0 = k - d;
					if (k0 < clo || k0 > chi || C[k0] >= BIG) continue;
					double v = C[k0] + c * 0.5 * d * d;
					if (v < best) { best = v; arg = d; }
				}
				T[k] = best;
				if (path) from[(size_t)(l - 1) * n + k] = (signed char)arg;
			}
			int ke = u[s] - a0 + K;
			if (!l0 && l >= 2 && ke >= tlo && ke <= thi && T[ke] < cur - EPS) {
				if (!all) return {l, cur - T[ke]};
				all->push_back({l, cur - T[ke]});
			}
			if (l == lend) break;
			int nlo = std::max(tlo, ke);  // u'_s >= u_s
			for (int k = clo; k <= chi; k++) C[k] = BIG;
			for (int k = nlo; k <= thi; k++) C[k] = T[k] >= BIG ? BIG : T[k] + site(s, a0 + k - K);
			clo = nlo; chi = thi;
			if (clo > chi) break;  // the line rose beyond every path: no forward path left
		}
		if (path) {
			int k = u[(i + lend) % L] - a0 + K;
			for (int l = lend; l >= 1; l--) {
				k -= from[(size_t)(l - 1) * n + k];
				if (l > 1) (*path)[(i + l - 1) % L] = a0 + k - K;
			}
		}
		return {};
	}

	// candidates of the windows starting at i, up to lm
	void candidates(int i, Candidates &C, int lm) const
	{
		C.w.clear(); C.R = 0;
		if (T > 0) {
			scan(i, 0, nullptr, &C.w, lm);
			for (const Window &w : C.w) C.R += rate(w.l);
		} else {
			Window w = scan(i, 0, nullptr, nullptr, lm);
			if (w.l) C.w.push_back(w);
		}
	}

	// the rearrangement among the candidates cd: its start (-1 if none) and window w
	int choose(const std::vector<Candidates> &cd, Window &w)
	{
		int ib = -1;
		if (T > 0) {
			// kinetic Monte Carlo: window i with probability R_i / R, then l with probability rate(l) / R_i
			double R = 0;
			for (int i = 0; i < L; i++) R += cd[i].R;
			if (R <= 0) return -1;
			std::uniform_real_distribution<double> uni(0.0, 1.0);
			double x = uni(kmc) * R;
			for (ib = 0; ib < L - 1 && (x >= cd[ib].R || cd[ib].w.empty()); ib++) x -= cd[ib].R;
			while (cd[ib].w.empty()) ib--;  // rounding at the end of the sum
			const auto &cw = cd[ib].w;
			size_t k = 0;
			for (; k + 1 < cw.size() && x >= rate(cw[k].l); k++) x -= rate(cw[k].l);
			w = cw[k];
			dt = -std::log(1.0 - uni(kmc)) / R;
		} else {
			for (int i = 0; i < L; i++) {
				if (cd[i].w.empty()) continue;
				const Window &c = cd[i].w[0];
				if (ib < 0 || c.l < w.l || (c.l == w.l && c.gain > w.gain)) { ib = i; w = c; }
			}
		}
		return ib;
	}

	long extended = 0;  // steps with no candidate up to lmax, taken among larger windows

	// One creep step from the metastable state u. On return u is gamma and beta is the activated
	// configuration (before relaxation). Returns the window, or l = 0 if no window lowers the energy.
	// If no window up to lmax lowers the energy, this step looks for windows up to 2 lmax, 4 lmax,
	// ... (at most L - 1): candidates beyond lmax are neglected only when smaller ones exist.
	Window creep_step(std::vector<int> &beta)
	{
		rebase();
		std::vector<int> todo;
		for (int i = 0; i < L; i++) if (!valid[i]) todo.push_back(i);
		#pragma omp parallel for schedule(dynamic, 4)
		for (size_t t = 0; t < todo.size(); t++) {
			candidates(todo[t], cand[todo[t]], lmax);
			valid[todo[t]] = 1;
		}
		Window w;
		int ib = choose(cand, w);
		for (int lm = lmax; ib < 0 && lm < L - 1;) {
			lm = std::min(2 * lm, L - 1);
			std::vector<Candidates> ext(L);
			#pragma omp parallel for schedule(dynamic, 4)
			for (int i = 0; i < L; i++) candidates(i, ext[i], lm);
			ib = choose(ext, w);
			extended++;
		}
		if (ib < 0) return {};
		std::vector<int> alpha = u;
		scan(ib, w.l, &u);
		beta = u;
		for (int s = 0; s < L; s++) dirty[s] = u[s] != alpha[s];
		relax_forward();
		// invalidate the windows [i, i + reach] that contain a moved site
		std::vector<int> moved(2 * L + 1, 0);
		for (int s = 0; s < 2 * L; s++) moved[s + 1] = moved[s] + (u[s % L] != alpha[s % L]);
		for (int i = 0; i < L; i++) {
			int reach = (T == 0 && !cand[i].w.empty()) ? cand[i].w[0].l : lmax;
			if (moved[i + reach + 1] - moved[i] > 0) valid[i] = 0;
		}
		last_i = ib;
		return w;
	}
};

// Persistence and four-point susceptibility of G. Russo et al., arXiv:2604.17600:
//   pi(tau) = (1/L) sum_i p_i(tau), p_i = 1 if site i does not move in [t0, t0 + tau],
//   averaged over origins t0 of the steady state, and chi4(tau) = L (<pi^2> - <pi>^2).
// Its spatial resolution is the four-point structure factor
//   S4(q, tau) = (1/L) <|sum_i p_i(tau) e^{i q i}|^2>,  q = 2 pi n / L, n = 1 .. L/2,
// on the modes n = 1 .. 16 and then 8 per decade; S4(q -> 0) -> chi4 when the line is much
// longer than the correlation length of p_i. For q != 0 the sum over all sites vanishes, so the
// sum over p_i is minus the sum over the sites moved by tau, accumulated in the order of their
// first move: O(L n_q) per origin.
// Origins are opened every tmax/N (N = 10 by default); each keeps the time of the first move of every site after
// t0, and is closed, adding pi, pi^2 and S4 on the grid of tau, once t > t0 + tmax.
class Persistence {
public:
	struct Origin { double t0; std::vector<double> first; };
private:
	int L;
	std::vector<double> cs, sn;  // cos and sin of 2 pi k / L
	void close(const Origin &o)
	{
		std::vector<std::pair<double, int>> f;
		for (int i = 0; i < L; i++) if (o.first[i] <= tmax) f.push_back({o.first[i], i});
		std::sort(f.begin(), f.end());
		const size_t nq = modes.size();
		std::vector<double> re(nq, 0), im(nq, 0);
		size_t k = 0;
		for (size_t b = 0; b < tau.size(); b++) {
			for (; k < f.size() && f[k].first <= tau[b]; k++)
				for (size_t m = 0; m < nq; m++) {
					long x = (long)modes[m] * f[k].second % L;
					re[m] += cs[x]; im[m] += sn[x];
				}
			double pi = (double)(L - (long)k) / L;
			S1[b] += pi; S2[b] += pi * pi;
			for (size_t m = 0; m < nq; m++) S4[b * nq + m] += (re[m] * re[m] + im[m] * im[m]) / L;
		}
		n++;
	}
public:
	double tmax = 0, every = 0, next = 0;
	std::deque<Origin> open;
	std::vector<int> modes;
	std::vector<double> tau, S1, S2, S4;  // S4[b * modes.size() + m]
	long n = 0;
	explicit Persistence(int L) : L(L), cs(L), sn(L)
	{
		for (int k = 0; k < L; k++) { cs[k] = std::cos(2 * M_PI * k / L); sn[k] = std::sin(2 * M_PI * k / L); }
		for (int m = 1; m <= std::min(16, L / 2); m++) modes.push_back(m);
		for (int j = 1; ; j++) {
			int m = (int)std::lround(16 * std::pow(10.0, j / 8.0));
			if (m > L / 2) break;
			if (m > modes.back()) modes.push_back(m);
		}
	}
	bool on() const { return tmax > 0; }
	// tau from 1e-3 tau1 to X tau1, 60 values per log scale, N origins per tau_max from time t on
	void start(double t, double tau1, double X, int N = 10)
	{
		tmax = X * tau1;
		int nb = (int)(60 * std::log10(X / 1e-3)) + 1;
		for (int b = 0; b < nb; b++) tau.push_back(1e-3 * tau1 * std::pow(X / 1e-3, (double)b / (nb - 1)));
		S1.assign(nb, 0); S2.assign(nb, 0); S4.assign((size_t)nb * modes.size(), 0);
		every = tmax / N; next = t;
	}
	// an event at time t moved the sites in moved
	void event(double t, const std::vector<int> &moved)
	{
		while (!open.empty() && t > open.front().t0 + tmax) { close(open.front()); open.pop_front(); }
		for (auto &o : open)
			for (int i : moved) if (o.first[i] > tmax) o.first[i] = t - o.t0;
		if (t >= next) { open.push_back({t, std::vector<double>(L, 2 * tmax)}); next = t + every; }
	}
	double chi4(size_t b) const { double m = S1[b] / n; return L * (S2[b] / n - m * m); }
	double s4(size_t b, size_t m) const { return S4[b * modes.size() + m] / n; }
	void save(std::ostream &o) const;
	void load(std::istream &i);
};

// ---- checkpoint: binary dumps of numbers, vectors and strings ----
template <class X> static void put(std::ostream &o, const X &x) { o.write((const char *)&x, sizeof x); }
template <class X> static void put(std::ostream &o, const std::vector<X> &v)
{
	put(o, v.size());
	o.write((const char *)v.data(), sizeof(X) * v.size());
}
static void put(std::ostream &o, const std::string &x) { put(o, x.size()); o.write(x.data(), x.size()); }
template <class X> static void get(std::istream &i, X &x) { i.read((char *)&x, sizeof x); }
template <class X> static void get(std::istream &i, std::vector<X> &v)
{
	size_t n = 0;
	get(i, n);
	v.resize(n);
	i.read((char *)v.data(), sizeof(X) * n);
}
static void get(std::istream &i, std::string &x) { size_t n = 0; get(i, n); x.resize(n); i.read(&x[0], n); }

void Persistence::save(std::ostream &o) const
{
	put(o, tmax); put(o, every); put(o, next); put(o, n);
	put(o, tau); put(o, S1); put(o, S2); put(o, S4);
	put(o, open.size());
	for (const auto &x : open) { put(o, x.t0); put(o, x.first); }
}
void Persistence::load(std::istream &i)
{
	get(i, tmax); get(i, every); get(i, next); get(i, n);
	get(i, tau); get(i, S1); get(i, S2); get(i, S4);
	size_t no = 0;
	get(i, no);
	open.resize(no);
	for (auto &x : open) { get(i, x.t0); get(i, x.first); }
}

static volatile std::sig_atomic_t stop_signal = 0;
static void on_signal(int) { stop_signal = 1; }

static void usage(const char *prog)
{
	std::cerr << "usage: " << prog << " [options]\n"
	          << "  --L N          line length (sites, periodic)                [64]\n"
	          << "  --M N          period of the disorder along u (0: none)    [1024]\n"
	          << "  --mcut N       maximal slope |u_{i+1}-u_i|                 [1]\n"
	          << "  --disorder X   disorder strength (Gaussian, std X)         [1.0]\n"
	          << "  --c X          elastic constant                            [2.0]\n"
	          << "  --f X          drive F/F_c (or F with --absolute)          [0.5]\n"
	          << "  --absolute     --f is the absolute force F (F_c is then not computed)\n"
	          << "  --seed N       seed of the disorder                        [12345]\n"
	          << "  --rng G        disorder: philox (on demand) or mt (table)  [philox]\n"
	          << "  --xi-x A --xi-u B philox disorder constant on A x B rectangles  [1 1]\n"
	          << "  --events N     number of creep steps                       [100]\n"
	          << "  --advance X    stop once the mean position advanced by X   [0: off]\n"
	          << "  --lmax N       largest rearrangement window        [L-1; at T > 0: 50]\n"
	          << "  --T X          temperature (0: the T -> 0+ limit)          [0]\n"
	          << "  --U0 X         barrier prefactor, U(l) = U0 l^theta        [1]\n"
	          << "  --theta X      barrier exponent                            [1/3]\n"
	          << "  --kmc-seed N   seed of the kinetic Monte Carlo             [1]\n"
	          << "  --conf-every N write the metastable state every N steps    [0: never]\n"
	          << "  --save-events  write alpha, activated and gamma of every step to events.bin\n"
	          << "  --save-moves   write the time and the moved sites of every step to moves.bin\n"
	          << "  --skip X       steady state after an advance X: velocity in summary.txt [0]\n"
	          << "  --persistence X  T > 0: persistence and chi4 up to tau = X/v (needs --skip) [0: off]\n"
	          << "  --origins N    persistence: origins opened per tau_max (each keeps L doubles) [10]\n"
	          << "  --checkpoint S write checkpoint.bin every S seconds of wall time [0: never]\n"
	          << "  --walltime S   after S seconds (or on SIGTERM, SIGINT, SIGUSR1 if checkpointing)\n"
	          << "                 write checkpoint.bin and exit with status 3  [0: off]\n"
	          << "  --restart      resume from DIR/checkpoint.bin if present (exit 0 if the run is done)\n"
	          << "  --out DIR      output directory                            [.]\n"
	          << "Threads: OMP_NUM_THREADS.\n";
}

int main(int argc, char **argv)
{
	Params p;
	for (int a = 1; a < argc; a++) {
		std::string o = argv[a];
		auto val = [&]() { if (a + 1 >= argc) { usage(argv[0]); std::exit(1); } return std::string(argv[++a]); };
		if (o == "--L") p.L = std::stoi(val());
		else if (o == "--M") p.M = std::stoi(val());
		else if (o == "--mcut") p.mcut = std::stoi(val());
		else if (o == "--disorder") p.disorder = std::stod(val());
		else if (o == "--c") p.c = std::stod(val());
		else if (o == "--f") p.f = std::stod(val());
		else if (o == "--absolute") p.f_absolute = true;
		else if (o == "--seed") p.seed = std::stoul(val());
		else if (o == "--rng") {
			std::string g = val();
			if (g != "philox" && g != "mt") { usage(argv[0]); return 1; }
			p.philox = g == "philox";
		}
		else if (o == "--events") p.events = std::stol(val());
		else if (o == "--advance") p.advance = std::stod(val());
		else if (o == "--lmax") p.lmax = std::stoi(val());
		else if (o == "--xi-x") p.xi_x = std::stoi(val());
		else if (o == "--xi-u") p.xi_u = std::stoi(val());
		else if (o == "--T") p.T = std::stod(val());
		else if (o == "--U0") p.U0 = std::stod(val());
		else if (o == "--theta") p.theta = std::stod(val());
		else if (o == "--kmc-seed") p.kmc_seed = std::stoul(val());
		else if (o == "--conf-every") p.conf_every = std::stoi(val());
		else if (o == "--save-events") p.save_events = true;
		else if (o == "--skip") p.skip = std::stod(val());
		else if (o == "--save-moves") p.save_moves = true;
		else if (o == "--persistence") p.persistence = std::stod(val());
		else if (o == "--origins") p.origins = std::stoi(val());
		else if (o == "--checkpoint") p.checkpoint = std::stod(val());
		else if (o == "--walltime") p.walltime = std::stod(val());
		else if (o == "--restart") p.restart = true;
		else if (o == "--out") p.out = val();
		else { usage(argv[0]); return 1; }
	}
	if (p.L < 3 || p.M < 0 || p.M == 1 || p.mcut < 1) { std::cerr << "need L >= 3, M = 0 or M >= 2, mcut >= 1\n"; return 1; }
	if (p.M == 0 && !p.philox) { std::cerr << "M = 0 (no period) needs --rng philox\n"; return 1; }
	if (p.M == 0 && !p.f_absolute) { std::cerr << "M = 0: F_c cannot be located, give F with --absolute\n"; return 1; }
	if (p.lmax >= p.L) { std::cerr << "need lmax < L\n"; return 1; }
	if (p.persistence > 0 && (p.T <= 0 || p.skip <= 0)) { std::cerr << "--persistence needs T > 0 and --skip\n"; return 1; }

	if (p.xi_x < 1 || p.xi_u < 1 || p.L % p.xi_x || (p.M > 0 && p.M % p.xi_u) || (!p.philox && (p.xi_x > 1 || p.xi_u > 1))) {
		std::cerr << "need xi_x, xi_u >= 1, xi_x dividing L and xi_u dividing M, and --rng philox if xi > 1\n";
		return 1;
	}
	Line line(p);
	std::ostringstream pt;
	pt << "L " << p.L << "\nM " << p.M << "\nmcut " << p.mcut << "\ndisorder " << p.disorder << "\nc " << p.c
	   << "\nf " << p.f << (p.f_absolute ? " (absolute)" : " (F/Fc)") << "\nseed " << p.seed
	   << "\nrng " << (p.philox ? "philox" : "mt") << "\nxi_x " << p.xi_x << "\nxi_u " << p.xi_u << "\nlmax " << line.lmax
	   << "\nT " << p.T << "\nU0 " << p.U0 << "\ntheta " << p.theta << "\nkmc_seed " << p.kmc_seed
	   << "\nevents " << p.events << "\nadvance " << p.advance << "\nskip " << p.skip << "\npersistence " << p.persistence << "\norigins " << p.origins
	   << "\nconf_every " << p.conf_every << "\nsave_events " << p.save_events << "\nsave_moves " << p.save_moves << "\n";
	const std::string ptxt = pt.str();
	namespace fs = std::filesystem;
	const std::string ckname = p.out + "/checkpoint.bin";
	const bool resume = p.restart && fs::exists(ckname);
	if (p.restart && !resume && fs::exists(p.out + "/summary.txt")) { std::cerr << p.out << ": already done\n"; return 0; }

	// state of the run, kept in the checkpoint with the line, the Monte Carlo generator, the
	// persistence and the sizes of the output files
	struct State {
		long s = 0, s_skip = 0, extended = 0;
		double Fc = 0, F = 0, t = 0, t_half = -1, u_half = 0, t_skip = -1, u_skip = 0, u_start = 0, u_now = 0;
	} st;
	const std::vector<std::string> outs = {"/events.dat", "/configs.bin", "/events.bin", "/moves.bin"};
	Persistence pers(p.L);
	if (resume) {
		std::ifstream in(ckname, std::ios::binary);
		std::string magic, ptxt0, kmc;
		get(in, magic); get(in, ptxt0);
		if (magic != "mr_creep checkpoint 1" || ptxt0 != ptxt) { std::cerr << ckname << ": not a checkpoint of this run\n"; return 1; }
		std::vector<long long> size;
		get(in, st); get(in, line.u); get(in, kmc); get(in, size);
		pers.load(in);
		if (!in) { std::cerr << ckname << ": truncated\n"; return 1; }
		std::istringstream(kmc) >> line.kmc;
		line.F = st.F;
		line.extended = st.extended;
		// output written after the checkpoint is discarded
		for (size_t k = 0; k < outs.size(); k++)
			if (size[k] >= 0) fs::resize_file(p.out + outs[k], size[k]);
		std::cerr << p.out << ": resumed at step " << st.s << "\n";
	} else {
		// F_c costs an advance of 2M with full scans of the blocks: skipped with --absolute
		std::vector<int> crit;
		if (!p.f_absolute) { crit.resize(p.L); st.Fc = line.critical_force(p.f, crit); }
		line.F = st.F = p.f_absolute ? p.f : p.f * st.Fc;
		std::fill(line.u.begin(), line.u.end(), 0);
		line.touch_all();
		line.relax_forward();
		std::ofstream log(p.out + "/params.txt");
		log << ptxt << "Fc " << (p.f_absolute ? "not computed" : std::to_string(st.Fc)) << "\nF " << line.F << std::endl;
		if (!p.f_absolute) {
			std::ofstream o(p.out + "/critical.dat");
			for (int v : crit) o << v << "\n";
		}
		for (int v : line.u) st.u_start += (double)v / p.L;
		st.u_now = st.u_start;
	}
	const auto mode = resume ? std::ios::app : std::ios::trunc;
	std::ofstream ev(p.out + "/events.dat", mode);
	// dt: time of the step at T > 0, in units of exp(U(2)/T)/nu0 (0 at T = 0)
	if (!resume) ev << "# step l gain L_act L_rel area mean_u E dt i x_left extent\n";
	ev.precision(10);
	std::ofstream conf, evconf, mvf;
	if (p.conf_every > 0) conf.open(p.out + "/configs.bin", std::ios::binary | mode);
	if (p.save_events) evconf.open(p.out + "/events.bin", std::ios::binary | mode);
	if (p.save_moves) mvf.open(p.out + "/moves.bin", std::ios::binary | mode);
	std::vector<std::ofstream *> outf = {&ev, &conf, &evconf, &mvf};

	const bool ckpt = p.checkpoint > 0 || p.walltime > 0;
	if (ckpt) for (int sig : {SIGTERM, SIGINT, SIGUSR1}) std::signal(sig, on_signal);
	const auto wall0 = std::chrono::steady_clock::now();
	double next_ckpt = p.checkpoint;
	auto write_checkpoint = [&]() {
		std::vector<long long> size;
		for (auto *f : outf) { f->flush(); size.push_back(f->is_open() ? (long long)f->tellp() : -1); }
		std::ostringstream kmc;
		kmc << line.kmc;
		{
			std::ofstream o(ckname + ".tmp", std::ios::binary);
			put(o, std::string("mr_creep checkpoint 1")); put(o, ptxt);
			put(o, st); put(o, line.u); put(o, kmc.str()); put(o, size);
			pers.save(o);
		}
		fs::rename(ckname + ".tmp", ckname);
	};

	std::vector<int> alpha, beta(p.L), moved;
	// time in units of 1/nu0: dt exp(U(2)/T)
	const double tscale = p.T > 0 ? std::exp(p.U0 * std::pow(2.0, p.theta) / p.T) : 0.0;
	if (!std::isfinite(tscale)) { std::cerr << "exp(U(2)/T) overflows: raise T\n"; return 1; }
	bool done = p.advance > 0 && st.u_now - st.u_start >= p.advance;
	while (!done && st.s < p.events) {
		const long s = ++st.s;
		alpha = line.u;
		Line::Window w = line.creep_step(beta);
		if (!w.l) { std::cerr << "step " << s << ": no window lowers the energy\n"; return 2; }
		// moved sites, and the smallest arc [x_left, x_left + extent) that contains them: the moved
		// set need not be contiguous, since the optimal path can touch the line inside the window
		moved.clear();
		for (int i = 0; i < p.L; i++) if (alpha[i] != line.u[i]) moved.push_back(i);
		int x_left = moved[0], gap = moved[0] + p.L - moved.back();
		for (size_t k = 1; k < moved.size(); k++)
			if (moved[k] - moved[k - 1] > gap) { gap = moved[k] - moved[k - 1]; x_left = moved[k]; }
		int extent = p.L - gap + 1;
		int Lact = 0, Lrel = 0;
		long area = 0, sum = 0;
		for (int i = 0; i < p.L; i++) {
			Lact += alpha[i] != beta[i];
			Lrel += alpha[i] != line.u[i];
			area += line.u[i] - alpha[i];
			sum += line.u[i];
		}
		ev << s << " " << w.l << " " << w.gain << " " << Lact << " " << Lrel << " " << area << " "
		   << (double)sum / p.L << " " << line.energy(line.u) << " " << (p.T > 0 ? line.dt : 0.0) << " "
		   << line.last_i << " " << x_left << " " << extent << "\n";
		if (p.save_events)
			for (const auto *x : {&alpha, &beta, &line.u}) evconf.write((const char *)x->data(), sizeof(int) * p.L);
		if (p.conf_every > 0 && s % p.conf_every == 0) conf.write((const char *)line.u.data(), sizeof(int) * p.L);
		// time, steady state and persistence
		st.extended = line.extended;
		st.t += line.dt * tscale;
		st.u_now = (double)sum / p.L;
		if (st.t_half < 0 && st.u_now - st.u_start >= p.skip / 2) { st.t_half = st.t; st.u_half = st.u_now; }
		if (st.t_skip < 0 && st.u_now - st.u_start >= p.skip) {
			st.t_skip = st.t; st.u_skip = st.u_now; st.s_skip = s;
			if (p.persistence > 0) {
				double v = (st.u_skip - st.u_half) / (st.t_skip - st.t_half);  // velocity over the second half of the transient
				pers.start(st.t, 1.0 / v, p.persistence, p.origins);
			}
		}
		if (pers.on()) pers.event(st.t, moved);
		if (p.save_moves) {  // record: double time (step at T = 0), int32 n, n x (int32 start, int32 length)
			std::vector<int32_t> runs;
			for (size_t k = 0; k < moved.size(); k++)
				if (k == 0 || moved[k] != moved[k - 1] + 1) { runs.push_back(moved[k]); runs.push_back(1); }
				else runs.back()++;
			double tt = p.T > 0 ? st.t : (double)s;
			int32_t n = (int32_t)runs.size() / 2;
			mvf.write((const char *)&tt, sizeof tt);
			mvf.write((const char *)&n, sizeof n);
			mvf.write((const char *)runs.data(), sizeof(int32_t) * runs.size());
		}
		done = p.advance > 0 && st.u_now - st.u_start >= p.advance;
		if (ckpt && !done && st.s < p.events) {
			double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
			bool stop = stop_signal || (p.walltime > 0 && wall >= p.walltime);
			if (stop || (p.checkpoint > 0 && wall >= next_ckpt)) {
				write_checkpoint();
				next_ckpt = wall + p.checkpoint;
				if (stop) { std::cerr << p.out << ": checkpoint at step " << s << ", stopped\n"; return 3; }
			}
		}
	}
	ev.flush();
	const long s_last = st.s, s_skip = st.s_skip;
	const double t = st.t, u_now = st.u_now, u_start = st.u_start, t_skip = st.t_skip, u_skip = st.u_skip;

	// steady-state summary; time in units of 1/nu0 (T > 0)
	std::ofstream sum(p.out + "/summary.txt");
	sum.precision(10);
	sum << "events " << s_last << "\nadvance " << u_now - u_start << "\nlmax_extended " << st.extended << "\n";
	if (t_skip >= 0) {
		sum << "steady_events " << s_last - s_skip << "\nsteady_advance " << u_now - u_skip << "\n";
		if (p.T > 0)
			sum << "steady_time " << t - t_skip << "\nv " << (u_now - u_skip) / (t - t_skip)
			    << "\nln_v " << std::log((u_now - u_skip) / (t - t_skip)) << "\n";
	}
	if (pers.on() && pers.n > 0) {
		std::ofstream po(p.out + "/persistence.dat");
		po.precision(10);
		po << "# tau pi pi2 chi4 (" << pers.n << " origins; tau in units of 1/nu0)\n";
		double tau_a = -1, cmax = -1, tau_c = 0;
		for (size_t b = 0; b < pers.tau.size(); b++) {
			double m = pers.S1[b] / pers.n;
			po << pers.tau[b] << " " << m << " " << pers.S2[b] / pers.n << " " << pers.chi4(b) << "\n";
			if (tau_a < 0 && m <= 0.5 && b > 0) {  // log-linear interpolation of pi = 1/2
				double m0 = pers.S1[b - 1] / pers.n;
				tau_a = pers.tau[b - 1] * std::pow(pers.tau[b] / pers.tau[b - 1], (m0 - 0.5) / (m0 - m));
			}
			if (pers.chi4(b) > cmax) { cmax = pers.chi4(b); tau_c = pers.tau[b]; }
		}
		sum << "persistence_origins " << pers.n << "\ntau_alpha " << tau_a << "\nchi4_max " << cmax
		    << "\ntau_chi4_max " << tau_c << "\n";
		std::ofstream so(p.out + "/s4.dat");
		so.precision(8);
		so << "# four-point structure factor S4(q, tau), " << pers.n << " origins: one row per tau, columns tau S4(q_1) S4(q_2) ...\n# q";
		for (int m : pers.modes) so << " " << 2 * M_PI * m / p.L;
		so << "\n";
		for (size_t b = 0; b < pers.tau.size(); b++) {
			so << pers.tau[b];
			for (size_t m = 0; m < pers.modes.size(); m++) so << " " << pers.s4(b, m);
			so << "\n";
		}
	}
	if (ckpt) fs::remove(ckname);
	return 0;
}
