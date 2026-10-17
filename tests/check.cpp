// Internal checks of mr_creep (tests/run_tests.sh):
//  1. the window scan (transfer matrix) against brute-force enumeration of the forward paths;
//  2. the fast relaxation (prefix sums, skipping clean blocks) against the plain scan of
//     krgk_creep over all blocks, from the same activated configurations;
//  3. the cached windows against recomputing every window at every step;
//  4. the O(L) lowest block-move energy against the minimum over all blocks;
//  5. at T > 0, the list of all improving windows against brute force, and the cached windows
//     against recomputing every window at every step (same kinetic Monte Carlo numbers);
//  6. the Philox disorder: known-answer test of Philox4x32-10, mean and variance of V, and the
//     cached band against direct generation.
#define main mr_main
#include "../src/mr_creep.cpp"
#undef main

// plain relaxation of krgk_creep: first favourable block in the order (il, i), all blocks scanned
static void relax_plain(Line &l)
{
	for (;;) {
		bool moved = false;
		for (int il = 2; il <= l.L + 1 && !moved; il++) {
			int nstart = (il <= l.L) ? l.L : 1;
			for (int i = 0; i < nstart && !moved; i++) {
				std::vector<int> x = l.u;
				int n = (il <= l.L) ? il - 1 : l.L;
				for (int j = 1; j <= n; j++) x[(i + j) % l.L] += 1;
				bool ok = true;
				for (int s = 0; s < l.L; s++) ok = ok && std::abs(x[(s + 1) % l.L] - x[s]) <= l.mcut;
				if (ok && l.energy(x) - l.energy(l.u) < 0) { l.u = x; moved = true; }
			}
		}
		if (!moved) return;
	}
}

// brute force: lowest energy of forward paths in window [i, i+w] with fixed end points
static double brute(const Line &l, int i, int w, std::vector<int> &x, int j, double best)
{
	if (j == w) {
		int e = (i + w) % l.L, p = (i + w - 1) % l.L;
		if (std::abs(l.u[e] - x[p]) > l.mcut) return best;
		double E = 0;
		for (int k = 1; k < w; k++) E += l.site((i + k) % l.L, x[(i + k) % l.L]);
		for (int k = 0; k < w; k++) { int a = (i + k) % l.L, b = (i + k + 1) % l.L; int d = (k + 1 == w ? l.u[b] : x[b]) - (k == 0 ? l.u[a] : x[a]); E += l.c * 0.5 * d * d; }
		return std::min(best, E);
	}
	int s = (i + j) % l.L, prev = (j == 1) ? l.u[i] : x[(i + j - 1) % l.L];
	for (int d = -l.mcut; d <= l.mcut; d++) {
		int v = prev + d;
		if (v < l.u[s] || std::abs(v - l.u[(i + w) % l.L]) > (w - j) * l.mcut) continue;
		x[s] = v;
		best = brute(l, i, w, x, j + 1, best);
	}
	return best;
}

int main()
{
	int bad = 0, found = 0, steps = 0;
	// 1. window scan against brute force, on metastable states of short runs
	for (unsigned long seed = 1; seed <= 6; seed++) {
		Params p; p.L = 24; p.M = 64; p.mcut = 1 + seed % 2; p.seed = seed; p.lmax = 9; p.f = 0.5;
		Line l(p);
		std::vector<int> crit(p.L), beta(p.L);
		double Fc = l.critical_force(p.f, crit);
		l.F = p.f * Fc;
		std::fill(l.u.begin(), l.u.end(), 0);
		l.touch_all(); l.relax_forward();
		for (int step = 0; step < 15; step++) {
			for (int i = 0; i < p.L; i++) {
				Line::Window w = l.scan(i);
				int lb = 0; double gb = 0;
				for (int ww = 2; ww <= p.lmax && !lb; ww++) {
					double cur = 0;
					for (int k = 1; k < ww; k++) cur += l.site((i + k) % p.L, l.u[(i + k) % p.L]);
					for (int k = 0; k < ww; k++) { int d = l.u[(i + k + 1) % p.L] - l.u[(i + k) % p.L]; cur += l.c * 0.5 * d * d; }
					std::vector<int> x = l.u;
					double b = brute(l, i, ww, x, 1, BIG);
					if (b < cur - EPS) { lb = ww; gb = cur - b; }
				}
				found += w.l > 0;
				if (w.l != lb || std::abs(w.gain - gb) > 1e-9) { bad++; printf("scan seed %lu i %d: l %d gain %g, brute l %d gain %g\n", seed, i, w.l, w.gain, lb, gb); }
			}
			if (!l.creep_step(beta).l) break;
			steps++;
		}
	}
	printf("1. window scan vs brute force: %d mismatches (%d creep steps, %d improving windows)\n", bad, steps, found);

	// 2. fast relaxation against the plain one, and 3. cached windows against recomputing all
	int bad2 = 0, bad3 = 0, bad4 = 0;
	for (unsigned long seed = 1; seed <= 4; seed++) {
		Params p; p.L = 48; p.M = 128; p.seed = seed; p.f = 0.4 + 0.1 * seed;
		Line a(p), b(p);
		std::vector<int> crit(p.L), ba(p.L), bb(p.L);
		double Fc = a.critical_force(p.f, crit);
		b.critical_force(p.f, crit);
		a.F = b.F = p.f * Fc;
		std::fill(a.u.begin(), a.u.end(), 0);
		a.touch_all(); a.relax_forward();
		b.u = std::vector<int>(p.L, 0);
		relax_plain(b);
		if (a.u != b.u) bad2++;
		b.u = a.u; b.touch_all(); b.relax_forward();
		for (int step = 0; step < 200; step++) {
			std::fill(b.valid.begin(), b.valid.end(), 0);  // b recomputes every window
			Line::Window wa = a.creep_step(ba), wb = b.creep_step(bb);
			if (wa.l != wb.l || a.u != b.u) { bad3++; break; }
			for (int dir : {+1, -1}) {  // 4. lowest block move, on gamma and on the activated state
				for (const auto *x : {&a.u, &ba}) {
					Line t = a; t.u = *x; t.prefix(dir);
					double lo = t.P[t.L];
					for (int il = 2; il <= t.L; il++)
						for (int i = 0; i < t.L; i++) {
							std::vector<int> y = t.u;
							for (int j = 1; j < il; j++) y[(i + j) % t.L] += dir;
							bool ok = true;
							for (int k = 0; k < t.L; k++) ok = ok && std::abs(y[(k + 1) % t.L] - y[k]) <= t.mcut;
							if (ok) lo = std::min(lo, t.energy(y) - t.energy(t.u));
						}
					if (std::abs(lo - t.lowest_move(dir)) > 1e-8) bad4++;
				}
			}
			Line c = a;  // relax the activated configuration with the plain scan
			c.u = ba; relax_plain(c);
			if (c.u != a.u) { bad2++; break; }
		}
	}
	printf("2. fast vs plain relaxation: %d mismatches\n", bad2);
	printf("3. cached vs recomputed windows: %d mismatches\n", bad3);
	printf("4. lowest block move, O(L) vs all blocks: %d mismatches\n", bad4);

	int bad5 = 0, cands = 0, steps5 = 0;
	for (unsigned long seed = 1; seed <= 4; seed++) {
		Params p; p.L = 24; p.M = 64; p.mcut = 1 + seed % 2; p.seed = seed; p.lmax = 9; p.f = 0.5;
		p.T = 0.3 * seed; p.kmc_seed = seed;
		Line a(p), b(p);
		std::vector<int> crit(p.L), ba(p.L), bb(p.L);
		double Fc = a.critical_force(p.f, crit);
		a.F = b.F = p.f * Fc;
		std::fill(a.u.begin(), a.u.end(), 0);
		a.touch_all(); a.relax_forward();
		b.u = a.u;
		for (int step = 0; step < 40; step++) {
			for (int i = 0; i < p.L; i++) {
				std::vector<Line::Window> all;
				a.scan(i, 0, nullptr, &all);
				std::vector<int> lb;
				for (int ww = 2; ww <= p.lmax; ww++) {
					double cur = 0;
					for (int k = 1; k < ww; k++) cur += a.site((i + k) % p.L, a.u[(i + k) % p.L]);
					for (int k = 0; k < ww; k++) { int d = a.u[(i + k + 1) % p.L] - a.u[(i + k) % p.L]; cur += a.c * 0.5 * d * d; }
					std::vector<int> x = a.u;
					if (brute(a, i, ww, x, 1, BIG) < cur - EPS) lb.push_back(ww);
				}
				cands += all.size();
				if (all.size() != lb.size()) bad5++;
				else for (size_t k = 0; k < lb.size(); k++) bad5 += all[k].l != lb[k];
			}
			std::fill(b.valid.begin(), b.valid.end(), 0);
			Line::Window wa = a.creep_step(ba), wb = b.creep_step(bb);
			if (!wa.l) break;
			steps5++;
			if (wa.l != wb.l || a.u != b.u || a.dt != b.dt) { bad5++; break; }
		}
	}
	printf("5. T > 0: candidates vs brute force and cached vs recomputed: %d mismatches (%d steps, %d candidates)\n", bad5, steps5, cands);

	int bad6 = 0;
	{
		r123::Philox4x32::ctr_type c = {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344}};
		r123::Philox4x32::key_type k = {{0xa4093822, 0x299f31d0}};
		auto r = r123::Philox4x32()(c, k);  // known answer of Random123 (kat_vectors)
		bad6 += !(r[0] == 0xd16cfe09 && r[1] == 0x94fdcceb && r[2] == 0x5001e420 && r[3] == 0x24126ea1);
		Params p; p.L = 1000; p.M = 0; p.disorder = 1.5;
		Line l(p);
		double m = 0, v = 0, n = 0;
		for (int i = 0; i < p.L; i++)
			for (int j = -500; j < 500; j++) { double x = l.pot(i, j); m += x; v += x * x; n++; }
		m /= n; v = v / n - m * m;
		if (std::abs(m) > 0.01 || std::abs(v / 2.25 - 1) > 0.01) { bad6++; printf("Philox V: mean %g variance %g (expected 0, 2.25)\n", m, v); }
		l.u.assign(p.L, 0);
		for (int i = 0; i < p.L; i++) l.u[i] = (i * 7919) % 300 - 150;
		l.rebase();
		for (int i = 0; i < p.L; i++)
			for (int j = l.u[i] - 10; j < l.u[i] + 400; j++) bad6 += l.pot(i, j) != l.pot_raw(i, j);
	}
	printf("6. Philox disorder (known answer, moments, cached band): %d mismatches\n", bad6);
	return bad + bad2 + bad3 + bad4 + bad5 + bad6 ? 1 : 0;
}
