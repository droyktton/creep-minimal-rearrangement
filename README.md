# creep-minimal-rearrangement

Creep of a directed elastic line in a random potential, driven below the depinning threshold, by thermally activated rearrangements followed by deterministic relaxations:

- **T → 0⁺**: the algorithm of Ferrero, Foini, Giamarchi, Kolton and Rosso [1]. It approximates the exact transition pathways of Kolton, Rosso, Giamarchi and Krauth [2, 3] (code: [krgk-creep](https://github.com/droyktton/krgk-creep)): the next metastable state is reached through the *smallest* compact rearrangement that lowers the energy, instead of the one with the smallest barrier.
- **T > 0**: its extension by Russo, Ferrero, Kolton, Rosso and Vandembroucq [4]. A kinetic Monte Carlo picks among all the energy-lowering rearrangements, with rates set by an assumed barrier U(ℓ).

The T → 0⁺ algorithm is the limit of the finite-temperature one, and both share the same code.

## Build, test and run

```
make                     # build/mr_creep (C++17, OpenMP)
make test                # internal checks; KRGK=path/to/krgk_creep make test also compares F_c
build/mr_creep --L 1024 --M 4096 --f 0.3 --events 10000 --out run1           # T -> 0+
build/mr_creep --L 1024 --M 4096 --f 0.3 --T 0.5 --events 10000 --out run2   # T = 0.5
```

Threads are set with `OMP_NUM_THREADS`.

Options (`build/mr_creep --help`):

| option | meaning | default |
|---|---|---|
| `--L` | line length (sites, periodic) | 64 |
| `--M` | period of the disorder along u; 0: not periodic (needs `--absolute`) | 1024 |
| `--mcut` | maximal slope \|u_{i+1} − u_i\| | 1 |
| `--disorder` | strength of the Gaussian disorder (std) | 1.0 |
| `--c` | elastic constant | 2.0 |
| `--f` | drive F/F_c (F with `--absolute`) | 0.5 |
| `--absolute` | `--f` is the force F; F_c is then not computed | off |
| `--seed` | seed of the disorder | 12345 |
| `--rng` | disorder generated on demand (`philox`) or stored as a table (`mt`) | `philox` |
| `--xi-x`, `--xi-u` | Philox disorder constant on rectangles of xi-x sites by xi-u along u | 1, 1 |
| `--events` | number of creep steps | 100 |
| `--advance` | stop earlier, once the mean position has advanced by this much | 0 (off) |
| `--lmax` | largest rearrangement window | L − 1 at T = 0, 50 at T > 0 |
| `--T` | temperature; 0 is the T → 0⁺ limit | 0 |
| `--U0`, `--theta` | barrier of a rearrangement of window ℓ, U(ℓ) = U0·ℓ^θ (T > 0) | 1, 1/3 |
| `--kmc-seed` | seed of the kinetic Monte Carlo | 1 |
| `--conf-every` | write the metastable state every n steps to `configs.bin` (int32, L per record) | 0 (never) |
| `--save-events` | write α, the activated configuration and γ of every step to `events.bin` (int32, 3 records of L) | off |
| `--out` | output directory | `.` |

Outputs:
- `params.txt`: the parameters, F_c and F.
- `critical.dat`: the critical configuration, the one that blocks the line at the last raise of F when locating F_c. Written unless `--absolute`.
- `events.dat`: one line per step, with columns
  - `step`;
  - `l`: window of the activated rearrangement;
  - `gain`: energy it lowers;
  - `L_act`: sites it moves;
  - `L_rel`: sites moved from α to the new metastable state γ;
  - `area`: Σ(γ − α);
  - `mean_u`, and `E`, the energy of γ;
  - `dt`: duration of the step at T > 0, in units of e^{U(2)/T}/ν₀ (0 at T = 0).

## Model

Energy of the line u_i (integer, i = 0..L−1, periodic):

E = Σ_i [ V(i, u_i mod M) − F u_i + (c/2)(u_{i+1} − u_i)² ],  with |u_{i+1} − u_i| ≤ mcut.

The disorder is Gaussian and uncorrelated. It is periodic in u with period M, or not periodic with `--M 0`.

**Disorder on demand (default).** V(i, u) is computed when needed from the counter (i, u) and the seed, with the counter-based generator Philox4x32-10 of Random123 [6] (`third_party/Random123`, header-only, BSD license).
- The four 32-bit outputs make two 64-bit uniforms, and the Gaussian value comes from the Box–Muller formula.
- Nothing of size L·M is stored. Each site keeps a cache of V over a band of ~3·min(ℓ_max·mcut, 64) values in front of the line, refilled when the line leaves it.
- At L = 4096 the code then needs 11 MB instead of 137 MB, and it is slightly faster than with the table, because the band is local in memory.

**Correlated disorder (`--xi-x a --xi-u b`, Philox only).** V(i, u) = g(⌊i/a⌋, ⌊u/b⌋): V is constant on rectangles of a sites by b along u, with independent Gaussian amplitudes. This sets the correlation lengths, possibly anisotropic, independently of the lattice. a must divide L and b must divide M.

**Table (`--rng mt`).** The disorder of krgk-creep and of the original code: Mersenne Twister numbers stored in an L·M table, minus their mean. With the same seed, krgk-creep and mr_creep then simulate the same sample. The energy, the elementary moves and the computation of F_c are also those of krgk-creep.

## Algorithm

Each step goes from a metastable state α to the next one, γ.

**1. Candidate rearrangements.** For every window [i, i+ℓ]:
- the end points u_i and u_{i+ℓ} are kept fixed;
- the optimal forward path between them (u'_j ≥ u_j) is computed by transfer matrix;
- the window is a candidate if this path has a lower energy than the line.

A single sweep from site i gives the optimal path of every window ℓ starting at i, at cost O(ℓ²·mcut²) up to ℓ.

**2. Choice.**
- At T = 0 the rearrangement is the candidate of the smallest ℓ [1]. Among candidates of that ℓ we take the one that lowers the energy most; ref. [1] does not specify this choice.
- At T > 0 every candidate up to `lmax` has the rate exp(−U(ℓ)/T), with U(ℓ) = U0·ℓ^θ. One candidate is drawn with probability proportional to its rate, and the time advances by −ln(r)/R, where R is the total rate [4].
  - The default θ = 1/3 is the equilibrium droplet exponent used in [4].
  - Ref. [4] tested two candidate sets and found them indistinguishable: all the energy-lowering windows, or a hierarchical, non-overlapping subset. This code uses all of them.
  - As T → 0 the choice tends to the smallest ℓ, but candidates of equal ℓ are then drawn at random, not by the largest gain.

**3. Deterministic relaxation.** Forward elementary moves (rigid displacement by +1 of a block of consecutive sites, any length) are applied until none lowers the energy. The order is that of krgk-creep: the shortest favourable block, then the leftmost.

**Costs.**
- **Windows.** The windows are independent of each other, so they are computed in parallel (OpenMP). After a step, only the windows that contain a moved site are computed again.
- **Relaxation.** It uses prefix sums of the site energies, and skips the blocks that touch no moved site; those cannot have become favourable.
  - Whether any favourable block remains is decided in O(L). The energy change of a block is separable into a term for each border, so the minimum over all blocks is a sliding-window minimum.

The method gives the sequence of metastable states and the events, not the energy barriers. At T > 0 the time follows from the assumed U(ℓ).

## Tests

`make test` runs `tests/check.cpp`:
1. the windows against brute-force enumeration of the forward paths (T = 0: smallest improving window and its gain);
2. the relaxation against the plain scan of all blocks;
3. the cached windows against recomputing all windows at every step;
4. the O(L) test for favourable blocks against the minimum over all blocks;
5. at T > 0, all the candidates against brute force, and the cached against the recomputed windows with the same Monte Carlo numbers;
6. the Philox disorder: the known answer of Philox4x32-10, the mean and variance of V, the cached band against direct generation, and the rectangles of correlated disorder.

With `KRGK=path/to/krgk_creep` it also checks that F_c and the critical configuration are those of krgk-creep (both with `--rng mt`).

**Against the exact algorithm.** On the same three samples (`--rng mt`, L = 64, f = 0.6, steady state, about 16 500 events each), krgk-creep and mr_creep give the same structure factor and the same distributions of L_rel and area. Only the smallest events differ slightly, as in Fig. S2 of [1]. mr_creep takes 3 s, krgk-creep about 10 minutes.

## Performance

On one desktop core, 3000 steps at L = 4096 and F = 0.093 (f ≈ 0.1) take:
- 3.4 s at T = 0;
- 7.2 s at T = 0.5 with ℓ_max = 50.

Limits:
- **Relaxation.** It still costs O(L) per elementary move (prefix sums and the test for favourable blocks). At T = 0 it dominates, and threads do not help.
- **F_c.** Locating it needs an advance of 2M with full scans of the blocks (145 s at L = 1024, M = 4096). Give F with `--absolute` to skip it.
- **Memory.** Only with `--rng mt`, where the disorder is stored as L·M doubles (512 MB at L = 4096, M = 16384).

## References

1. E. E. Ferrero, L. Foini, T. Giamarchi, A. B. Kolton and A. Rosso, *Spatiotemporal patterns in ultraslow domain wall creep dynamics*, Phys. Rev. Lett. **118**, 147208 (2017). doi:10.1103/PhysRevLett.118.147208, arXiv:1604.03726
2. A. B. Kolton, A. Rosso, T. Giamarchi and W. Krauth, *Dynamics below the depinning threshold in disordered elastic systems*, Phys. Rev. Lett. **97**, 057001 (2006). doi:10.1103/PhysRevLett.97.057001
3. A. B. Kolton, A. Rosso, T. Giamarchi and W. Krauth, *Creep dynamics of elastic manifolds via exact transition pathways*, Phys. Rev. B **79**, 184207 (2009). doi:10.1103/PhysRevB.79.184207
4. G. Russo, E. E. Ferrero, A. B. Kolton, A. Rosso and D. Vandembroucq, *Activation and avalanche length scales in the finite-temperature creep of an elastic interface*, arXiv:2604.17600 (2026)
5. A. Rosso, A. K. Hartmann and W. Krauth, *Depinning of elastic manifolds*, Phys. Rev. E **67**, 021602 (2003) (the elementary moves of the relaxation)
6. J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, *Parallel random numbers: as easy as 1, 2, 3*, Proc. SC11 (2011). doi:10.1145/2063384.2063405

If you use this code, please cite [1] for T → 0⁺ and [4] for T > 0, as well as [3].
