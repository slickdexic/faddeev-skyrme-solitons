# Topological solitons of a hyperelastic vacuum

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21940804.svg)](https://doi.org/10.5281/zenodo.21940804)

Computational supplement to two companion papers, which share one codebase:

- **`papers/lattice/`** — *Faddeev–Skyrme soliton energies from controlled lattice
  extrapolation.* The numerics: the Nyquist null mode, two symmetrised schemes,
  the joint spacing/volume extrapolation, validation against an exactly solvable
  sector, and a 2.2% disagreement with the published charge-one energy.
- **`papers/framework/`** — *Topological solitons of a hyperelastic vacuum.* The
  interpretation: the exact energy identity and its corollary, the anchoring
  theorem, spin-statistics, selection rules, a quantitative no-go on generations,
  and a forward-scattering constraint.

Everything quoted in either paper is produced by the code here, and
`scripts/verify_claims.py` re-derives each headline number from the stored fields
rather than from intermediate summaries.

## Layout

| path | contents |
| --- | --- |
| `papers/lattice/` | REVTeX source, bibliography, `build.sh` |
| `papers/framework/` | REVTeX source, bibliography, `build.sh` |
| `rt/` | library |
| `scripts/` | drivers, one per result |
| `results/` | JSON summaries (field snapshots are regenerated, not stored) |
| `figures/` | figures as they appear in the papers |

Library modules: `constants.py` (CODATA-2018, the metric tension $\Gamma$);
`invariants.py` (symbolic proof of Theorem 1 and the virial identity);
`fs_core.py` (lattice energy, exact gradient, Hopf charge, array backend);
`ansatz.py` (seeds of prescribed charge); `relax.py` (minimiser, Derrick
rescaling, Lorentzian evolution); `skyrme.py` (frame sector: baryon number,
rational-map seeds, exact profile equation); `calibration.py` (Theorem 2);
`eikonal.py` (saturated eikonal, forward data, chi-square).

## Reproducing

```bash
pip install -r requirements.txt

python scripts/validate.py            # gradient, charge estimator, stability
python scripts/run_static.py          # soliton spectrum, Q = 1..4
python scripts/run_volume_study.py    # finite-volume systematic
python scripts/run_hopfion_resolution.py   # lattice-spacing series at fixed box
python scripts/run_order4_check.py    # same soliton, O(h^4) scheme
python scripts/run_dynamics.py        # annihilation and fusion
python scripts/run_skyrme.py          # frame sector, convergence and box study
python scripts/run_skyrme_spectrum.py # frame sector, B = 1..4
python scripts/run_breathing_mode.py  # width of the internal oscillation
python scripts/run_eikonal.py         # forward scattering fits and predictions
python scripts/make_figures.py        # figures
python scripts/verify_claims.py       # re-derives every quoted number
python scripts/verify_manuscript.py   # checks the LaTeX against those numbers
```

The static and dynamic runs write the field snapshots that `verify_claims.py`
reads, so run them before it. Each paper directory has its own `build.sh`.

## GPU

The lattice kernels are written against a swappable array backend and run
unchanged on CPU or, with `--gpu`, on a CUDA device. Double precision is used
throughout: the kernels are limited by memory bandwidth rather than
floating-point throughput, so a consumer card's reduced FP64 rate costs nothing.
On an RTX 2070 the GPU path is 6-8x faster and agrees with the CPU path to
rounding for every energy, charge and radius reported.

The one exception is the Derrick rescaling step, where the host and device
cubic-spline interpolators differ at the 1e-7 level. Because that step is
accepted only when it lowers the energy, this perturbs the relaxation path but
not the minimum reached. **All numbers quoted in the paper were produced on the
CPU path.**

## Notes on the discretisation

Two points that are easy to get wrong and are worth stating plainly.

*Central differences must not be used in the energy.* Their symbol vanishes at
the Nyquist frequency, so a checkerboard mode costs nothing, topological charge
unwinds for free, and the energy falls below the Vakulenko-Kapitanski bound. The
energy here is the forward/backward symmetrisation, whose symbol vanishes only at
`k = 0`. Because `(D+)^T = -D-` exactly on a periodic lattice, the analytic
variation remains the exact gradient of the discrete energy.

*Two schemes are available.* `fs_core.set_scheme("o2")` (default) builds the
symmetrisation from first-order one-sided differences and is accurate to
`O(h^2)`; `"o4"` uses third-order one-sided differences, whose leading errors are
equal and opposite, giving `O(h^4)`. Both are free of the null mode and both
preserve the exact-gradient property. Their errors have opposite sign, so
together they bracket the continuum limit.

## Licence

The software (`rt/`, `scripts/`, build and configuration files) is MIT
licensed, so that anyone can re-run these computations and check the results
without restriction.

The manuscripts in `papers/` and the figures in `figures/` are not covered by
that licence. They are included so the claims can be read alongside the code
that produces them; their reuse will be governed by the terms of publication.
See `LICENSE`.
