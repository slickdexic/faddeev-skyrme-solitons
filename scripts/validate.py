"""Validation suite: functional gradient, Hopf-charge estimator, lattice convergence."""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import ansatz, fs_core as fs, relax  # noqa: E402

print("=" * 72)
print("1. Analytic functional gradient vs finite differences")
print("=" * 72)
grid, h = fs.make_grid(32, 10.0)
seed = fs.normalise(ansatz.hedgehog_hopf(grid, degree=1, R=3.0)
                    + 0.05 * np.random.default_rng(1).normal(size=(3, 32, 32, 32)))
print(f"  max relative error = {fs.gradient_check(seed, h, npts=15):.3e}")

print()
print("=" * 72)
print("2. Hopf-charge estimator: lattice convergence (fixed box L = 12)")
print("=" * 72)
print(f"  {'N':>5} {'h':>8} | {'d=1 (1)':>10} {'d=2 (2)':>10} "
      f"{'A11 (1)':>10} {'A12 (2)':>10} {'A13 (3)':>10} {'A22 (4)':>10} {'curl res':>10}")
for N in (48, 64, 80, 96):
    grid, h = fs.make_grid(N, 12.0)
    fields = {
        "d1": ansatz.hedgehog_hopf(grid, degree=1, R=3.0),
        "d2": ansatz.hedgehog_hopf(grid, degree=2, R=3.0),
        "A11": ansatz.torus_anm(grid, 1, 1, a=2.0, eta0=1.6),
        "A12": ansatz.torus_anm(grid, 1, 2, a=2.0, eta0=1.6),
        "A13": ansatz.torus_anm(grid, 1, 3, a=2.0, eta0=1.6),
        "A22": ansatz.torus_anm(grid, 2, 2, a=2.0, eta0=1.6),
    }
    qs = {k: fs.hopf_charge(v, h) for k, v in fields.items()}
    res = fs.curl_residual(fields["d1"], h)
    print(f"  {N:5d} {h:8.4f} | " + " ".join(f"{qs[k]:10.4f}" for k in
          ("d1", "d2", "A11", "A12", "A13", "A22")) + f" {res:10.2e}")

print()
print("=" * 72)
print("3. Domain reflection reverses the Hopf charge")
print("=" * 72)
grid, h = fs.make_grid(80, 12.0)
f1 = ansatz.torus_anm(grid, 1, 2, a=2.0, eta0=1.6)
print(f"  Q(A12) = {fs.hopf_charge(f1, h):+.4f}   "
      f"Q(reflect A12) = {fs.hopf_charge(ansatz.conjugate(f1), h):+.4f}")

print()
print("=" * 72)
print("4. Short minimisation of the Q = 1 seed (stability check)")
print("=" * 72)
grid, h = fs.make_grid(64, 14.0)
n0 = ansatz.hedgehog_hopf(grid, degree=1, R=3.5)
n, hist = relax.minimise(n0, h, steps=600, report=150)
E2, E4 = hist[-1][1], hist[-1][2]
c0 = 32 * np.pi**2 * np.sqrt(2)
print(f"  E = {E2+E4:.3f}   E2/E4 = {E2/E4:.4f}   Q = {hist[-1][5]:+.4f}   "
      f"E/c0 = {(E2+E4)/c0:.4f}  (converged: 1.215 at N=80; published 1.204)")
