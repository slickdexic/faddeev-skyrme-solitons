"""Finite-volume systematic for the Q = 1 Hopfion.

The Faddeev-Skyrme director is massless, so the soliton has power-law tails and
its energy on a periodic lattice retains a finite-volume contribution from its
own images. This study holds the lattice spacing fixed at h = 1/6 and varies the
box, isolating the volume dependence from the discretisation error measured in
run_static.py.

Output: results/volume_study.json
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import ansatz, fs_core as fs, relax  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

BOUND = 32 * np.pi**2 * np.sqrt(2)
CASES = [(72, 12.0), (96, 16.0), (120, 20.0)]      # all h = 1/6


def main():
    rows = []
    t0 = time.time()
    for N, L in CASES:
        grid, h = fs.make_grid(N, L)
        seed = ansatz.hedgehog_hopf(grid, degree=1, R=4.5)
        n, _ = relax.minimise(seed, h, steps=1100, report=550, max_rot=0.012,
                              momentum=0.9, rescale_every=120, verbose=False)
        e2, e4 = fs.energy(n, h)
        R, _c = fs.energy_radius(n, h, grid)
        row = dict(N=N, L=L, h=h, E=e2 + e4, E_over_c0=(e2 + e4) / BOUND,
                   virial=e2 / e4, Q=fs.hopf_charge(n, h), R_rms=R,
                   leak=fs.boundary_leakage(n, h))
        rows.append(row)
        print(f"  N={N:4d} L={L:5.1f} h={h:.4f}  E={row['E']:9.3f}  "
              f"E/c0={row['E_over_c0']:.4f}  virial={row['virial']:.4f}  "
              f"Q={row['Q']:+.4f}  R={R:.3f}  leak={row['leak']:.2e}  "
              f"[{time.time()-t0:.0f}s]")

    # Richardson-style extrapolation in 1/L assuming a power-law tail contribution
    if len(rows) >= 2:
        invL = np.array([1.0 / r["L"] for r in rows])
        E = np.array([r["E"] for r in rows])
        a, b = np.polyfit(invL, E, 1)
        print(f"\n  linear extrapolation in 1/L:  E(L -> inf) = {b:.3f}  "
              f"(E/c0 = {b/BOUND:.4f})")
        payload = dict(rows=rows, E_infinite_volume=float(b),
                       E_over_c0_infinite_volume=float(b / BOUND))
    else:
        payload = dict(rows=rows)

    (RES / "volume_study.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwritten to {RES/'volume_study.json'}")


if __name__ == "__main__":
    main()
