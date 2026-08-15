"""Static Hopfion spectrum: minimise E[n] in the sectors Q_H = 1..4.

Outputs
-------
results/static_solitons.json : energies, virial ratios, radii, charges
results/field_Q{n}.npy       : relaxed fields (used by the dynamics and
                               form-factor scripts)
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

# Ward's conjectured bound constant. Ward normalises the energy by 1/(32 pi^2 sqrt 2)
# and conjectures E >= |Q|^{3/4} in those units; the integrand is identical to ours.
BOUND = 32 * np.pi**2 * np.sqrt(2)

N, L = 80, 12.0
STEPS = 1400


def relax_sector(Q, seeds, N=N, L=L, steps=STEPS):
    grid, h = fs.make_grid(N, L)
    best = None
    for label, seed in seeds(grid):
        print(f"  [Q={Q}] seed '{label}'  (N={N}, L={L}, h={h:.4f})")
        n, _ = relax.minimise(seed, h, steps=steps, report=200, max_rot=0.012,
                              momentum=0.9, rescale_every=120)
        e2, e4 = fs.energy(n, h)
        q = fs.hopf_charge(n, h)
        R, _c = fs.energy_radius(n, h, grid)
        rec = dict(seed=label, N=N, L=L, h=h, E2=e2, E4=e4, E=e2 + e4,
                   virial=e2 / e4, Q=q, R_rms=R,
                   leak=fs.boundary_leakage(n, h), E_over_bound=(e2 + e4) / BOUND)
        print(f"    -> E={rec['E']:.3f}  E/bound={rec['E_over_bound']:.4f}  "
              f"Q={q:+.4f}  virial={rec['virial']:.4f}  R={R:.3f}")
        if best is None or rec["E"] < best[0]["E"]:
            best = (rec, n)
    return best


def seeds_for(Q):
    def f(grid):
        out = [(f"hedgehog d={Q}", ansatz.hedgehog_hopf(grid, degree=Q, R=4.0 + 0.5 * Q))]
        if Q >= 2:
            out.append((f"A_1,{Q}", ansatz.torus_anm(grid, 1, Q, a=2.2, eta0=1.7)))
            out.append((f"A_{Q},1", ansatz.torus_anm(grid, Q, 1, a=2.2, eta0=1.7)))
        if Q == 4:
            out.append(("A_2,2", ansatz.torus_anm(grid, 2, 2, a=2.4, eta0=1.8)))
        return out
    return f


def main():
    t0 = time.time()
    results = {"convention": "E = Int [ I1 + I2 ] d3x  (c2 = c4 = 1)",
               "bound_constant": BOUND, "sectors": {}, "resolution_study": []}

    for Q in (1, 2, 3, 4):
        rec, n = relax_sector(Q, seeds_for(Q))
        results["sectors"][str(Q)] = rec
        np.save(RES / f"field_Q{Q}.npy", n)
        print(f"  [Q={Q}] done, {time.time()-t0:.0f}s elapsed\n")

    print("== resolution study, Q = 1 ==")
    for Nr in (56, 72, 88, 104):
        grid, h = fs.make_grid(Nr, L)
        seed = ansatz.hedgehog_hopf(grid, degree=1, R=4.5)
        n, _ = relax.minimise(seed, h, steps=900, report=450, max_rot=0.012,
                              momentum=0.9, rescale_every=120, verbose=False)
        e2, e4 = fs.energy(n, h)
        row = dict(N=Nr, h=h, E=e2 + e4, virial=e2 / e4, Q=fs.hopf_charge(n, h),
                   E_over_bound=(e2 + e4) / BOUND)
        results["resolution_study"].append(row)
        print(f"  N={Nr:4d} h={h:.4f}  E={row['E']:9.3f}  E/bound={row['E_over_bound']:.4f}  "
              f"Q={row['Q']:+.4f}  virial={row['virial']:.4f}  [{time.time()-t0:.0f}s]")

    (RES / "static_solitons.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {RES/'static_solitons.json'}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
