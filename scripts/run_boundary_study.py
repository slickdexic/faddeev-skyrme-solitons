"""Paired periodic/Dirichlet comparison for the charge-one Hopfion.

The reference computation truncates to the vacuum on a fixed boundary; we use a
periodic box.  This script settles whether that difference of convention, rather
than a genuine discrepancy, accounts for the excess in E_1.  Both conventions
are relaxed from the same seed on the same grids with the same schedule, so the
only thing that differs between a pair is the boundary.

Outputs
-------
results/boundary_study.json : paired energies and the joint (h, L) extrapolation
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import ansatz, fs_core as fs, relax  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

BOUND = 32 * np.pi**2 * np.sqrt(2)

# (L, N) pairs spanning three spacings at three box sizes, so that the h^2 and
# the L^-p dependence can be separated rather than traded against each other.
GRIDS = [(10.0, 68), (10.0, 80), (10.0, 100),
         (12.0, 80), (12.0, 96), (12.0, 120),
         (14.0, 94), (14.0, 112), (14.0, 128)]

QUICK = [(10.0, 68), (12.0, 80), (14.0, 94)]


def clamp_vacuum(seed, width):
    """Impose the Dirichlet datum exactly: the collar is the vacuum, not the seed."""
    xp = fs.xp
    vac = fs.to_backend(fs.VACUUM).reshape(3, 1, 1, 1)
    mask = relax.frozen_mask(seed.shape, width)
    seed = seed.copy()
    seed[:, mask] = xp.broadcast_to(vac, seed.shape)[:, mask]
    return fs.normalise(seed)


def run_one(bc, L, N, steps, width=4):
    fs.set_boundary(bc)
    grid, h = fs.make_grid(N, L)
    seed = ansatz.hedgehog_hopf(grid, degree=1, R=4.5)
    if bc == "fixed":
        seed = clamp_vacuum(seed, width)

    t0 = time.time()
    n, _ = relax.minimise(seed, h, steps=steps, report=steps, max_rot=0.012,
                          momentum=0.9, rescale_every=120, verbose=False,
                          bc_width=width)
    e2, e4 = fs.energy(n, h)
    rec = dict(bc=bc, L=L, N=N, h=h, E=e2 + e4, E_over_bound=(e2 + e4) / BOUND,
               virial=e2 / e4, Q=fs.hopf_charge(n, h),
               leak=fs.boundary_leakage(n, h), secs=time.time() - t0)
    print(f"  {bc:8s} L={L:5.1f} N={N:4d} h={h:.4f}  "
          f"E/c0={rec['E_over_bound']:.4f}  virial={rec['virial']:.4f}  "
          f"Q={rec['Q']:+.4f}  leak={rec['leak']:.2e}  [{rec['secs']:.0f}s]")
    return rec


def joint_fit(rows):
    """E = E_inf - a h^2 + c L^-p, the same form used for the periodic series."""
    h = np.array([r["h"] for r in rows])
    L = np.array([r["L"] for r in rows])
    y = np.array([r["E_over_bound"] for r in rows])

    def model(X, Einf, a, c, p):
        hh, LL = X
        return Einf - a * hh**2 + c * LL**(-p)

    popt, pcov = curve_fit(model, (h, L), y, p0=[1.23, 1.0, 1.0, 3.0], maxfev=40000)
    resid = y - model((h, L), *popt)
    return dict(E_inf=popt[0], a=popt[1], c=popt[2], p=popt[3],
                sigma_E_inf=float(np.sqrt(np.diag(pcov))[0]),
                rms_resid=float(np.sqrt((resid**2).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=1400)
    args = ap.parse_args()

    if args.gpu:
        fs.use_gpu(True)
        print("backend: CuPy")

    grids = QUICK if args.quick else GRIDS
    out = {"bound_constant": BOUND, "steps": args.steps, "runs": []}

    t0 = time.time()
    for L, N in grids:
        for bc in ("periodic", "fixed"):
            out["runs"].append(run_one(bc, L, N, args.steps))
        a, b = out["runs"][-2], out["runs"][-1]
        d = b["E_over_bound"] - a["E_over_bound"]
        print(f"    -> fixed - periodic = {d:+.5f}  ({100*d/a['E_over_bound']:+.3f}%)\n")

    if not args.quick:
        for bc in ("periodic", "fixed"):
            rows = [r for r in out["runs"] if r["bc"] == bc]
            out[f"fit_{bc}"] = joint_fit(rows)
            f = out[f"fit_{bc}"]
            print(f"{bc:8s} continuum, infinite volume: E1/c0 = {f['E_inf']:.4f} "
                  f"+- {f['sigma_E_inf']:.4f}   (p={f['p']:.2f}, rms={f['rms_resid']:.1e})")
        gap = out["fit_fixed"]["E_inf"] - out["fit_periodic"]["E_inf"]
        out["boundary_shift"] = gap
        print(f"\nboundary convention moves the continuum limit by {gap:+.4f} "
              f"({100*gap/out['fit_periodic']['E_inf']:+.2f}%)")
        print(f"the discrepancy against the published 1.204 is "
              f"{100*(out['fit_fixed']['E_inf']-1.204)/1.204:+.2f}% under Dirichlet walls")

    (RES / "boundary_study.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/boundary_study.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
