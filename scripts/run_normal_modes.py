"""Normal-mode spectrum of a relaxed soliton.

The one route to a generation-like structure that topology does not forbid is a
discrete tower of *vibrational* excitations: states with the same charge and the
same spin but higher mass. This computes the low end of the Hessian spectrum to
see whether such a tower exists.

The director is massless -- the vacuum manifold is S^2, so fluctuations about the
vacuum are Goldstone modes -- which means the radiation continuum starts at
omega = 0 and there is no gap for a bound state to sit in. The test is therefore
not "is the spectrum discrete" (on a finite periodic box everything is discrete)
but "is any low mode *localised* on the soliton, with a frequency independent of
the box". Radiation gives delocalised modes with omega ~ 2 pi / L.

Usage:  python scripts/run_normal_modes.py [--gpu] [--L 12] [--N 64] [--k 24]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, ansatz, relax  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def hessian_operator(n, h, eps=1e-5):
    """Second variation on the constraint surface, as a matrix-free operator.

    Renormalising the displaced field before differencing makes the curvature of
    the target sphere part of the operator automatically.
    """
    shape = n.shape
    size = int(np.prod(shape))

    def project(v):
        return v - fs.xp.einsum("axyz,axyz->xyz", v, n) * n

    def matvec(vflat):
        v = fs.to_backend(np.asarray(vflat, dtype=float).reshape(shape))
        v = project(v)
        gp = fs.variation(fs.normalise(n + eps * v), h)
        gm = fs.variation(fs.normalise(n - eps * v), h)
        return fs.to_numpy(project((gp - gm) / (2.0 * eps))).ravel()

    return LinearOperator((size, size), matvec=matvec, dtype=float)


def localisation(vec, grid, r0=3.0):
    """Fraction of the mode's weight inside r < r0, against the volume fraction."""
    X, Y, Z = (fs.to_numpy(g) for g in grid)
    r2 = X**2 + Y**2 + Z**2
    w = (vec**2).sum(axis=0)
    inside = r2 < r0**2
    return float(w[inside].sum() / w.sum()), float(inside.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()

    if args.gpu:
        fs.use_gpu()
    t0 = time.time()
    grid, h = fs.make_grid(args.N, args.L)
    n = ansatz.hedgehog_hopf(grid, 1, R=args.L / 4)
    n, _ = relax.minimise(n, h, steps=args.steps, max_rot=0.010, momentum=0.92,
                          rescale_every=100, track_charge=False, verbose=False)
    e2, e4 = fs.energy(n, h)
    Q = fs.hopf_charge(n, h)
    print(f"  L={args.L} N={args.N} h={h:.4f}: E={e2+e4:.3f}  "
          f"Q={Q:+.4f}  virial={e2/e4:.4f}  [{time.time()-t0:.0f}s]",
          flush=True)
    if abs(Q) < 0.5:
        sys.exit("  the soliton unwound on this lattice; use a finer spacing")

    H = hessian_operator(n, h)
    vals, vecs = eigsh(H, k=args.k, which="SA", tol=1e-6,
                       ncv=min(3 * args.k, H.shape[0] - 1))
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    k_box = 2.0 * np.pi / args.L                  # lowest radiation wavenumber
    print(f"\n  lowest radiation mode expected at omega^2 ~ {k_box**2:.4f} "
          f"(2 pi / L)^2\n")
    print(f"  {'i':>3} {'omega^2':>12} {'omega':>9} {'loc(r<3)':>9} "
          f"{'vol.frac':>9}  verdict")
    rows = []
    for i, lam in enumerate(vals):
        loc, vf = localisation(vecs[:, i].reshape(n.shape), grid)
        kind = ("zero mode" if abs(lam) < 0.02 * k_box**2
                else "LOCALISED" if loc > 6 * vf else "radiation")
        rows.append(dict(i=i, omega2=float(lam), loc=loc, vol_frac=vf, kind=kind))
        print(f"  {i:3d} {lam:12.5f} {np.sign(lam)*np.sqrt(abs(lam)):9.4f} "
              f"{loc:9.3f} {vf:9.3f}  {kind}")

    out = dict(L=args.L, N=args.N, h=h, E=float(e2 + e4),
               k_box2=float(k_box**2), modes=rows)
    (RES / f"normal_modes_L{args.L:.0f}_N{args.N}.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
