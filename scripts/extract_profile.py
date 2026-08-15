"""Extract the impact-parameter profile of a relaxed soliton.

The eikonal opacity of an extended topological soliton should be governed by the
transverse projection of its own energy density, not by an assumed Gaussian. This
script computes, from the relaxed lattice field,

    T(b) = Int dz  e(b, z),      b = sqrt(x^2 + y^2),

azimuthally averaged, normalised to T(0) = 1 and to unit rms radius so that the
overall scale is carried by the eikonal's R(s) rather than by the shape.

Output: results/soliton_profile.json  (tabulated shape, ready for rt/eikonal.py)
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

NB = 160          # radial bins


def profile(field_path, L):
    n = np.load(field_path)
    N = n.shape[1]
    h = L / N
    grid, _ = fs.make_grid(N, L)
    X, Y, Z = grid
    e2, e4 = fs.energy_density(n, h)
    e = e2 + e4

    # centre on the energy centroid, then project along z and average over phi
    tot = e.sum()
    cx, cy = (X * e).sum() / tot, (Y * e).sum() / tot
    b = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    bmax = L / 2
    edges = np.linspace(0.0, bmax, NB + 1)
    idx = np.clip(np.digitize(b.ravel(), edges) - 1, 0, NB - 1)
    # column density: sum over z within each annulus, divided by annulus area
    summed = np.bincount(idx, weights=e.ravel(), minlength=NB) * h**3
    centres = 0.5 * (edges[1:] + edges[:-1])
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    T = summed / area

    # unit rms transverse radius so the shape is scale-free
    norm = np.trapezoid(T * centres, centres)
    b_rms = np.sqrt(np.trapezoid(T * centres**3, centres) / norm)
    return centres / b_rms, T / T[0], float(b_rms)


def fit_gaussian_sum(u, T, K=10, wmin=0.15, wmax=5.0):
    """Fit T(u) = sum_k a_k exp(-u^2 / w_k^2) with a_k >= 0 on a fixed width grid.

    A Gaussian sum is used rather than a spline because each term stays analytic
    in R^2, which is what lets the eikonal's crossing-even continuation
    L -> ln(s/s0) - i pi/2 be carried out with a tabulated shape. The widths are
    held fixed on a logarithmic grid and only the non-negative amplitudes are
    solved for, which keeps the decomposition unique -- fitting widths and
    amplitudes together is badly degenerate and produces large cancelling terms.
    """
    from scipy.optimize import nnls

    w = np.geomspace(wmin, wmax, K)
    M = np.exp(-(u[:, None] ** 2) / w[None, :] ** 2)
    a, _ = nnls(M, T)
    rms = float(np.sqrt(np.mean((M @ a - T) ** 2)))
    keep = a > 1e-6
    return a[keep].tolist(), w[keep].tolist(), rms


def main():
    out = {}
    for Q in (1, 2):
        path = RES / f"field_Q{Q}.npy"
        if not path.exists():
            continue
        u, t, b_rms = profile(path, 12.0)
        keep = u <= 6.0
        u, t = u[keep], t[keep]
        a, w, rms = fit_gaussian_sum(u, t)
        out[str(Q)] = dict(u=u.tolist(), T=t.tolist(), b_rms=b_rms,
                           gauss_a=a, gauss_w=w, gauss_rms=rms)
        m = (u > 1.0) & (u < 3.0) & (t > 0)
        r_exp = np.corrcoef(u[m], np.log(t[m]))[0, 1] ** 2
        r_gauss = np.corrcoef(u[m] ** 2, np.log(t[m]))[0, 1] ** 2
        print(f"Q={Q}: b_rms={b_rms:.3f}  tail 1<u<3: exponential R^2={r_exp:.4f}, "
              f"single-Gaussian R^2={r_gauss:.4f}")
        print(f"      4-Gaussian fit rms residual = {rms:.2e}")
        print(f"      a = {np.round(a, 4).tolist()}")
        print(f"      w = {np.round(w, 4).tolist()}")

    (RES / "soliton_profile.json").write_text(json.dumps(out))
    print(f"written to {RES/'soliton_profile.json'}")


if __name__ == "__main__":
    main()
