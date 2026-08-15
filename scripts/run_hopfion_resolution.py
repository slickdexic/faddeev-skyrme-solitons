"""Rebuild the Q = 1 resolution series from a converged solution.

The published resolution study is non-monotonic in h once the production point is
included, which is impossible for a clean O(h^2) trend and points to the coarse
runs having been relaxed from a seed for a fixed number of steps rather than to
convergence. Here every lattice is initialised by resampling the *converged*
N = 96 solution, so each point has the same convergence quality.
"""

import json
import pathlib
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, relax  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
C0 = 32 * np.pi**2 * np.sqrt(2)
L = 12.0


def resample(n, N):
    z = N / n.shape[1]
    out = np.stack([ndimage.zoom(n[a], z, order=3, mode="grid-wrap")
                    for a in range(n.shape[0])])
    return fs.normalise(fs.to_backend(out))


def joint_fit(rows):
    """Separate the two lattice systematics, as in the frame sector.

    Combines this uniformly relaxed h-series at L = 12 with the fixed-h volume
    study, and fits E = E_inf - a h^2 + c L^-p. The box exponent is poorly
    determined by three box sizes at one spacing, so E_inf is also reported over
    a range of fixed p to show it is insensitive to that choice.
    """
    from scipy.optimize import curve_fit
    vol = json.loads((RES / "volume_study.json").read_text())["rows"]
    pts = [(r["h"], L, r["E_over_c0"]) for r in rows]
    pts += [(r["h"], r["L"], r["E"] / C0) for r in vol]
    h = np.array([p[0] for p in pts])
    Lb = np.array([p[1] for p in pts])
    E = np.array([p[2] for p in pts])

    out = {}
    for p in (2.0, 3.0, 4.0, 5.0, 6.0):
        def model(X, Einf, a, c, _p=p):
            hh, LL = X
            return Einf - a * hh**2 + c * LL**-_p
        popt, _ = curve_fit(model, (h, Lb), E, p0=[1.23, 1.0, 10.0], maxfev=200000)
        resid = E - model((h, Lb), *popt)
        out[f"p={p:.0f}"] = dict(E_inf=float(popt[0]), a=float(popt[1]),
                                 box_pct_at_L12=float(100 * popt[2] * 12.0**-p),
                                 rms=float(np.sqrt((resid**2).mean())))
    vals = [v["E_inf"] for v in out.values()]
    return dict(by_p=out, E_inf=float(np.mean(vals)),
                E_inf_spread=float(np.ptp(vals)), n_points=len(pts))


def main():
    if "--gpu" in sys.argv:
        fs.use_gpu()
        print("  lattice backend: GPU (float64)")
    if "--analyse" in sys.argv:
        out = json.loads((RES / "hopfion_resolution.json").read_text())
        out["joint"] = joint_fit(out["rows"])
        j = out["joint"]
        print(f"joint (h, L) fit on {j['n_points']} points:")
        for k, v in j["by_p"].items():
            print(f"  {k}: E_inf={v['E_inf']:.4f}  a={v['a']:.3f}  "
                  f"box at L=12 {v['box_pct_at_L12']:+.2f}%  rms={v['rms']:.1e}")
        print(f"  -> E_inf/c0 = {j['E_inf']:.4f} +/- {j['E_inf_spread']/2:.4f} "
              f"(spread over box models)")
        print(f"  -> E_inf    = {j['E_inf']*C0:.1f}")
        print(f"  Sutcliffe published 1.204 -> {100*(j['E_inf']/1.204-1):+.1f}%")
        (RES / "hopfion_resolution.json").write_text(json.dumps(out, indent=2))
        return

    t0 = time.time()
    base = np.load(RES / "field_Q1.npy")
    rows = []
    for N in (56, 72, 88, 96, 104, 120):
        h = L / N
        n = fs.to_backend(base) if N == base.shape[1] else resample(base, N)
        n, _ = relax.minimise(n, h, steps=500, max_rot=0.008, momentum=0.92,
                              rescale_every=100, track_charge=False, verbose=False)
        e2, e4 = fs.energy(n, h)
        q = fs.hopf_charge(n, h)
        R, _ = fs.energy_radius(n, h, fs.make_grid(N, L)[0])
        rows.append(dict(N=N, h=h, E=float(e2 + e4), E_over_c0=float((e2 + e4) / C0),
                         virial=float(e2 / e4), Q=float(q), R_rms=R,
                         leak=fs.boundary_leakage(n, h)))
        print(f"  N={N:4d} h={h:.4f}  E={e2+e4:9.3f}  E/c0={(e2+e4)/C0:.5f}  "
              f"Q={q:+.4f}  vir={e2/e4:.4f}  leak={rows[-1]['leak']:.1e}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    h = np.array([r["h"] for r in rows])
    E = np.array([r["E_over_c0"] for r in rows])
    A = np.vstack([np.ones_like(h), h**2]).T
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    resid = E - A @ coef
    print(f"\n  h^2 fit at L={L:.0f}: E/c0 -> {coef[0]:.5f} as h -> 0 "
          f"(slope {-coef[1]:.3f}), rms residual {np.sqrt((resid**2).mean()):.2e}")

    out = dict(L=L, c0=C0, rows=rows, h2_intercept=float(coef[0]),
               h2_slope=float(-coef[1]),
               h2_rms=float(np.sqrt((resid**2).mean())))
    (RES / "hopfion_resolution.json").write_text(json.dumps(out, indent=2))
    print(f"written to {RES/'hopfion_resolution.json'}")


if __name__ == "__main__":
    main()
