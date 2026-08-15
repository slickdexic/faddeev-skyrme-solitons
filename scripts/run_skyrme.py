"""Frame-sector (Skyrme) solitons: relax B = 1 and B = 2, check against the
literature, and extract the baryon-density impact-parameter profile used by the
eikonal of Sec. VIII.

Usage:  python scripts/run_skyrme.py [--quick]

Outputs
-------
results/skyrme.json         : energies, baryon numbers, radii, profile fit
results/skyrme_field_B1.npy : relaxed B = 1 field
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import nnls

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, relax, skyrme as sk  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

LIT = {1: 1.2322, 2: 1.1791}   # Battye-Sutcliffe, Rev. Math. Phys. 14 (2002) 29, table 14


def extrapolate(records, exact):
    """Separate the two lattice systematics.

    The discrete energy is low by O(h^2) because finite differences underestimate
    gradients, and high by a power of the box size because a periodic box lets the
    soliton interact with its own images through the massless model's power-law
    tail. Fitting E = E_inf - a h^2 + c L^-p to every run separates them.
    """
    from scipy.optimize import curve_fit
    h = np.array([r["h"] for r in records])
    L = np.array([r["L"] for r in records])
    E = np.array([r["E_over_bound"] for r in records])

    def model(X, Einf, a, c, p):
        hh, LL = X
        return Einf - a * hh**2 + c * LL**-p

    def fit(mask):
        popt, _ = curve_fit(model, (h[mask], L[mask]), E[mask],
                            p0=[1.23, 0.8, 1.2, 2.2], maxfev=200000)
        return popt

    popt = fit(np.ones(len(E), bool))
    resid = E - model((h, L), *popt)
    # 4 parameters on 6 points makes the covariance meaningless; quote instead the
    # leave-one-out spread, which measures how much any single run drives the result
    jack = [fit(np.arange(len(E)) != i)[0] for i in range(len(E))]
    return dict(E_inf=float(popt[0]), a=float(popt[1]), c=float(popt[2]),
                p=float(popt[3]), jack_spread=float(np.ptp(jack)),
                rms_resid=float(np.sqrt(np.mean(resid**2))),
                exact=exact, rel_dev=float(popt[0] / exact - 1.0))



def upsample(phi, N):
    """Interpolate a field onto a finer lattice of the same physical box."""
    host = fs.to_numpy(phi)
    z = N / host.shape[1]
    out = np.stack([ndimage.zoom(host[a], z, order=3, mode="grid-wrap")
                    for a in range(host.shape[0])])
    return fs.normalise(fs.to_backend(out))


def fit_gaussian_sum(b, C, K=16, wmin=0.1, wmax=8.0):
    """Non-negative Gaussian-sum representation, fitted to the unbinned column
    density. Each term is analytic in R^2, which is what the crossing-even
    continuation of Sec. VIII requires."""
    w = np.geomspace(wmin, wmax, K)
    M = np.exp(-(b[:, None] ** 2) / w[None, :] ** 2)
    a, _ = nnls(M, C)
    keep = a > 1e-6 * a.max()
    return a[keep], w[keep], float(np.sqrt(np.mean((M @ a - C) ** 2)))


def measure(phi, h, L, B, tag, t0):
    grid, _ = fs.make_grid(phi.shape[1], L)
    e2, e4 = fs.energy(phi, h)
    Bn = sk.baryon_number(phi, h)
    Rr, _ = fs.energy_radius(phi, h, grid)
    rec = dict(tag=tag, N=phi.shape[1], L=L, h=h, B=float(Bn), E=float(e2 + e4),
               E_over_bound=float((e2 + e4) / (sk.FB_BOUND * abs(B))),
               virial=float(e2 / e4), R_rms=Rr,
               leak=fs.boundary_leakage(phi, h))
    print(f"    {tag:>16s} E={rec['E']:9.3f}  E/(12pi^2 B)={rec['E_over_bound']:.4f}"
          f"  (lit {LIT.get(B, float('nan')):.3f})  B={Bn:+.5f}"
          f"  virial={rec['virial']:.4f}  R={Rr:.3f}  leak={rec['leak']:.1e}"
          f"  [{time.time()-t0:.0f}s]", flush=True)
    return rec


def ladder(B, L, resolutions, steps, t0):
    """Relax at the coarsest lattice, then interpolate up and re-relax."""
    recs = []
    phi = h = None
    for k, N in enumerate(resolutions):
        grid, h = fs.make_grid(N, L)
        phi = sk.hedgehog(grid, degree=B, R=L / 4) if k == 0 else upsample(phi, N)
        ns = steps if k == 0 else max(steps // 3, 60)
        print(f"  [B={B} L={L}] N={N} h={h:.4f} steps={ns}", flush=True)
        phi, _ = relax.minimise(phi, h, steps=ns, report=max(ns // 2, 1),
                                max_rot=0.015, momentum=0.9, rescale_every=100,
                                track_charge=False, verbose=False)
        recs.append(measure(phi, h, L, B, f"B={B} L={L} N={N}", t0))
    return phi, h, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyse", action="store_true",
                    help="redo only the extrapolation from an existing skyrme.json")
    ap.add_argument("--gpu", action="store_true", help="run the lattice on the GPU")
    args = ap.parse_args()

    if args.gpu:
        fs.use_gpu()
        print("  lattice backend: GPU (float64)")

    if args.analyse:
        out = json.loads((RES / "skyrme.json").read_text())
        E_ode = sk.hedgehog_ode_energy() / sk.FB_BOUND
        out["literature"] = LIT
        out["ode"] = E_ode
        rows = out["convergence"] + out["box"]
        out["extrapolation"] = extrapolate(rows, E_ode)
        out["binding_2_lit"] = 1.0 - LIT[2] / LIT[1]
        e = out["extrapolation"]
        print(f"  exact hedgehog ODE      E/12pi^2 = {E_ode:.5f}")
        print(f"  joint (h,L) extrapolate E/12pi^2 = {e['E_inf']:.5f} "
              f"({100*e['rel_dev']:+.3f}% vs exact); leave-one-out spread "
              f"{e['jack_spread']:.5f}")
        print(f"    h^2 coefficient a = {e['a']:.3f},  box term c/L^p with "
              f"c = {e['c']:.3f}, p = {e['p']:.2f}")
        print(f"    rms residual = {e['rms_resid']:.2e}")
        b = 1.0 - out["sectors"]["2"]["E"] / (2 * out["sectors"]["1"]["E"])
        print(f"  B=2 binding {100*b:.2f}%  (literature {100*out['binding_2_lit']:.2f}%)")
        (RES / "skyrme.json").write_text(json.dumps(out, indent=2))
        return

    res = [48, 64] if args.quick else [64, 80, 96]
    steps = 400 if args.quick else 1500
    tag = "_quick" if args.quick else ""      # smoke tests must not clobber production
    L = 12.0
    t0 = time.time()
    out = {"bound": sk.FB_BOUND, "literature": LIT, "L": L,
           "convergence": [], "box": [], "sectors": {}}

    phi1, h1, recs = ladder(1, L, res, steps, t0)          # B = 1, h -> 0
    out["convergence"] = recs
    out["sectors"]["1"] = recs[-1]

    _, _, recsL = ladder(1, 8.0, res, steps, t0)           # box-size systematic
    out["box"] = recsL

    _, _, recs2 = ladder(2, L, res, 2 * steps, t0)         # B = 2 binding check
    out["sectors"]["2"] = recs2[-1]
    out["binding_2"] = 1.0 - recs2[-1]["E"] / (2 * recs[-1]["E"])
    out["binding_2_lit"] = 1.0 - LIT[2] * 2 / (LIT[1] * 2)
    print(f"\n  B=2 binding: {100*out['binding_2']:.1f}%  "
          f"(literature {100*out['binding_2_lit']:.1f}%)", flush=True)

    np.save(RES / f"skyrme_field_B1{tag}.npy", fs.to_numpy(phi1))
    b, C, b_rms = sk.column_density(phi1, h1, L, weight="baryon")
    a, w, rms = fit_gaussian_sum(b / b_rms, C)
    u, T, _ = sk.transverse_profile(phi1, h1, L, weight="baryon")
    out["profile"] = dict(b_rms=float(b_rms), gauss_a=a.tolist(),
                          gauss_w=w.tolist(), gauss_rms=rms,
                          u=u.tolist(), T=T.tolist())
    print(f"  column density: b_rms={b_rms:.4f}, {len(a)}-Gaussian fit "
          f"rms={rms:.2e} on {len(b)} unbinned samples", flush=True)

    (RES / f"skyrme{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {RES/f'skyrme{tag}.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
