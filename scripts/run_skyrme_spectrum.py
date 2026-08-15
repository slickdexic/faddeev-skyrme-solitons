"""Frame-sector spectrum B = 1..4 from symmetric rational-map seeds.

The B = 1 soliton is the nucleon of this sector; B >= 2 are the multi-baryon
bound states. Their energies test whether the framework produces a binding-energy
systematics, which is the closest it gets to computing a mass spectrum.

Usage:  python scripts/run_skyrme_spectrum.py [--gpu]
"""

import json
import pathlib
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, relax, skyrme as sk  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
L = 12.0
LIT = {1: 1.2322, 2: 1.1791, 3: 1.1462, 4: 1.1201}   # Battye-Sutcliffe table 14


def upsample(phi, N):
    host = fs.to_numpy(phi)
    z = N / host.shape[1]
    out = np.stack([ndimage.zoom(host[a], z, order=3, mode="grid-wrap")
                    for a in range(host.shape[0])])
    return fs.normalise(fs.to_backend(out))


def main():
    if "--gpu" in sys.argv:
        fs.use_gpu()
        print("  lattice backend: GPU (float64)")
    t0 = time.time()
    out = {"bound": sk.FB_BOUND, "L": L, "literature": LIT, "sectors": {}}

    for B in (1, 2, 3, 4):
        phi = None
        for N in (64, 80, 96):
            grid, h = fs.make_grid(N, L)
            phi = (sk.rational_map(grid, B, R=2.4 + 0.4 * B) if phi is None
                   else upsample(phi, N))
            steps = 2500 if N == 64 else 900
            phi, _ = relax.minimise(phi, h, steps=steps, max_rot=0.012,
                                    momentum=0.92, rescale_every=100,
                                    track_charge=False, verbose=False)
        e2, e4 = fs.energy(phi, h)
        Bn = sk.baryon_number(phi, h)
        Rr, _ = fs.energy_radius(phi, h, fs.make_grid(N, L)[0])
        rec = dict(B=B, symmetry=sk.RATIONAL_MAPS[B][1], N=N, h=h,
                   B_measured=float(Bn), E=float(e2 + e4),
                   E_over_B=float((e2 + e4) / (sk.FB_BOUND * B)),
                   virial=float(e2 / e4), R_rms=Rr,
                   leak=fs.boundary_leakage(phi, h))
        out["sectors"][str(B)] = rec
        print(f"  B={B} ({rec['symmetry']:>12}): E/(12pi^2 B)={rec['E_over_B']:.4f} "
              f"(lit {LIT[B]:.4f}, {100*(rec['E_over_B']/LIT[B]-1):+.2f}%)  "
              f"B_meas={Bn:+.4f}  R={Rr:.3f}  leak={rec['leak']:.1e}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    e1 = out["sectors"]["1"]["E"]
    print("\n  binding energy per baryon, 1 - E_B/(B E_1):")
    for B in (2, 3, 4):
        ours = 1.0 - out["sectors"][str(B)]["E"] / (B * e1)
        lit = 1.0 - LIT[B] / LIT[1]
        out["sectors"][str(B)]["binding"] = ours
        out["sectors"][str(B)]["binding_lit"] = lit
        print(f"    B={B}: {100*ours:5.2f}%   (literature {100*lit:5.2f}%)")

    (RES / "skyrme_spectrum.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {RES/'skyrme_spectrum.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
