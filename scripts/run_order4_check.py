"""Fourth-order cross-check of the Q = 1 energy.

The 2.2% disagreement with the published charge-one energy rests on an O(h^2)
extrapolation, while the published value comes from a fourth-order accurate
scheme where the discretisation error at its spacing is negligible. This reruns
the same soliton with the O(h^4) symmetrised scheme, where almost no
extrapolation is needed, and asks which value it lands on.

Usage:  python scripts/run_order4_check.py [--gpu]
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


def resample(host, N):
    z = N / host.shape[1]
    out = np.stack([ndimage.zoom(host[a], z, order=3, mode="grid-wrap")
                    for a in range(host.shape[0])])
    return fs.normalise(fs.to_backend(out))


def main():
    if "--gpu" in sys.argv:
        fs.use_gpu()
    t0 = time.time()
    base = np.load(RES / "field_Q1.npy")
    rows = []
    for scheme in ("o2", "o4"):
        fs.set_scheme(scheme)
        for N in (64, 80, 96):
            h = L / N
            n = resample(base, N)
            n, _ = relax.minimise(n, h, steps=2500, max_rot=0.008, momentum=0.92,
                                  rescale_every=100, track_charge=False,
                                  verbose=False)
            e2, e4 = fs.energy(n, h)
            q = fs.hopf_charge(n, h)
            rows.append(dict(scheme=scheme, N=N, h=h, E=float(e2 + e4),
                             E_over_c0=float((e2 + e4) / C0), Q=float(q),
                             virial=float(e2 / e4)))
            print(f"  {scheme}  N={N:3d} h={h:.4f}: E/c0={rows[-1]['E_over_c0']:.5f}"
                  f"  Q={q:+.4f}  virial={e2/e4:.4f}  [{time.time()-t0:.0f}s]",
                  flush=True)
    fs.set_scheme("o2")

    print("\n  published value 1.204;  our O(h^2) extrapolation 1.2307")
    for scheme in ("o2", "o4"):
        s = [r for r in rows if r["scheme"] == scheme]
        h = np.array([r["h"] for r in s])
        E = np.array([r["E_over_c0"] for r in s])
        p = 2 if scheme == "o2" else 4
        A = np.vstack([np.ones_like(h), h**p]).T
        c, *_ = np.linalg.lstsq(A, E, rcond=None)
        print(f"    {scheme}: E -> {c[0]:.5f} as h -> 0   (fit in h^{p}, "
              f"slope {-c[1]:.3f}); spread over the three lattices "
              f"{E.max()-E.min():.5f}")

    (RES / "order4_check.json").write_text(json.dumps(dict(rows=rows), indent=2))
    print(f"\n  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
