"""Is the stored Q = 1 Hopfion at the bottom of its own well?

Continued relaxation of the production field, to separate a converged solution
from one that merely stopped after a fixed number of steps. The lattice-spacing
series itself lives in run_hopfion_resolution.py.
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, relax  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
C0 = 32 * np.pi**2 * np.sqrt(2)
L = 12.0


def report(tag, n, h, t0):
    e2, e4 = fs.energy(n, h)
    q = fs.hopf_charge(n, h)
    R, _ = fs.energy_radius(n, h, fs.make_grid(n.shape[1], L)[0])
    print(f"    {tag:22s} E={e2+e4:9.3f}  E/c0={(e2+e4)/C0:.5f}  Q={q:+.4f}  "
          f"vir={e2/e4:.4f}  R={R:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    return dict(tag=tag, E=float(e2 + e4), E_over_c0=float((e2 + e4) / C0),
                Q=float(q), virial=float(e2 / e4), R_rms=R)


def main():
    if "--gpu" in sys.argv:
        fs.use_gpu()
    t0 = time.time()
    n = fs.to_backend(np.load(RES / "field_Q1.npy"))
    h = L / n.shape[1]
    out = {"c0": C0, "L": L, "h": h, "stages": [report("stored (before)", n, h, t0)]}
    for k in range(3):
        n, _ = relax.minimise(n, h, steps=600, max_rot=0.008, momentum=0.92,
                              rescale_every=60, track_charge=False, verbose=False)
        out["stages"].append(report(f"stored (+{600*(k+1)})", n, h, t0))

    first, last = out["stages"][0]["E"], out["stages"][-1]["E"]
    out["drift"] = abs(last / first - 1.0)
    print(f"\n  relative energy drift over 1800 further steps: {out['drift']:.2e}")
    (RES / "hopfion_convergence.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
