"""Diagnostics for the frame-sector B = 1 result.

Two questions the production run leaves open:
  (a) is the residual virial violation E2/E4 < 1 under-relaxation or discretisation?
  (b) where does the h -> 0 extrapolation actually land?
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, relax, skyrme as sk  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
L = 12.0


def main():
    phi = np.load(RES / "skyrme_field_B1.npy")
    N = phi.shape[1]
    h = L / N
    E_ode = sk.hedgehog_ode_energy()
    e2, e4 = fs.energy(phi, h)
    print(f"loaded N={N} h={h:.4f}")
    print(f"  start   E/12pi^2 = {(e2+e4)/sk.FB_BOUND:.5f}  virial = {e2/e4:.5f}")

    t0 = time.time()
    for block in range(6):
        phi, _ = relax.minimise(phi, h, steps=500, max_rot=0.010, momentum=0.92,
                                rescale_every=50, track_charge=False, verbose=False)
        e2, e4 = fs.energy(phi, h)
        print(f"  +{500*(block+1):5d}  E/12pi^2 = {(e2+e4)/sk.FB_BOUND:.5f}  "
              f"virial = {e2/e4:.5f}  B = {sk.baryon_number(phi, h):+.5f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\nexact ODE E/12pi^2 = {E_ode/sk.FB_BOUND:.5f}")
    np.save(RES / "skyrme_field_B1.npy", phi)
    print("re-saved relaxed field")


if __name__ == "__main__":
    main()
