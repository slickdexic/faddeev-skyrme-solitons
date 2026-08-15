"""Does the soliton have a long-lived internal oscillation?

A generation-like structure needs excited states with the same charge and spin as
the ground state but higher mass, i.e. a vibrational tower. The director is
massless -- its vacuum manifold is S^2, so fluctuations about the vacuum are
Goldstone modes -- which means the radiation continuum reaches down to zero
frequency and any internal oscillation is embedded in it. Whether such an
oscillation can nevertheless be long lived is a quantitative question, answered
here by exciting the breathing (scale) mode and watching it decay.

The observable is the width-to-frequency ratio Gamma/omega. A particle needs
this to be tiny: for the muon it is about 3e-19.

Usage:  python scripts/run_breathing_mode.py [--gpu]
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, ansatz, relax  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
L, N = 20.0, 96
MU = 1.06                      # initial dilation of the relaxed soliton


def main():
    if "--gpu" in sys.argv:
        fs.use_gpu()
    t0 = time.time()
    grid, h = fs.make_grid(N, L)
    n = ansatz.hedgehog_hopf(grid, 1, R=3.2)
    n, _ = relax.minimise(n, h, steps=4000, max_rot=0.010, momentum=0.92,
                          rescale_every=100, track_charge=False, verbose=False)
    e2, e4 = fs.energy(n, h)
    R0, _ = fs.energy_radius(n, h, grid)
    print(f"  relaxed: E={e2+e4:.3f}  Q={fs.hopf_charge(n,h):+.4f}  "
          f"R={R0:.4f}  virial={e2/e4:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    n = relax.rescale(n, MU)                       # excite the scale mode only
    e2, e4 = fs.energy(n, h)
    print(f"  dilated by {MU}: E={e2+e4:.3f}  (excess {e2+e4-(e2+e4):.3f})", flush=True)

    damping = relax.absorbing_mask(grid, L, width=0.16, strength=1.5)
    rec = []

    def probe(step, t, field, vel):
        R, _ = fs.energy_radius(field, h, grid)
        a2, a4 = fs.energy(field, h)
        rec.append((float(t), float(R), float(a2 + a4)))
        return None

    dt = 0.04 * h
    relax.evolve(n, h, dt=dt, steps=12000, callback=probe, every=20,
                 damping=damping, verbose=False)

    t = np.array([r[0] for r in rec])
    R = np.array([r[1] for r in rec])
    print(f"\n  R(t) over t in [0, {t[-1]:.1f}]: min {R.min():.4f}  max {R.max():.4f}"
          f"  final {R[-1]:.4f}  equilibrium {R0:.4f}")

    dR = R - R0
    sign = np.sign(dR)
    crossings = np.where(np.diff(sign) != 0)[0]
    print(f"  zero crossings of R(t) - R_eq: {len(crossings)}")
    if len(crossings) >= 2:
        period = 2.0 * np.mean(np.diff(t[crossings]))
        omega = 2.0 * np.pi / period
        peaks = np.abs(dR[crossings[0]:])
        env0, env1 = np.abs(dR[:crossings[0]]).max(), np.abs(dR[crossings[-1]:]).max()
        span = t[crossings[-1]] - t[0]
        gamma = np.log(max(env0, 1e-12) / max(env1, 1e-12)) / max(span, 1e-9)
        print(f"  period {period:.3f}  omega {omega:.4f}  "
              f"decay rate {gamma:.4f}  Gamma/omega {gamma/omega:.3f}")
        out = dict(R_eq=R0, period=float(period), omega=float(omega),
                   gamma=float(gamma), ratio=float(gamma / omega))
    else:
        print("  no oscillation resolved: the excitation relaxes without ringing")
        out = dict(R_eq=R0, oscillation=False)

    out["series"] = [[float(a), float(b), float(c)] for a, b, c in rec]
    (RES / "breathing_mode.json").write_text(json.dumps(out, indent=2))
    print(f"\n  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
