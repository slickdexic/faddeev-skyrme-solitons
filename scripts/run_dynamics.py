"""Dynamical test of topological protection: Hopfion + anti-Hopfion vs Hopfion + Hopfion.

Two collisions are evolved with identical initial energy and velocity:

  (a) Q = +1 and Q = -1  -> total Q_H = 0.  The pair is in the topologically
      trivial sector, so nothing forbids relaxation to the vacuum: the cores
      merge and the entire rest energy is converted into outgoing director
      radiation.  This is the field-theoretic content of "annihilation".

  (b) Q = +1 and Q = +1  -> total Q_H = +2.  The sector is non-trivial, so a
      localised remnant must survive; the pair radiates only its binding energy
      and settles onto the Q = 2 soliton.

The contrast isolates topology as the operative mechanism: the two runs differ
only by a spatial reflection of one constituent.

Outputs: results/dynamics.json, results/dynamics_{annihilation,fusion}.npz
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import ansatz, fs_core as fs, relax  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

N, L = 96, 20.0
SEP = 4.0            # half-separation of the initial cores
VEL = 0.30           # approach speed (units of the wave speed c)
STEPS = 2600
EVERY = 50
CORE_RADIUS = 5.0    # sphere used to separate "localised" from "radiated" energy


def boost(n, h, v, axis=3):
    """Initial velocity field for a configuration translating with speed v along `axis`."""
    return -v * fs.d1(n, h, axis=axis)


def build(grid, h, mode):
    a = ansatz.hedgehog_hopf(grid, degree=1, R=4.0, centre=(0.0, 0.0, -SEP))
    if mode == "annihilation":
        # the reflection z -> -z carries a core at -SEP to +SEP and flips Q_H
        b = ansatz.conjugate(ansatz.hedgehog_hopf(grid, degree=1, R=4.0,
                                                  centre=(0.0, 0.0, -SEP)))
    else:
        b = ansatz.hedgehog_hopf(grid, degree=1, R=4.0, centre=(0.0, 0.0, SEP))
    n = ansatz.superpose([a, b], grid)
    va = boost(a, h, +VEL)
    vb = boost(b, h, -VEL)
    v = np.where(grid[2][None, ...] < 0.0, va, vb)
    v -= np.einsum("axyz,axyz->xyz", v, n) * n
    return fs.normalise(n), v


def make_callback(grid, h, log):
    X, Y, Z = grid
    r2 = X**2 + Y**2 + Z**2
    core = r2 < CORE_RADIUS**2

    def cb(step, t, n, vel):
        e2, e4 = fs.energy_density(n, h)
        e = e2 + e4
        kin = 1.0 * np.einsum("axyz,axyz->xyz", vel, vel)      # c2 = 1
        tot = float((e + kin).sum() * h**3)
        loc = float(((e + kin) * core).sum() * h**3)
        q = fs.hopf_charge(n, h)
        row = dict(step=step, t=float(t), E_total=tot, E_core=loc,
                   E_grad=float(e.sum() * h**3), E_kin=float(kin.sum() * h**3),
                   Q=float(q))
        log.append(row)
        return row

    return cb


def run(mode):
    grid, h = fs.make_grid(N, L)
    n0, v0 = build(grid, h, mode)
    damping = relax.absorbing_mask(grid, L, width=0.20, strength=2.0)
    log = []
    cb = make_callback(grid, h, log)
    print(f"  [{mode}] N={N} L={L} h={h:.4f}  Q_initial={fs.hopf_charge(n0,h):+.4f}")
    t0 = time.time()
    n, v, _ = relax.evolve(n0, h, dt=0.05 * h, steps=STEPS, v0=v0, callback=cb,
                           every=EVERY, damping=damping, verbose=False)
    print(f"  [{mode}] done in {time.time()-t0:.0f}s")
    return grid, h, n, v, log


def main(modes=("annihilation", "fusion")):
    path = RES / "dynamics.json"
    out = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    out["config"] = dict(N=N, L=L, separation=SEP, velocity=VEL, steps=STEPS,
                         core_radius=CORE_RADIUS)
    for mode in modes:
        grid, h, n, v, log = run(mode)
        first, last = log[0], log[-1]
        summary = dict(
            Q_initial=first["Q"], Q_final=last["Q"],
            E_initial=first["E_total"], E_final=last["E_total"],
            E_core_initial=first["E_core"], E_core_final=last["E_core"],
            core_fraction_final=last["E_core"] / first["E_total"],
            radiated_fraction=1.0 - last["E_core"] / first["E_total"],
        )
        out["runs"][mode] = dict(summary=summary, trace=log)
        e2, e4 = fs.energy(n, h)
        print(f"  [{mode}] Q: {first['Q']:+.4f} -> {last['Q']:+.4f} | "
              f"E_core: {first['E_core']:.1f} -> {last['E_core']:.1f} "
              f"({100*summary['core_fraction_final']:.1f}% retained) | "
              f"final static E = {e2+e4:.1f}")
        np.savez_compressed(RES / f"dynamics_{mode}.npz", n=n.astype(np.float32))

    (RES / "dynamics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {RES/'dynamics.json'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(tuple(args) if args else ("annihilation", "fusion"))
