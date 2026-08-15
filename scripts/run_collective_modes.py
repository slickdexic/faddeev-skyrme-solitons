"""Collective-coordinate level spacing for the charge-one Hopfion.

Rigid-body quantisation of the soliton's zero modes is the standard route to a
tower of states carrying the same charge and the same spin -- superficially what
a generation looks like.  This prices that route.  For a collective mode
n -> exp(theta K) n the kinetic term c2 |n_t|^2 gives T = (1/2) Lambda thetadot^2
with Lambda = 2 c2 Int |K n|^2, and the quantised tower is E_k = k^2 / (2 Lambda).

The quartic term contributes further positive-definite pieces to Lambda, so
omitting them *under*estimates Lambda, *over*estimates the level spacing and
*under*estimates the k needed to reach a given mass: the exclusion below is
conservative in favour of the hypothesis it excludes.

Outputs
-------
results/collective_modes.json
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

MU_OVER_E = 206.7683
TAU_OVER_E = 3477.23


def main():
    n = np.load(RES / "field_Q1.npy")
    rec = json.load(open(RES / "static_solitons.json"))["sectors"]["1"]
    breath = json.load(open(RES / "breathing_mode.json"))

    h, L, N, M0 = rec["h"], rec["L"], rec["N"], rec["E"]
    omega_b = breath["omega"]
    dv = h**3
    (X, Y, Z), _ = fs.make_grid(N, L)

    def inertia(Kn):
        return float(np.einsum("axyz,axyz->xyz", Kn, Kn).sum() * dv * 2.0)

    zero = np.zeros_like(n[0])
    modes = {
        # target rotation about the vacuum axis: K n = zhat x n
        "internal_u1": np.stack([-n[1], n[0], zero]),
        "spatial_rot_x": Y * fs.d1(n, h, axis=3) - Z * fs.d1(n, h, axis=2),
        "spatial_rot_y": Z * fs.d1(n, h, axis=1) - X * fs.d1(n, h, axis=3),
        "spatial_rot_z": X * fs.d1(n, h, axis=2) - Y * fs.d1(n, h, axis=1),
    }

    out = {"M0": M0, "h": h, "L": L, "N": N, "R_rms": rec["R_rms"],
           "omega_breathing": omega_b, "modes": {}}

    for name, Kn in modes.items():
        Lam = inertia(Kn)
        gap = 1.0 / (2.0 * Lam)
        k_mu = float(np.sqrt(2.0 * Lam * M0 * (MU_OVER_E - 1.0)))
        k_tau = float(np.sqrt(2.0 * Lam * M0 * (TAU_OVER_E - 1.0)))
        out["modes"][name] = dict(
            Lambda=Lam, gap=gap, gap_over_M0=gap / M0,
            k_for_muon=k_mu, k_for_tau=k_tau,
            omega_rot_at_k_muon=k_mu / Lam,
            adiabaticity_muon=(k_mu / Lam) / omega_b)
        m = out["modes"][name]
        print(f"{name:16s} Lambda={Lam:9.2f}  gap/M0={m['gap_over_M0']:.3e}  "
              f"k_mu={k_mu:7.0f}  w_rot/w_breath={m['adiabaticity_muon']:6.1f}")

    iso = out["modes"]["internal_u1"]
    print()
    print(f"soliton mass {M0:.2f}, R_rms {rec['R_rms']:.3f}, breathing omega {omega_b:.4f}")
    print(f"level spacing is {iso['gap_over_M0']:.2e} of the soliton mass, so the tower is")
    print(f"near-degenerate; reaching the muon needs k = {iso['k_for_muon']:.0f}, at which the")
    print(f"rotation frequency exceeds the soliton's own deformation frequency by "
          f"{iso['adiabaticity_muon']:.0f}x,")
    print("so the rigid-body approximation that produced the tower has already failed.")

    (RES / "collective_modes.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/collective_modes.json")


if __name__ == "__main__":
    main()
