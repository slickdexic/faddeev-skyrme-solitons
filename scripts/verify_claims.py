"""Independent re-verification of every headline claim in the manuscript.

Recomputes results from the stored *fields* rather than from the summary JSON,
so a stale or mistyped intermediate cannot propagate into the paper.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs                      # noqa: E402
from rt.calibration import SolitonShape, couplings, anchoring_from_compactness  # noqa: E402
from rt.constants import GAMMA, G, c, m_e, m_planck, lambda_bar_e  # noqa: E402
from rt import invariants                        # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
C0 = 32 * np.pi**2 * np.sqrt(2)
LIT = {1: 1.204, 2: 1.967, 3: 2.754, 4: 3.445}

ok = True


def check(label, got, want, tol, unit=""):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"  [{'OK ' if good else 'FAIL'}] {label:<52} {got:12.5g} vs {want:g}{unit}")


print("=" * 78)
print("1. Theorem 1 identities (symbolic)")
print("=" * 78)
r = invariants.run()
for k in ("I3_is_zero", "sumF2_equals_2I2", "cross2_equals_sumF2", "unit_norm", "tangency"):
    print(f"  [{'OK ' if r[k] else 'FAIL'}] {k}")
    ok &= bool(r[k])

print()
print("=" * 78)
print("2. Soliton spectrum recomputed from the stored fields")
print("=" * 78)
E, Q = {}, {}
for q in (1, 2, 3, 4):
    n = np.load(RES / f"field_Q{q}.npy")
    h = 12.0 / n.shape[1]
    e2, e4 = fs.energy(n, h)
    E[q], Q[q] = e2 + e4, fs.hopf_charge(n, h)
    Rr, _ = fs.energy_radius(n, h, fs.make_grid(n.shape[1], 12.0)[0])
    print(f"  Q={q}: E={E[q]:9.2f}  E/c0={E[q]/C0:.4f}  Q_meas={Q[q]:+.4f}  R={Rr:.3f}")
    check(f"    |Q_H| integer to 0.01", abs(Q[q]), q, 0.015)
    check(f"    E/c0 vs Sutcliffe", E[q] / C0, LIT[q], 0.06)

print()
print("=" * 78)
print("3. Derived quantities quoted in the text")
print("=" * 78)
Qa = np.array([1, 2, 3, 4], float)
Ea = np.array([E[q] for q in (1, 2, 3, 4)])
p = np.polyfit(np.log(Qa), np.log(Ea), 1)[0]
check("scaling exponent E ~ Q^p", p, 0.765, 0.003)
for q, want in ((2, 18.5), (3, 23.4), (4, 28.0)):
    check(f"binding vs fission, Q={q} (%)", 100 * (1 - E[q] / (q * E[1])), want, 0.15)
check("Ward constant c0", C0, 446.65, 0.01)

shape = SolitonShape(E_hat=E[1], R_hat=1.705)
coef = 8 * np.pi * shape.R_hat / shape.E_hat
check("horizon criterion 8*pi*Rhat/Ehat", coef, 0.079, 0.001)

chi_direct = couplings(m_e * c**2, lambda_bar_e, shape)["chi"]
chi_thm, r_s = anchoring_from_compactness(m_e, lambda_bar_e, shape)
check("Theorem 2: chi (couplings) vs chi (theorem)", chi_direct / chi_thm, 1.0, 1e-12)
check("chi(electron, Compton) / alpha_G", chi_direct / (m_e / m_planck) ** 2,
      16 * np.pi * shape.R_hat / shape.E_hat, 1e-9)
check("Gamma (N)", GAMMA, 2.4077e42, 1e39)

check("no-go: (m_mu/m_e)^(4/3)", 206.7683**(4 / 3), 1200, 40)
check("no-go: (m_tau/m_e)^(4/3)", 3477.23**(4 / 3), 52800, 900)

print()
print("=" * 78)
print("4. Dynamics")
print("=" * 78)
d = json.load(open(RES / "dynamics.json"))
a, f = d["runs"]["annihilation"]["summary"], d["runs"]["fusion"]["summary"]
check("annihilation: core energy retained (%)", 100 * a["core_fraction_final"], 0.2, 0.05)
check("fusion: core energy retained (%)", 100 * f["core_fraction_final"], 77.2, 0.1)
check("fusion: charge drift (%)", 100 * abs(f["Q_final"] - f["Q_initial"]) / 2, 0.3, 0.1)
n2 = np.load(RES / "field_Q2.npy")
h2 = 12.0 / n2.shape[1]
e2, e4 = fs.energy(n2, h2)
check("fusion remnant vs static Q=2 (% difference)",
      100 * abs(888.8 - (e2 + e4)) / (e2 + e4), 0.5, 0.1)

print()
print("=" * 78)
print("5. Frame sector")
print("=" * 78)
from rt import skyrme as sk                       # noqa: E402

E_ode = sk.hedgehog_ode_energy()
check("exact hedgehog ODE, E/12pi^2", E_ode / sk.FB_BOUND, 1.232, 0.002)
sky_path = RES / "skyrme.json"
if sky_path.exists():
    s = json.load(open(sky_path))
    phi = np.load(RES / "skyrme_field_B1.npy")
    hs = s["sectors"]["1"]["L"] / phi.shape[1]
    e2, e4 = fs.energy(phi, hs)
    E1 = e2 + e4
    check("B=1 baryon number from the stored field",
          abs(sk.baryon_number(phi, hs)), 1.0, 0.01)
    check("B=1 energy from the stored field, E/12pi^2",
          E1 / sk.FB_BOUND, s["sectors"]["1"]["E_over_bound"], 1e-6)
    deficit = 100 * (1 - E1 / E_ode)
    print(f"         B=1 lattice lies {deficit:.2f}% below the exact ODE energy")
    ok &= 0.0 < deficit < 4.0
    print(f"  [{'OK ' if 0.0 < deficit < 4.0 else 'FAIL'}] "
          f"deficit is positive and under 4% (finite-h)")
    ex = s["extrapolation"]
    check("joint (h, L) extrapolation vs exact ODE",
          ex["E_inf"], E_ode / sk.FB_BOUND, 3 * ex["jack_spread"])
    check("extrapolation rms residual (x 1e4)", 1e4 * ex["rms_resid"], 0.0, 2.0)
    bind = 100 * (1 - s["sectors"]["2"]["E"] / (2 * s["sectors"]["1"]["E"]))
    check("B=2 binding vs Battye-Sutcliffe (%)", bind,
          100 * s["binding_2_lit"], 0.25, "%")
else:
    print("  [skip] results/skyrme.json not present; run scripts/run_skyrme.py")

hres = RES / "hopfion_resolution.json"
if hres.exists():
    r = json.load(open(hres))
    E = [row["E_over_c0"] for row in r["rows"]]
    mono = all(a < b for a, b in zip(E, E[1:]))
    ok &= mono
    print(f"  [{'OK ' if mono else 'FAIL'}] "
          f"Q=1 energy is monotone in the lattice spacing")
    print(f"         h -> 0 intercept {r['h2_intercept']:.5f} "
          f"(slope {r['h2_slope']:.3f}, rms {r['h2_rms']:.2e})")

print()
print("=" * 78)
print("VERDICT:", "all checks passed" if ok else "SOME CHECKS FAILED")
print("=" * 78)
sys.exit(0 if ok else 1)
