"""Does the rigidity bound survive a change of saturation function?

The manuscript says the framework's content is that a bound on the interaction
radius exists, not the functional form by which it is approached -- and then
quotes a numerical bound extracted with one particular form.  This tests that.

Three forms with identical limits are compared: each reduces to R_bare when
R_bare << R_inf and freezes at R_inf asymptotically, but they reach the bound at
very different rates (tanh and 1-exp exponentially, x/sqrt(1+x^2) as a power).
If the quoted bound is a property of the data rather than of the parameterisation,
it should be stable across the three.

Outputs
-------
results/saturation_forms.json
"""

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import eikonal as ek  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

FORMS = ["tanh", "exp", "algebraic"]
GRID = [4.0, 4.5, 5.0, 5.25, 5.5, 6.0, 7.0, 9.0, 14.0, 30.0, 150.0]


def profile_Rinf(values, x0, saturation):
    """Profile chi^2 over R_inf, re-minimising the remaining parameters."""
    out = []
    for R_inf in values:
        def build(p):
            return ek.Eikonal(C=abs(p[0]), Delta=abs(p[1]), R0=abs(p[2]),
                              alphap=abs(p[3]), R_inf=R_inf, saturate=True,
                              saturation=saturation)
        r = minimize(lambda p: ek.chi2(build(p)), x0, method="Nelder-Mead",
                     options=dict(maxiter=40000, maxfev=40000,
                                  xatol=1e-7, fatol=1e-7))
        out.append((R_inf, float(r.fun)))
    return out


def main():
    out = {"forms": {}}
    uns, chi_uns, _ = ek.fit(saturate=False, verbose=False)
    print(f"unsaturated reference: chi2 = {chi_uns:.2f}\n")
    out["chi2_unsaturated"] = chi_uns

    for form in FORMS:
        m, chi, _ = ek.fit(saturate=True, saturation=form, verbose=False)
        prof = profile_Rinf(GRID, [m.C, m.Delta, m.R0, m.alphap], form)
        best = min(c for _, c in prof)
        allowed = [R for R, c in prof if c - best < 3.84]
        lower_fm = min(allowed) / ek.FM_TO_GEV_INV
        rho13 = m.rho(13000.0**2)

        out["forms"][form] = dict(
            chi2=chi, R_inf_GeVinv=m.R_inf, R_inf_fm=m.R_inf / ek.FM_TO_GEV_INV,
            lower_95_fm=lower_fm, rho_13TeV=float(np.real(rho13)),
            dchi2_vs_unsaturated=chi_uns - chi, profile=prof)
        r = out["forms"][form]
        print(f"  {form:10s}  chi2 = {chi:7.2f}   "
              f"R_inf = {r['R_inf_fm']:.2f} fm   "
              f"95% CL: R_inf > {lower_fm:.2f} fm   "
              f"rho(13 TeV) = {r['rho_13TeV']:.4f}")

    lows = [out["forms"][f]["lower_95_fm"] for f in FORMS]
    rhos = [out["forms"][f]["rho_13TeV"] for f in FORMS]
    out["lower_95_fm_range"] = [min(lows), max(lows)]
    out["rho_13TeV_range"] = [min(rhos), max(rhos)]
    print()
    print(f"95% lower bound across forms: {min(lows):.2f} to {max(lows):.2f} fm")
    print(f"rho(13 TeV) across forms:     {min(rhos):.4f} to {max(rhos):.4f}")
    print(f"the quoted bound of 1.0 fm is "
          f"{'ROBUST' if min(lows) >= 0.95 else 'NOT robust'} to the choice of form")

    (RES / "saturation_forms.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/saturation_forms.json")


if __name__ == "__main__":
    main()
