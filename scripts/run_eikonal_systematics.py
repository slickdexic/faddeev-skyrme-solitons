"""Systematic checks on the forward-scattering fit.

The dominant experimental tension in the dataset is the TOTEM/ATLAS disagreement
on sigma_tot and B at 13 TeV. This script repeats the fit and the R_inf bound on
three subsets to show how much of the result depends on that tension.

Output: results/eikonal_systematics.json
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
RES.mkdir(exist_ok=True)


def subset(keep):
    """Return (sigma, rho, slope) restricted to labels containing any of `keep`."""
    def f(rows):
        return [r for r in rows if any(k in r[3] for k in keep)]
    return f(ek.SIGMA_TOT), f(ek.RHO), f(ek.SLOPE)


def chi2_on(model, sig, rho, slp):
    c = 0.0
    for rs, v, e, _ in sig:
        c += ((model.sigma_tot(rs**2) - v) / e) ** 2
    for rs, v, e, _ in rho:
        c += ((model.rho(rs**2) - v) / e) ** 2
    for rs, v, e, _ in slp:
        c += ((model.slope(rs**2) - v) / e) ** 2
    return float(c)


def fit_on(sig, rho, slp, R_inf=None, x0=(0.46, 0.09, 3.9, 0.3)):
    def build(p):
        kw = dict(C=abs(p[0]), Delta=abs(p[1]), R0=abs(p[2]), alphap=abs(p[3]))
        if R_inf is None:
            kw["saturate"] = False
        else:
            kw["saturate"] = True
            kw["R_inf"] = R_inf
        return ek.Eikonal(**kw)

    r = minimize(lambda p: chi2_on(build(p), sig, rho, slp), list(x0),
                 method="Nelder-Mead",
                 options=dict(maxiter=30000, maxfev=30000, xatol=1e-7, fatol=1e-7))
    return build(r.x), float(r.fun)


def bound_on(sig, rho, slp, grid=(5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 8.0, 10.0,
                                  14.0, 20.0, 60.0, 300.0)):
    prof = [(R, fit_on(sig, rho, slp, R_inf=R)[1]) for R in grid]
    cmin = min(c for _, c in prof)
    allowed = [R for R, c in prof if c - cmin < 3.84]
    return min(allowed), prof, cmin


DATASETS = {
    "all": ["UA4", "UA5", "E710", "TOTEM", "ATLAS"],
    "TOTEM only at LHC": ["UA4", "UA5", "E710", "TOTEM"],
    "ATLAS only at LHC": ["UA4", "UA5", "E710", "ATLAS"],
    "LHC pp only": ["TOTEM", "ATLAS"],
}


def main():
    out = {}
    print(f"{'dataset':22s} {'N':>4} {'chi2/dof unsat':>15} {'chi2/dof sat':>13} "
          f"{'dchi2':>7} {'R_inf 95% CL lower':>20}")
    for name, keep in DATASETS.items():
        sig, rho, slp = subset(keep)
        n = len(sig) + len(rho) + len(slp)
        _mu, cu = fit_on(sig, rho, slp)
        Rlim, prof, cmin = bound_on(sig, rho, slp)
        out[name] = dict(n_data=n, chi2_unsat=cu, chi2_sat_min=cmin,
                         R_inf_lower_GeVinv=Rlim,
                         R_inf_lower_fm=Rlim / ek.FM_TO_GEV_INV,
                         profile=prof)
        print(f"{name:22s} {n:4d} {cu/(n-4):15.2f} {cmin/(n-5):13.2f} "
              f"{cu-cmin:7.2f} {Rlim/ek.FM_TO_GEV_INV:17.3f} fm")

    (RES / "eikonal_systematics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {RES/'eikonal_systematics.json'}")


if __name__ == "__main__":
    main()
