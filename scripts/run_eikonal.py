"""Forward elastic pp scattering: fit, profile likelihood on R_inf, and predictions.

Outputs: results/eikonal.json, figures/forward_scattering.png
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import eikonal as ek  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

PRED_ENERGIES = [7000.0, 8000.0, 13000.0, 14000.0, 27000.0, 100000.0, 1.0e6]


def profile_Rinf(values, x0, shape="gaussian"):
    """Profile chi^2 over R_inf, re-minimising the remaining parameters."""
    from scipy.optimize import minimize
    out = []
    for R_inf in values:
        def build(p):
            return ek.Eikonal(C=abs(p[0]), Delta=abs(p[1]), R0=abs(p[2]),
                              alphap=abs(p[3]), R_inf=R_inf, saturate=True,
                              profile_shape=shape)
        r = minimize(lambda p: ek.chi2(build(p)), x0, method="Nelder-Mead",
                     options=dict(maxiter=20000, maxfev=20000, xatol=1e-7, fatol=1e-7))
        out.append((R_inf, float(r.fun)))
    return out


def dip_position(model, s, tmin=0.15, tmax=1.5, npts=4000):
    """First local minimum of dsigma/dt, i.e. the diffractive dip."""
    t = -np.linspace(tmin, tmax, npts)
    y = np.log(model.dsigma_dt(s, t, in_mb=False))
    for i in range(1, npts - 1):
        if y[i] < y[i - 1] and y[i] < y[i + 1]:
            return float(-t[i])
    return float("nan")


def main():
    print("== fits ==")
    m_uns, chi_uns, npar_uns = ek.fit(saturate=False)
    m_sat, chi_sat, npar_sat = ek.fit(saturate=True)
    ndat = ek.n_data()
    print(f"  Delta chi2 (unsaturated - saturated) = {chi_uns - chi_sat:.2f} "
          f"for one extra parameter")

    print("\n== profile likelihood on the rigidity radius R_inf ==")
    grid = [4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 14.0, 20.0, 40.0, 200.0]
    prof = profile_Rinf(grid, [m_sat.C, m_sat.Delta, m_sat.R0, m_sat.alphap])
    chi_min = min(c for _, c in prof)
    print(f"  {'R_inf [GeV^-1]':>14} {'R_inf [fm]':>11} {'chi2':>9} {'dchi2':>8}")
    for R, c in prof:
        print(f"  {R:14.1f} {R/ek.FM_TO_GEV_INV:11.3f} {c:9.2f} {c-chi_min:8.2f}")
    allowed = [R for R, c in prof if c - chi_min < 3.84]
    print(f"  95% CL (dchi2 < 3.84): R_inf > {min(allowed):.1f} GeV^-1 "
          f"= {min(allowed)/ek.FM_TO_GEV_INV:.3f} fm")

    print("\n== predictions ==")
    hdr = (f"  {'sqrt(s)':>9} | {'sig_tot [mb]':>22} | {'rho':>19} | "
           f"{'B [GeV^-2]':>19} | {'sig_el/sig_tot':>17}")
    print(hdr)
    print(f"  {'[TeV]':>9} | {'unsat':>10} {'saturated':>11} | "
          f"{'unsat':>9} {'sat':>9} | {'unsat':>9} {'sat':>9} | {'unsat':>8} {'sat':>8}")
    rows = []
    for rs in PRED_ENERGIES:
        s = rs**2
        row = dict(sqrt_s_GeV=rs,
                   sigma_unsat=m_uns.sigma_tot(s), sigma_sat=m_sat.sigma_tot(s),
                   rho_unsat=m_uns.rho(s), rho_sat=m_sat.rho(s),
                   B_unsat=m_uns.slope(s), B_sat=m_sat.slope(s),
                   ratio_unsat=m_uns.sigma_el(s) / m_uns.sigma_tot(s),
                   ratio_sat=m_sat.sigma_el(s) / m_sat.sigma_tot(s),
                   R_sat_fm=m_sat.radius_fm(s))
        rows.append(row)
        print(f"  {rs/1000:9.2f} | {row['sigma_unsat']:10.2f} {row['sigma_sat']:11.2f} | "
              f"{row['rho_unsat']:+9.4f} {row['rho_sat']:+9.4f} | "
              f"{row['B_unsat']:9.2f} {row['B_sat']:9.2f} | "
              f"{row['ratio_unsat']:8.3f} {row['ratio_sat']:8.3f}")

    payload = dict(
        unsaturated=dict(C=m_uns.C, Delta=m_uns.Delta, R0=m_uns.R0,
                         alphap=m_uns.alphap, chi2=chi_uns, npar=npar_uns),
        saturated=dict(C=m_sat.C, Delta=m_sat.Delta, R0=m_sat.R0,
                       alphap=m_sat.alphap, R_inf=m_sat.R_inf, chi2=chi_sat,
                       npar=npar_sat, R_inf_fm=m_sat.R_inf / ek.FM_TO_GEV_INV),
        n_data=ndat, profile=prof, predictions=rows,
        residuals=dict(
            sigma=[(rs, lab, v, e, m_sat.sigma_tot(rs**2),
                    (m_sat.sigma_tot(rs**2) - v) / e) for rs, v, e, lab in ek.SIGMA_TOT],
            rho=[(rs, lab, v, e, m_sat.rho(rs**2),
                  (m_sat.rho(rs**2) - v) / e) for rs, v, e, lab in ek.RHO],
            slope=[(rs, lab, v, e, m_sat.slope(rs**2),
                    (m_sat.slope(rs**2) - v) / e) for rs, v, e, lab in ek.SLOPE],
        ),
    )

    print("\n== profile-shape dependence ==")
    shapes = {}
    for shape in ("dipole", "skyrmion"):
        try:
            s_sat, s_chi, _ = ek.fit(saturate=True, profile_shape=shape)
            s_uns, su_chi, _ = ek.fit(saturate=False, profile_shape=shape)
        except FileNotFoundError as exc:
            print(f"  [{shape}] skipped: {exc}")
            continue
        shapes[shape] = (s_sat, s_uns, s_chi, su_chi)

    s13 = 13000.0**2
    print(f"\n  {'':10} {'chi2':>8} {'rho(13)':>9} {'sig(13)':>9} {'B(13)':>7}")
    print(f"  {'gaussian':10} {ek.chi2(m_sat):8.2f} {m_sat.rho(s13):9.4f} "
          f"{m_sat.sigma_tot(s13):9.2f} {m_sat.slope(s13):7.2f}")
    for nm, (s_sat, *_ignore) in shapes.items():
        print(f"  {nm:10} {ek.chi2(s_sat):8.2f} {s_sat.rho(s13):9.4f} "
              f"{s_sat.sigma_tot(s13):9.2f} {s_sat.slope(s13):7.2f}")

    measured = {2760: 0.72, 7000: 0.53, 8000: 0.52, 13000: 0.47}
    for shape, (s_sat, s_uns, s_chi, su_chi) in shapes.items():
        s_prof = profile_Rinf([4.5, 5.0, 5.25, 5.5, 6.0, 7.0, 9.0, 14.0, 30.0, 150.0],
                              [s_sat.C, s_sat.Delta, s_sat.R0, s_sat.alphap],
                              shape=shape)
        smin = min(c for _, c in s_prof)
        allowed = [R for R, c in s_prof if c - smin < 3.84]
        print(f"\n  {shape} 95% CL: R_inf > {min(allowed):.2f} GeV^-1 = "
              f"{min(allowed)/ek.FM_TO_GEV_INV:.3f} fm")

        print(f"  diffractive dip ({shape} profile)")
        dip_rows = []
        print(f"  {'sqrt(s) [TeV]':>13} {'saturated':>10} {'unsat':>8} {'measured':>9}")
        for rs in (2760, 7000, 8000, 13000, 14000, 27000, 100000):
            a, b = dip_position(s_sat, rs**2), dip_position(s_uns, rs**2)
            mm = measured.get(rs)
            dip_rows.append(dict(sqrt_s_GeV=rs, sat=a, uns=b, measured=mm))
            print(f"  {rs/1000:13.2f} {a:10.3f} {b:8.3f} "
                  f"{(f'{mm:.2f}' if mm else '--'):>9}")

        payload[shape] = dict(chi2_sat=s_chi, chi2_uns=su_chi, profile=s_prof,
                              R_inf_lower_fm=min(allowed) / ek.FM_TO_GEV_INV,
                              rho13=s_sat.rho(s13), sigma13=s_sat.sigma_tot(s13),
                              slope13=s_sat.slope(s13), dip=dip_rows)
    (RES / "eikonal.json").write_text(json.dumps(payload, indent=2))


    make_figure(m_uns, m_sat)
    print(f"\nwritten to {RES/'eikonal.json'} and {FIG/'forward_scattering.png'}")


def make_figure(m_uns, m_sat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = np.logspace(np.log10(300.0), np.log10(2.0e6), 160)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax[0].plot(rs / 1000, [m_uns.sigma_tot(r**2) for r in rs], "k--",
               label=r"unsaturated ($\ln^2 s$)")
    ax[0].plot(rs / 1000, [m_sat.sigma_tot(r**2) for r in rs], "C3-",
               label=r"rigidity-saturated")
    for r, v, e, lab in ek.SIGMA_TOT:
        ax[0].errorbar(r / 1000, v, yerr=e, fmt="o", ms=4, color="C0", capsize=2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\sqrt{s}$ [TeV]")
    ax[0].set_ylabel(r"$\sigma_{\rm tot}$ [mb]")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].grid(alpha=0.25)

    ax[1].plot(rs / 1000, [m_uns.rho(r**2) for r in rs], "k--")
    ax[1].plot(rs / 1000, [m_sat.rho(r**2) for r in rs], "C3-")
    for r, v, e, lab in ek.RHO:
        ax[1].errorbar(r / 1000, v, yerr=e, fmt="o", ms=4, color="C0", capsize=2)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\sqrt{s}$ [TeV]")
    ax[1].set_ylabel(r"$\rho$")
    ax[1].grid(alpha=0.25)

    fig.suptitle("Forward elastic $pp$ scattering: rigidity-saturated vs. "
                 "Froissart-saturating eikonal", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "forward_scattering.png", dpi=160)


if __name__ == "__main__":
    main()
