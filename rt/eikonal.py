"""Forward elastic pp scattering with a rigidity-saturated interaction radius.

Physical setting
----------------
The proton is an extended topological soliton of the vacuum director sector.
Its strain field sets the eikonal opacity.  The standard (Froissart-saturating)
picture takes a supercritical Pomeron-like opacity

    Omega(b, s) = C (s/s_0)^Delta exp( -b^2 / R^2(s) ),
    R^2(s)      = R_0^2 + 4 alpha' L ,          L = ln(s/s_0),

whose blackness radius grows like L, giving sigma_tot ~ ln^2 s.

Resonant Topology adds one ingredient: the vacuum's quartic rigidity bounds the
strain that can be transmitted, so the *range* of the interaction cannot grow
without limit.  We implement this minimally as

    R(s) -> R_inf tanh( R_bare(s) / R_inf ),

which is exactly the unsaturated model for R_bare << R_inf and freezes at R_inf
asymptotically.  The two hypotheses are nested (R_inf -> infinity recovers the
conventional model), so they can be compared by a likelihood-ratio test.

Saturation converts the asymptotic growth from ln^2 s to ln s, and therefore
*halves* the asymptotic rho parameter, since analyticity ties rho to the local
logarithmic growth rate.  This is the framework's sharpest forward-scattering
signature.

Analyticity
-----------
For crossing-even amplitudes the real part follows with no extra parameters from
the substitution L -> ln(s/s_0) - i pi/2, applied inside every s-dependent
quantity.

Conventions (GeV units)
-----------------------
    F(s, t)   = 2 pi i Int_0^inf b db J_0(q b) [ 1 - exp(-Omega(b, s)) ] , q^2 = -t
    sigma_tot = 2 Im F(s, 0)
    rho       = Re F(s, 0) / Im F(s, 0)
    dsigma/dt = |F(s, t)|^2 / (4 pi)
    sigma_el  = 2 pi Int b db |1 - exp(-Omega)|^2
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import j0

GEV2_TO_MB = 0.3893793721        # 1 GeV^-2 expressed in mb
FM_TO_GEV_INV = 5.067730716      # 1 fm expressed in GeV^-1

_NB = 3000                        # impact-parameter quadrature points


def _dipole_overlap(u):
    """Impact-parameter overlap of two dipole form factors, normalised to 1 at u=0.

    For G(t) = (1 + |t|/Lambda^2)^{-2} the b-space overlap is proportional to
    u^3 K_3(u) with u = Lambda b; the u -> 0 limit of u^3 K_3(u) is 8.
    """
    from scipy.special import kv
    u = np.asarray(u, dtype=float)
    out = np.ones_like(u)
    nz = u > 1e-8
    out[nz] = u[nz] ** 3 * kv(3, u[nz]) / 8.0
    return out


def _fit_gaussian_sum(shape_fn, K=14, wmin=0.1, wmax=8.0, npts=600):
    """Represent a radial shape as sum_k a_k exp(-u^2/w_k^2) with a_k >= 0.

    The Gaussian sum is what makes a non-Gaussian profile usable here: each term
    remains analytic in R^2, so the crossing-even continuation
    L -> ln(s/s0) - i pi/2 still goes through with complex R. Widths are fixed on
    a logarithmic grid and only the non-negative amplitudes are solved for, which
    keeps the decomposition unique.
    """
    from scipy.optimize import nnls
    u = np.linspace(0.0, 6.0, npts)
    T = shape_fn(u)
    rms = np.sqrt(np.trapezoid(T * u**3, u) / np.trapezoid(T * u, u))
    u = u / rms                                   # unit rms radius
    w = np.geomspace(wmin, wmax, K)
    M = np.exp(-(u[:, None] ** 2) / w[None, :] ** 2)
    a, _ = nnls(M, T)
    keep = a > 1e-8
    return np.asarray(a[keep]), np.asarray(w[keep]), float(np.sqrt(
        np.mean((M @ a - T) ** 2)))


_DIPOLE_A, _DIPOLE_W, _DIPOLE_RES = _fit_gaussian_sum(_dipole_overlap)

_SKYRME_CACHE = {}


def _skyrme_shape():
    """Gaussian-sum representation of the B = 1 Skyrmion column density.

    Read from results/skyrme.json, which is produced by scripts/run_skyrme.py by
    fitting the unbinned lattice column density of the relaxed frame-sector
    soliton. The widths are already expressed in units of the profile's rms
    radius, matching the convention of `_fit_gaussian_sum`.
    """
    if "aw" not in _SKYRME_CACHE:
        import json
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "results" / "skyrme.json"
        if not p.exists():
            raise FileNotFoundError(
                "profile_shape='skyrmion' needs results/skyrme.json; "
                "run scripts/run_skyrme.py first")
        d = json.loads(p.read_text())["profile"]
        _SKYRME_CACHE["aw"] = (np.asarray(d["gauss_a"]), np.asarray(d["gauss_w"]))
    return _SKYRME_CACHE["aw"]


@dataclass
class Eikonal:
    C: float = 1.0                # opacity normalisation
    Delta: float = 0.10           # effective Pomeron intercept - 1
    R0: float = 3.5               # GeV^-1
    alphap: float = 0.25          # GeV^-2, Pomeron slope
    R_inf: float = 6.0            # GeV^-1, rigidity-bounded maximum radius
    s0: float = 1.0               # GeV^2
    saturate: bool = True
    profile_shape: str = "gaussian"   # "gaussian", "dipole" or "skyrmion"
    saturation: str = "tanh"          # "tanh", "exp" or "algebraic"

    # ------------------------------------------------------------------ core
    def _L(self, s):
        return np.log(s / self.s0) - 0.5j * np.pi

    def _R(self, s):
        L = self._L(s)
        R_bare = np.sqrt(self.R0**2 + 4.0 * self.alphap * L + 0j)
        if not self.saturate:
            return R_bare
        # every form obeys f -> R_bare as R_bare -> 0 and f -> R_inf as R_bare -> inf;
        # they differ only in how fast the bound is approached
        x = R_bare / self.R_inf
        if self.saturation == "tanh":
            return self.R_inf * np.tanh(x)
        if self.saturation == "exp":
            return self.R_inf * (1.0 - np.exp(-x))
        if self.saturation == "algebraic":
            return R_bare / np.sqrt(1.0 + x * x)
        raise ValueError(self.saturation)

    def _omega0(self, s):
        return self.C * np.exp(self.Delta * self._L(s))

    def _bgrid(self, s):
        R = abs(self._R(s))
        bmax = 9.0 * R
        b = np.linspace(bmax / (2 * _NB), bmax, _NB)
        return b, bmax / _NB

    def profile(self, b, s):
        """Complex profile function h(b, s) = 1 - exp(-Omega(b, s))."""
        R2 = self._R(s) ** 2
        if self.profile_shape == "gaussian":
            shape = np.exp(-(b**2) / R2)
        else:
            av, wv = (_DIPOLE_A, _DIPOLE_W) if self.profile_shape == "dipole" \
                else _skyrme_shape()
            shape = sum(a * np.exp(-(b**2) / (w**2 * R2))
                        for a, w in zip(av, wv))
        return 1.0 - np.exp(-self._omega0(s) * shape)

    # ---------------------------------------------------------- observables
    def amplitude(self, s, t):
        b, db = self._bgrid(s)
        h = self.profile(b, s)
        q = np.sqrt(np.maximum(-np.asarray(t, dtype=float), 0.0))
        q = np.atleast_1d(q)[:, None]
        integ = (j0(q * b[None, :]) * h[None, :] * b[None, :]).sum(axis=1) * db
        return np.squeeze(2.0j * np.pi * integ)

    def sigma_tot(self, s, in_mb=True):
        val = 2.0 * np.imag(self.amplitude(s, 0.0))
        return float(val) * (GEV2_TO_MB if in_mb else 1.0)

    def rho(self, s):
        F0 = self.amplitude(s, 0.0)
        return float(np.real(F0) / np.imag(F0))

    def dsigma_dt(self, s, t, in_mb=True):
        val = np.abs(self.amplitude(s, t)) ** 2 / (4.0 * np.pi)
        return val * (GEV2_TO_MB if in_mb else 1.0)

    def sigma_el(self, s, in_mb=True):
        b, db = self._bgrid(s)
        h = self.profile(b, s)
        val = 2.0 * np.pi * float((np.abs(h) ** 2 * b).sum() * db)
        return val * (GEV2_TO_MB if in_mb else 1.0)

    def slope(self, s, tmin=0.005, tmax=0.2, npts=40):
        """Effective forward slope B from a straight-line fit of ln(dsigma/dt)
        over |t| in [tmin, tmax], i.e. the quantity experiments report."""
        t = -np.linspace(tmin, tmax, npts)
        y = np.log(self.dsigma_dt(s, t, in_mb=False))
        return float(np.polyfit(t, y, 1)[0])

    def radius_fm(self, s):
        return float(abs(self._R(s))) / FM_TO_GEV_INV

    def dip(self, s, tmin=0.2, tmax=1.6, npts=1400):
        """Position |t_dip| [GeV^2] and value of the first diffractive minimum.

        The dip sits where the black-disc and peripheral contributions cancel, so
        |t_dip| ~ 1/R^2: if the interaction radius saturates, the dip stops moving
        with energy. This is the framework's sharpest near-term signature.
        """
        t = -np.linspace(tmin, tmax, npts)
        y = self.dsigma_dt(s, t, in_mb=False)
        i = int(np.argmin(y))
        if i in (0, npts - 1):
            return float("nan"), float("nan")
        return float(-t[i]), float(y[i])

    def unsaturated(self):
        return replace(self, saturate=False)


# --------------------------------------------------------------------------
# measured forward observables (sqrt(s) in GeV)
# --------------------------------------------------------------------------
SIGMA_TOT = [
    (546.0,   61.9,  1.5,  "UA4 (ppbar)"),
    (630.0,   62.2,  1.5,  "UA5 (ppbar)"),
    (1800.0,  71.71, 2.02, "E710/E811 (ppbar)"),
    (2760.0,  84.7,  3.3,  "TOTEM"),
    (7000.0,  98.0,  2.5,  "TOTEM"),
    (7000.0,  95.35, 1.36, "ATLAS-ALFA"),
    (8000.0, 101.7,  2.9,  "TOTEM"),
    (8000.0,  96.07, 0.92, "ATLAS-ALFA"),
    (13000.0, 110.5, 2.4,  "TOTEM"),
    (13000.0, 104.7, 1.1,  "ATLAS-ALFA"),
]

RHO = [
    (546.0,   0.135, 0.015, "UA4/2 (ppbar)"),
    (1800.0,  0.140, 0.069, "E710 (ppbar)"),
    (8000.0,  0.12,  0.03,  "TOTEM"),
    (13000.0, 0.10,  0.01,  "TOTEM"),
    (13000.0, 0.098, 0.011, "ATLAS-ALFA"),
]

SLOPE = [
    (546.0,  15.35, 0.30, "UA4 (ppbar)"),
    (1800.0, 16.99, 0.47, "E710 (ppbar)"),
    (2760.0, 17.1,  0.3,  "TOTEM"),
    (7000.0, 19.89, 0.27, "TOTEM"),
    (8000.0, 19.74, 0.19, "ATLAS-ALFA"),
    (13000.0, 20.36, 0.19, "TOTEM"),
    (13000.0, 21.14, 0.17, "ATLAS-ALFA"),
]


def chi2(model: Eikonal, use_slope=True, extra_sigma_sys=0.0):
    c = 0.0
    for rs, val, err, _ in SIGMA_TOT:
        e = np.hypot(err, extra_sigma_sys)
        c += ((model.sigma_tot(rs**2) - val) / e) ** 2
    for rs, val, err, _ in RHO:
        c += ((model.rho(rs**2) - val) / err) ** 2
    if use_slope:
        for rs, val, err, _ in SLOPE:
            c += ((model.slope(rs**2) - val) / err) ** 2
    return float(c)


def n_data(use_slope=True):
    return len(SIGMA_TOT) + len(RHO) + (len(SLOPE) if use_slope else 0)


def fit(saturate=True, x0=None, use_slope=True, extra_sigma_sys=0.0, verbose=True,
        profile_shape="gaussian", saturation="tanh"):
    """Fit (C, Delta, R0, alpha', [R_inf]) by chi-square minimisation."""
    from scipy.optimize import minimize

    def build(p):
        kw = dict(C=abs(p[0]), Delta=abs(p[1]), R0=abs(p[2]), alphap=abs(p[3]),
                  saturate=saturate, profile_shape=profile_shape)
        if saturate:
            kw["R_inf"] = abs(p[4])
            kw["saturation"] = saturation
        return Eikonal(**kw)

    p0 = list(x0) if x0 is not None else [1.0, 0.10, 3.5, 0.25] + ([6.0] if saturate else [])
    res = minimize(lambda p: chi2(build(p), use_slope, extra_sigma_sys), p0,
                   method="Nelder-Mead",
                   options=dict(maxiter=40000, maxfev=40000, xatol=1e-7, fatol=1e-7))
    model = build(res.x)
    npar = len(p0)
    if verbose:
        tag = "saturated  " if saturate else "unsaturated"
        print(f"  {tag} [{profile_shape:8s}]: chi2 = {res.fun:7.2f} / "
              f"{n_data(use_slope) - npar} dof = "
              f"{res.fun/(n_data(use_slope)-npar):5.2f}   "
              f"params = {np.round(np.abs(res.x), 4).tolist()}")
    return model, float(res.fun), npar
