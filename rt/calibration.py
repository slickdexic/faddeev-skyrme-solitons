"""Calibration: mapping the vacuum metric tension to the director-sector couplings.

Structure
---------
The static director energy is  E = Int [ c2 I1 + c4 I2 ] d^3x,  with

    [c2] = force  (energy / length)      -- a tension,
    [c4] = energy * length,

so the model carries exactly one length,

    l_c = sqrt(c4 / c2)      (the Cosserat / rigidity length),

and one energy, sqrt(c2 c4) = c2 l_c.  Any solution therefore obeys

    E_soliton = Ehat * c2 * l_c ,     R_soliton = Rhat * l_c ,

where (Ehat, Rhat) are the dimensionless lattice results for the sector in
question.  Inverting,

    c2 = (E / Ehat) * (Rhat / R) ,    c4 = (E / Ehat) * (R / Rhat) .

Anchoring theorem
-----------------
Define the anchoring coefficient chi = c2 / Gamma with Gamma = c^4 / (16 pi G)
the metric tension.  Substituting c2 above and using r_s = 2 G M / c^2,

    chi = 8 pi (Rhat / Ehat) * (r_s / R) .

The dimensionless ratio between the director stiffness and the metric tension is
therefore fixed, with no free parameters, by the *gravitational compactness* of
the soliton it supports.  Since chi <= 1 (the director sector cannot be stiffer
than the metric it decorates), the framework requires

    R >= 8 pi (Rhat / Ehat) r_s ,

i.e. a maximally metric-anchored defect (chi = 1) sits inside its own horizon and
is a black hole.  The enormous hierarchy between Gamma and particle mass scales
is thus not a tuning: it is the statement that elementary particles are
gravitationally extremely non-compact, quantified.

Vacuum non-linearity
--------------------
Expanding about n = -z with n = (theta_1, theta_2, ...), the quadratic Lagrangian
gives two massless transverse modes, and the quartic term generates the leading
self-interaction

    L_4 = -(c4 / (4 c2^2)) [ (d phi_1)^2 (d phi_2)^2 - (d phi_1 . d phi_2)^2 ] ,

for canonically normalised phi_a = sqrt(2 c2) theta_a.  Its coupling
g4 = c4/(4 c2^2) = Ehat R^3 / (4 Rhat^3 M) (natural units) is directly comparable
with the Euler-Heisenberg four-photon coupling 2 alpha^2 / (45 m_e^4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import (GAMMA, G, c, hbar, l_planck, m_planck, m_e, m_p,
                        alpha_fs, lambda_bar_e, MeV, GeV, fm)

# Euler-Heisenberg four-photon coefficient scale, in units of 1/m_e^4
EH_COEFF = 2.0 * alpha_fs**2 / 45.0


@dataclass
class SolitonShape:
    """Dimensionless lattice results for one topological sector."""
    E_hat: float
    R_hat: float
    label: str = "Q=1"


def couplings(E_joule: float, R_metre: float, shape: SolitonShape):
    """Return dict of c2 [N], c4 [J m], l_c [m], and the anchoring chi = c2/Gamma."""
    c2 = (E_joule / shape.E_hat) * (shape.R_hat / R_metre)
    c4 = (E_joule / shape.E_hat) * (R_metre / shape.R_hat)
    l_c = math.sqrt(c4 / c2)
    return dict(c2=c2, c4=c4, l_c=l_c, chi=c2 / GAMMA)


def anchoring_from_compactness(mass_kg: float, R_metre: float, shape: SolitonShape):
    """chi = 8 pi (Rhat/Ehat) (r_s / R) -- the anchoring theorem, evaluated directly."""
    r_s = 2.0 * G * mass_kg / c**2
    return 8.0 * math.pi * (shape.R_hat / shape.E_hat) * (r_s / R_metre), r_s


def blackhole_radius(mass_kg: float, shape: SolitonShape):
    """Radius at which chi = 1: the maximally anchored defect."""
    r_s = 2.0 * G * mass_kg / c**2
    return 8.0 * math.pi * (shape.R_hat / shape.E_hat) * r_s, r_s


def four_wave_coupling_over_EH(mass_kg: float, R_metre: float, shape: SolitonShape):
    """g4 / (2 alpha^2 / 45 m_e^4), i.e. the induced light-by-light amplitude
    relative to the QED (Euler-Heisenberg) one, for a soliton of mass `mass_kg`
    and core radius `R_metre`."""
    # natural units: express masses in 1/m via  m -> m c / hbar
    inv_len_mass = mass_kg * c / hbar          # [1/m]
    inv_len_me = m_e * c / hbar                # [1/m]
    g4 = shape.E_hat * R_metre**3 / (4.0 * shape.R_hat**3 * inv_len_mass)   # [m^4]
    g4_EH = EH_COEFF / inv_len_me**4                                        # [m^4]
    return g4 / g4_EH


def max_core_radius_from_EH(mass_kg: float, shape: SolitonShape, tolerance=0.1):
    """Largest core radius whose induced light-by-light amplitude stays below
    `tolerance` times the QED one."""
    inv_len_mass = mass_kg * c / hbar
    inv_len_me = m_e * c / hbar
    g4_max = tolerance * EH_COEFF / inv_len_me**4
    return (g4_max * 4.0 * shape.R_hat**3 * inv_len_mass / shape.E_hat) ** (1.0 / 3.0)


def report(shape: SolitonShape):
    lines = [f"Soliton shape constants: Ehat = {shape.E_hat:.2f}, Rhat = {shape.R_hat:.4f}"
             f"   ({shape.label})",
             f"Gamma = c^4/(16 pi G) = {GAMMA:.4e} N",
             f"     = hbar c /(16 pi l_p^2) = {hbar*c/(16*math.pi*l_planck**2):.4e} N",
             ""]

    cases = [
        ("electron, R = reduced Compton", m_e, lambda_bar_e),
        ("electron, R = classical radius", m_e, alpha_fs * lambda_bar_e),
        ("electron, R = LEP bound 1e-19 m", m_e, 1.0e-19),
        ("proton,   R = 0.84 fm", m_p, 0.84 * fm),
    ]
    lines.append(f"{'case':34s} {'c2 [N]':>12s} {'c4 [J m]':>12s} {'l_c [m]':>11s} "
                 f"{'chi=c2/Gamma':>13s} {'chi (thm)':>11s} {'g4/g4_EH':>11s}")
    for label, M, R in cases:
        E = M * c**2
        k = couplings(E, R, shape)
        chi_thm, r_s = anchoring_from_compactness(M, R, shape)
        ratio = four_wave_coupling_over_EH(M, R, shape)
        lines.append(f"{label:34s} {k['c2']:12.4e} {k['c4']:12.4e} {k['l_c']:11.4e} "
                     f"{k['chi']:13.4e} {chi_thm:11.4e} {ratio:11.4e}")

    lines.append("")
    R_bh, r_s = blackhole_radius(m_e, shape)
    lines.append(f"electron: r_s = {r_s:.4e} m ; radius at chi = 1 is "
                 f"{R_bh:.4e} m = {R_bh/r_s:.4f} r_s  -> inside the horizon")
    lines.append(f"electron: largest core radius allowed by a 10% light-by-light test = "
                 f"{max_core_radius_from_EH(m_e, shape):.3e} m "
                 f"({max_core_radius_from_EH(m_e, shape)/fm:.2f} fm)")
    lines.append(f"electron: alpha_G = (m_e/m_Pl)^2 = {(m_e/m_planck)**2:.4e}, "
                 f"r_s/lambdabar = {2*(m_e/m_planck)**2:.4e}")
    return "\n".join(lines)


if __name__ == "__main__":
    # converged Q_H = 1 shape constants from scripts/run_static.py
    print(report(SolitonShape(E_hat=542.646, R_hat=1.705)))
