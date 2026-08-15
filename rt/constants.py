"""CODATA-2018 physical constants and derived vacuum-tension scales (SI unless noted)."""

import math

# --- exact / CODATA-2018 ---
c = 299_792_458.0                 # m/s (exact)
h_planck = 6.626_070_15e-34       # J s (exact)
hbar = h_planck / (2.0 * math.pi)  # J s
G = 6.674_30e-11                  # m^3 kg^-1 s^-2
e_charge = 1.602_176_634e-19      # C (exact)
alpha_fs = 7.297_352_5693e-3      # fine-structure constant
m_e = 9.109_383_7015e-31          # kg
m_mu = 1.883_531_627e-28          # kg
m_p = 1.672_621_923_69e-27        # kg

# --- derived ---
l_planck = math.sqrt(hbar * G / c**3)      # m
m_planck = math.sqrt(hbar * c / G)         # kg
E_planck = m_planck * c**2                 # J

GAMMA = c**4 / (16.0 * math.pi * G)        # N  -- fundamental metric tension
GAMMA_ALT = hbar * c / (16.0 * math.pi * l_planck**2)   # identical, quantum form

# electron scales
lambda_bar_e = hbar / (m_e * c)            # reduced Compton wavelength, m
r_e_classical = alpha_fs * lambda_bar_e    # classical electron radius, m
r_s_e = 2.0 * G * m_e / c**2               # Schwarzschild radius of the electron, m
alpha_G_e = (m_e / m_planck) ** 2          # gravitational coupling of the electron

# unit conversions
MeV = 1.602_176_634e-13                    # J
GeV = 1.0e3 * MeV
fm = 1.0e-15                               # m
mb = 1.0e-31                               # m^2  (1 millibarn)
hbarc_GeVfm = hbar * c / (GeV * fm)        # ~0.1973 GeV fm

if __name__ == "__main__":
    print(f"Gamma            = {GAMMA:.6e} N   (alt {GAMMA_ALT:.6e} N)")
    print(f"l_planck         = {l_planck:.6e} m")
    print(f"m_planck         = {m_planck:.6e} kg")
    print(f"lambda_bar_e     = {lambda_bar_e:.6e} m")
    print(f"r_s(electron)    = {r_s_e:.6e} m")
    print(f"r_s/lambda_bar_e = {r_s_e / lambda_bar_e:.6e}")
    print(f"alpha_G(e)       = {alpha_G_e:.6e}   (2*alpha_G = {2*alpha_G_e:.6e})")
    print(f"hbar*c           = {hbarc_GeVfm:.6f} GeV fm")
