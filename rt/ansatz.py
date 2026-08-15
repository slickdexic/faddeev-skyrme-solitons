"""Initial configurations carrying prescribed Hopf charge.

Two constructions are provided.

1. `hedgehog_hopf` - compose a degree-d map R^3 u {inf} = S^3 -> S^3 (the Skyrme
   hedgehog) with the Hopf projection S^3 -> S^2.  Since the Hopf invariant is
   the integral of the Chern-Simons 3-form A ^ dA, pulling it back through a
   degree-d map multiplies it by d, so Q_H = d exactly.

2. `torus_anm` - the A_{n,m} toroidal ansatz, whose Hopf invariant is n*m.  It
   provides independent seeds (different link types) for the same charges.

Both return arrays of shape (3, N, N, N) normalised to the unit sphere, equal to
(0, 0, -1) outside a compact core.
"""

from __future__ import annotations

import numpy as np

from . import fs_core as fs


def _hopf_project(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Hopf map S^3 -> S^2 for (Z, W) in C^2 with |Z|^2 + |W|^2 = 1."""
    ZW = Z * np.conj(W)
    n = np.stack([2.0 * ZW.real, 2.0 * ZW.imag, np.abs(Z) ** 2 - np.abs(W) ** 2])
    return n / np.sqrt((n * n).sum(axis=0))[None, ...]


def hedgehog_hopf(grid, degree: int = 1, R: float = 3.0, centre=(0.0, 0.0, 0.0)):
    """Hopf projection of a degree-`degree` hedgehog.  Q_H = degree, vacuum = -z."""
    X, Y, Z = (fs.to_numpy(g) for g in grid)
    x, y, z = X - centre[0], Y - centre[1], Z - centre[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r < 1e-12, 1e-12, r)

    u = np.clip(r / R, 0.0, 1.0)
    f = np.pi * (1.0 - u) ** 2                       # f(0) = pi, f(R) = 0, f'(R) = 0
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)

    # degree-`degree` map S^3 -> S^3
    n0 = np.cos(f)
    s = np.sin(f)
    n1 = s * np.sin(theta) * np.cos(degree * phi)
    n2 = s * np.sin(theta) * np.sin(degree * phi)
    n3 = s * np.cos(theta)

    return fs.to_backend(_hopf_project(n2 + 1j * n1, n0 + 1j * n3))


def torus_anm(grid, n_wind: int = 1, m_wind: int = 1, a: float = 2.0, eta0: float = 1.6,
              centre=(0.0, 0.0, 0.0)):
    """A_{n,m} toroidal seed with Hopf charge n_wind * m_wind.

    `a` is the radius of the core ring, `eta0` the compact support radius of the
    meridian profile (must satisfy eta0 < a so that the z axis stays at vacuum).
    """
    X, Y, Z = (fs.to_numpy(g) for g in grid)
    x, y, z = X - centre[0], Y - centre[1], Z - centre[2]
    rho = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    s = rho - a
    eta = np.sqrt(s * s + z * z)
    psi = np.arctan2(z, s)

    u = np.clip(eta / eta0, 0.0, 1.0)
    Theta = 0.5 * np.pi * (1.0 - u) ** 2             # pi/2 on the ring, 0 outside

    w1 = np.sin(Theta) * np.exp(1j * n_wind * phi)
    w2 = np.cos(Theta) * np.exp(1j * m_wind * psi)
    return fs.to_backend(_hopf_project(w1, w2))


def conjugate(n: np.ndarray) -> np.ndarray:
    """Spatial reflection z -> -z: reverses the orientation of the domain, Q_H -> -Q_H.

    (A reflection of the *target* S^2 leaves Q_H invariant, because the Hopf
    invariant is quadratic in the pull-back of the area form.)
    """
    return n[:, :, :, ::-1].copy()


def superpose(fields, grid, vac=(0.0, 0.0, -1.0)) -> np.ndarray:
    """Combine well-separated configurations by nearest-deviation selection.

    Each field is assumed to equal the vacuum outside a compact core; the
    combined field takes, at every point, the configuration that deviates most
    from the vacuum.  Exact for non-overlapping cores.
    """
    v = np.array(vac)[:, None, None, None]
    hosts = [fs.to_numpy(f) for f in fields]
    dev = [np.einsum("axyz,axyz->xyz", f - v, f - v) for f in hosts]
    pick = np.argmax(np.stack(dev), axis=0)
    out = np.broadcast_to(v, hosts[0].shape).copy()
    for i, f in enumerate(hosts):
        mask = pick == i
        out[:, mask] = f[:, mask]
    return fs.to_backend(out)
