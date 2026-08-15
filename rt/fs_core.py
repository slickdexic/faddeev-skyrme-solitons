"""Core lattice machinery for the Faddeev-Skyrme (FS) director model.

Static energy functional (code units):

    E[n] = Int d^3x [ c2 * I1 + c4 * I2 ],
    I1  = d_i n . d_i n,
    I2  = 1/2 [ I1^2 - (d_i n . d_j n)(d_i n . d_j n) ] = 1/2 F_ij F_ij,
    F_ij = n . (d_i n x d_j n).

Discretisation.  Every configuration used here equals the constant vacuum in a
collar around the boundary, so the periodic extension of the field is smooth.

Central differences must *not* be used: their Fourier symbol vanishes at the
Nyquist wavenumber, so a checkerboard mode carries zero energy and the discrete
configuration can unwind at no cost, destroying topological protection.  We
therefore build the energy from one-sided differences,

    (D+ f)_i = (f_{i+1} - f_i)/h ,   (D- f)_i = (f_i - f_{i-1})/h ,

whose symbols vanish only at k = 0, and symmetrise the *energy*:

    E = 1/2 [ E(D+ n) + E(D- n) ] ,

which is second-order accurate and has no null mode.  Because (D+)^T = -D- and
(D-)^T = -D+ exactly on a periodic lattice, the returned variation is the exact
gradient of the discrete energy, not merely a discretisation of the continuum
one.  Central fourth-order differences are retained for the topological charge,
which is only a diagnostic.

Field layout: array `n` of shape (3, N, N, N); axis 0 = target component,
axes 1..3 = x, y, z.
"""

from __future__ import annotations

import numpy as np

xp = np                      # array backend; swapped wholesale by use_gpu()


def use_gpu(enable: bool = True) -> bool:
    """Switch every lattice kernel between NumPy and CuPy.

    The kernels are memory-bandwidth bound, so double precision is kept
    throughout: the GPU reproduces the CPU result to rounding.
    """
    global xp
    if not enable:
        xp = np
        return False
    import cupy
    xp = cupy
    return True


def on_gpu() -> bool:
    return xp is not np


def to_numpy(a):
    return xp.asnumpy(a) if on_gpu() else np.asarray(a)


def to_backend(a):
    return xp.asarray(a)


VACUUM = np.array([0.0, 0.0, -1.0])

# fourth-order antisymmetric central-difference weights for f(i-2..i+2)
_W = (1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0)

_BC = "periodic"


def set_boundary(kind: str) -> str:
    """Select 'periodic' or 'fixed' (Dirichlet) lattice boundaries.

    Under 'fixed' the stencils stop at the wall instead of wrapping, so the
    field is continued by the boundary value rather than by its periodic image.
    Relaxation must additionally freeze the collar (see relax.minimise); with a
    vacuum collar the two conventions differ only through the soliton tail.
    """
    global _BC
    if kind not in ("periodic", "fixed"):
        raise ValueError(kind)
    _BC = kind
    return _BC


def boundary() -> str:
    return _BC


def _shift(f: np.ndarray, s: int, axis: int) -> np.ndarray:
    """xp.roll, except that fixed boundaries replicate the edge instead of wrapping.

    The collar is held at the vacuum, so edge replication *is* vacuum padding,
    without the operators needing to know the target-space vacuum (the frame
    sector carries four components, the director sector three).
    """
    out = xp.roll(f, s, axis=axis)
    if _BC == "periodic":
        return out
    n = out.shape[axis]
    dead = [slice(None)] * out.ndim
    edge = [slice(None)] * out.ndim
    if s > 0:
        dead[axis], edge[axis] = slice(0, s), slice(s, s + 1)
    else:
        dead[axis], edge[axis] = slice(n + s, None), slice(n + s - 1, n + s)
    out[tuple(dead)] = out[tuple(edge)]
    return out


# --------------------------------------------------------------------------
# differential operators
# --------------------------------------------------------------------------
def d1(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    """Fourth-order central derivative along a spatial `axis` (1..3)."""
    w2, w1, wm1, wm2 = _W
    return (w2 * _shift(f, 2, axis)
            + w1 * _shift(f, 1, axis)
            + wm1 * _shift(f, -1, axis)
            + wm2 * _shift(f, -2, axis)) / h


def dplus(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    return (_shift(f, -1, axis) - f) / h


def dminus(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    return (f - _shift(f, 1, axis)) / h


def dplus3(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    """Third-order one-sided forward difference, (D+)^T = -D-."""
    return (-11.0 * f + 18.0 * _shift(f, -1, axis)
            - 9.0 * _shift(f, -2, axis)
            + 2.0 * _shift(f, -3, axis)) / (6.0 * h)


def dminus3(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    return (11.0 * f - 18.0 * _shift(f, 1, axis)
            + 9.0 * _shift(f, 2, axis)
            - 2.0 * _shift(f, 3, axis)) / (6.0 * h)


_SCHEME = "o2"


def set_scheme(name: str) -> str:
    """Select the symmetrised difference pair.

    'o2' uses first-order one-sided differences, whose symmetrised energy is
    accurate to O(h^2); 'o4' uses third-order one-sided differences, whose
    leading errors are equal and opposite so the symmetrised energy is accurate
    to O(h^4). Both have a non-vanishing symbol at the Nyquist frequency and
    both satisfy (D+)^T = -D-, so the analytic variation stays exact.
    """
    global _SCHEME
    if name not in ("o2", "o4"):
        raise ValueError(name)
    _SCHEME = name
    return _SCHEME


def _pair():
    return (dplus, dminus) if _SCHEME == "o2" else (dplus3, dminus3)


def dfield(n: np.ndarray, h: float, kind: str = "central") -> np.ndarray:
    """dn[i] = d_i n, shape (3, 3, N, N, N) with dn[i, a] = d_i n_a."""
    fwd, bwd = _pair()
    op = {"central": d1, "plus": fwd, "minus": bwd}[kind]
    return xp.stack([op(n, h, axis=i + 1) for i in range(3)])


def metric(dn: np.ndarray) -> np.ndarray:
    """Strain tensor g[i, j] = d_i n . d_j n, shape (3, 3, N, N, N)."""
    return xp.einsum("iaxyz,jaxyz->ijxyz", dn, dn)


def invariants(g: np.ndarray):
    """Principal invariants (I1, I2) of the strain tensor (I3 = det g == 0)."""
    I1 = g[0, 0] + g[1, 1] + g[2, 2]
    trg2 = xp.einsum("ijxyz,ijxyz->xyz", g, g)
    return I1, 0.5 * (I1 * I1 - trg2)


def _density_one(n, h, kind, c2, c4):
    I1, I2 = invariants(metric(dfield(n, h, kind)))
    return c2 * I1, c4 * I2


def energy_density(n: np.ndarray, h: float, c2: float = 1.0, c4: float = 1.0):
    """Symmetrised (forward/backward averaged) energy density (e2, e4)."""
    a2, a4 = _density_one(n, h, "plus", c2, c4)
    b2, b4 = _density_one(n, h, "minus", c2, c4)
    return 0.5 * (a2 + b2), 0.5 * (a4 + b4)


def energy(n: np.ndarray, h: float, c2: float = 1.0, c4: float = 1.0):
    """Return (E2, E4); the total static energy is E2 + E4."""
    e2, e4 = energy_density(n, h, c2, c4)
    dv = h**3
    return float(e2.sum() * dv), float(e4.sum() * dv)


def _momentum(n, h, kind, c2, c4):
    """P_l = de/d(D_l n) = 2 c2 D_l n + 2 c4 [ I1 D_l n - sum_k g_lk D_k n ]."""
    dn = dfield(n, h, kind)
    g = metric(dn)
    I1 = g[0, 0] + g[1, 1] + g[2, 2]
    P = xp.empty_like(dn)
    for l in range(3):
        acc = I1 * dn[l]
        for k in range(3):
            acc -= g[l, k] * dn[k]
        P[l] = 2.0 * c2 * dn[l] + 2.0 * c4 * acc
    return P


def variation(n: np.ndarray, h: float, c2: float = 1.0, c4: float = 1.0) -> np.ndarray:
    """Exact tangential gradient of the discrete energy (per unit volume).

    dE/dn = -1/2 [ sum_l D-_l P+_l + sum_l D+_l P-_l ],  using (D+)^T = -D-.
    """
    Pp = _momentum(n, h, "plus", c2, c4)
    Pm = _momentum(n, h, "minus", c2, c4)

    fwd, bwd = _pair()
    var = xp.zeros_like(n)
    for l in range(3):
        var -= 0.5 * (bwd(Pp[l], h, axis=l + 1) + fwd(Pm[l], h, axis=l + 1))

    var -= xp.einsum("axyz,axyz->xyz", var, n) * n     # project onto T_n S^2
    return var


# --------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------
def magnetic_field(n: np.ndarray, h: float) -> np.ndarray:
    """B_k = 1/2 eps_kij F_ij with F_ij = n . (d_i n x d_j n)."""
    dn = dfield(n, h)

    def F(i, j):
        return xp.einsum("axyz,axyz->xyz", n, xp.cross(dn[i], dn[j], axis=0))

    return xp.stack([F(1, 2), F(2, 0), F(0, 1)])


def _wavevectors(N: int, h: float):
    k = 2.0 * np.pi * xp.fft.fftfreq(N, d=h)
    KX, KY, KZ = xp.meshgrid(k, k, k, indexing="ij")
    K = xp.stack([KX, KY, KZ])
    K2 = (K * K).sum(axis=0)
    K2[0, 0, 0] = 1.0
    return K, K2


def hopf_density(n: np.ndarray, h: float):
    """Return (A . B, B) with curl A = B in Coulomb gauge."""
    B = magnetic_field(n, h)
    N = n.shape[1]
    K, K2 = _wavevectors(N, h)
    Bk = xp.fft.fftn(B, axes=(1, 2, 3))
    Ak = 1j * xp.cross(K, Bk, axis=0) / K2
    Ak[:, 0, 0, 0] = 0.0
    A = xp.fft.ifftn(Ak, axes=(1, 2, 3)).real
    return xp.einsum("axyz,axyz->xyz", A, B), B


def hopf_charge(n: np.ndarray, h: float) -> float:
    """Q_H = 1/(16 pi^2) Int A . B d^3x."""
    AB, _ = hopf_density(n, h)
    return float(AB.sum() * h**3 / (16.0 * np.pi**2))


def curl_residual(n: np.ndarray, h: float) -> float:
    """Diagnostic: relative || curl A - B || of the gauge-potential reconstruction."""
    B = magnetic_field(n, h)
    N = n.shape[1]
    K, K2 = _wavevectors(N, h)
    Bk = xp.fft.fftn(B, axes=(1, 2, 3))
    Ak = 1j * xp.cross(K, Bk, axis=0) / K2
    Ak[:, 0, 0, 0] = 0.0
    curlA = xp.fft.ifftn(1j * xp.cross(K, Ak, axis=0), axes=(1, 2, 3)).real
    return float(xp.linalg.norm(curlA - B) / max(float(xp.linalg.norm(B)), 1e-300))


# --------------------------------------------------------------------------
# diagnostics and helpers
# --------------------------------------------------------------------------
def energy_radius(n: np.ndarray, h: float, grid, c2: float = 1.0, c4: float = 1.0):
    """RMS energy radius about the energy centroid; returns (R_rms, centroid)."""
    X, Y, Z = grid
    e2, e4 = energy_density(n, h, c2, c4)
    e = e2 + e4
    tot = e.sum()
    cx, cy, cz = (X * e).sum() / tot, (Y * e).sum() / tot, (Z * e).sum() / tot
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return float(xp.sqrt((r2 * e).sum() / tot)), (float(cx), float(cy), float(cz))


def boundary_leakage(n: np.ndarray, h: float, width: int = 6, c2=1.0, c4=1.0) -> float:
    """Fraction of the total energy sitting in a collar of `width` cells."""
    e2, e4 = energy_density(n, h, c2, c4)
    e = e2 + e4
    inner = e[width:-width, width:-width, width:-width].sum()
    return float(1.0 - inner / e.sum())


def make_grid(N: int, L: float):
    """Periodic grid of N points spanning [-L/2, L/2) with spacing h = L / N."""
    x = -L / 2 + L * xp.arange(N) / N
    h = L / N
    X, Y, Z = xp.meshgrid(x, x, x, indexing="ij")
    return (X, Y, Z), h


def normalise(n: np.ndarray) -> np.ndarray:
    return n / xp.sqrt(xp.einsum("axyz,axyz->xyz", n, n))[None, ...]


def gradient_check(n: np.ndarray, h: float, c2=1.0, c4=1.0, npts=12, eps=1e-4, seed=0):
    """Max relative discrepancy between the analytic gradient and finite differences."""
    rng = np.random.default_rng(seed)
    var = variation(n, h, c2, c4)
    N = n.shape[1]
    ncomp = n.shape[0]
    worst = 0.0
    for _ in range(npts):
        idx = tuple(rng.integers(4, N - 4, size=3))
        key = (slice(None),) + idx
        nv = to_numpy(n[key])
        t = rng.normal(size=ncomp)
        t -= np.dot(t, nv) * nv
        t /= np.linalg.norm(t)
        analytic = np.dot(to_numpy(var[key]), t) * h**3

        def density(sign):
            m = n.copy()
            w = nv + sign * eps * t
            m[key] = to_backend(w / np.linalg.norm(w))
            e2, e4 = energy_density(m, h, c2, c4)
            return e2 + e4

        numeric = float((density(+1) - density(-1)).sum()) * h**3 / (2 * eps)
        scale = max(abs(numeric), abs(analytic), 1e-30)
        worst = max(worst, abs(numeric - analytic) / scale)
    return worst
