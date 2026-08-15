"""Frame sector: SU(2)-valued Skyrme field, baryon number, and the B=1 soliton.

The order parameter is a unit four-vector phi : R^3 -> S^3 = SU(2). The strain
tensor D_ij = d_i phi . d_j phi and its invariants are defined exactly as in
`fs_core`, so the energy and its gradient are computed by the same routines --
they never reference the target dimension. Only the topological charge differs.

Structural point (Theorem 1', see the manuscript): for phi : R^3 -> S^n the
Jacobian is (n+1) x 3, so rank D <= min(n, 3) and

    n = 2 (director) : I_3 = det D == 0 identically  -> Mooney-Rivlin is exact,
    n = 3 (frame)    : I_3 != 0                     -> a sextic term is allowed.

The Faddeev-Bogomolny bound follows from AM-GM on the principal stretches,
I_1 >= 3 (l1 l2 l3)^{2/3} and I_2 >= 3 (l1 l2 l3)^{4/3}, giving

    E = Int [a2 I1 + a4 I2] >= 12 pi^2 sqrt(a2 a4) |B| ,

i.e. E >= 12 pi^2 |B| in the units a2 = a4 = 1 used here.
"""

from __future__ import annotations

import numpy as np

from . import fs_core as fs

FB_BOUND = 12.0 * np.pi**2          # Faddeev-Bogomolny constant, a2 = a4 = 1
VACUUM4 = np.array([1.0, 0.0, 0.0, 0.0])


def baryon_number(phi: np.ndarray, h: float) -> float:
    """B = (1/2 pi^2) Int det[phi, d_1 phi, d_2 phi, d_3 phi] d^3x.

    Obtained by contracting the pull-back of the S^3 volume form (total volume
    2 pi^2) with the spatial Levi-Civita symbol.
    """
    xp = fs.xp
    d = fs.dfield(phi, h, kind="central")           # (3, 4, N, N, N)
    N = phi.shape[1]
    M = xp.empty((N, N, N, 4, 4))
    M[..., 0] = xp.moveaxis(phi, 0, -1)
    for i in range(3):
        M[..., i + 1] = xp.moveaxis(d[i], 0, -1)
    return float(xp.linalg.det(M).sum() * h**3 / (2.0 * np.pi**2))


def baryon_density(phi: np.ndarray, h: float) -> np.ndarray:
    """Local baryon density, normalised so that its integral is B."""
    xp = fs.xp
    d = fs.dfield(phi, h, kind="central")
    N = phi.shape[1]
    M = xp.empty((N, N, N, 4, 4))
    M[..., 0] = xp.moveaxis(phi, 0, -1)
    for i in range(3):
        M[..., i + 1] = xp.moveaxis(d[i], 0, -1)
    return xp.linalg.det(M) / (2.0 * np.pi**2)


def hedgehog(grid, degree: int = 1, R: float = 3.0, centre=(0.0, 0.0, 0.0)):
    """Skyrme hedgehog phi = (cos f, sin f * r_hat) with f(0)=pi, f(R)=0; B = degree."""
    X, Y, Z = (fs.to_numpy(g) for g in grid)
    x, y, z = X - centre[0], Y - centre[1], Z - centre[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r < 1e-12, 1e-12, r)

    u = np.clip(r / R, 0.0, 1.0)
    f = np.pi * (1.0 - u) ** 2
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi_a = np.arctan2(y, x)

    s = np.sin(f)
    out = np.stack([
        np.cos(f),
        s * np.sin(theta) * np.cos(degree * phi_a),
        s * np.sin(theta) * np.sin(degree * phi_a),
        s * np.cos(theta),
    ])
    return fs.normalise(fs.to_backend(out))


_S3 = np.sqrt(3.0)

# Degree-B maps as homogeneous polynomial pairs (p, q) in (Z1, Z0), so that
# R = p/q; this stays regular at the poles of R and at spatial infinity.
RATIONAL_MAPS = {
    1: (lambda a, b: (a, b), "spherical"),
    2: (lambda a, b: (a**2, b**2), "axial"),
    3: (lambda a, b: (_S3 * 1j * a**2 * b - b**3, a**3 - _S3 * 1j * a * b**2),
        "tetrahedral"),
    4: (lambda a, b: (a**4 + 2 * _S3 * 1j * a**2 * b**2 + b**4,
                      a**4 - 2 * _S3 * 1j * a**2 * b**2 + b**4), "cubic"),
}


def rational_map(grid, degree: int = 1, R: float = 3.0, centre=(0.0, 0.0, 0.0)):
    """Seed of baryon number `degree` from the symmetric rational-map ansatz.

    For B >= 3 the axially symmetric hedgehog is not the minimum but a saddle,
    so seeds must carry the right point symmetry: the standard degree-B maps of
    Houghton, Manton and Sutcliffe give tetrahedral (B=3) and cubic (B=4)
    configurations. With (Z1, Z0) the homogeneous Riemann-sphere coordinates of
    the spatial direction and R = p/q the map, the field is
    phi = (cos f, sin f * n_R) with

        n_R = (2 Re(p q*), 2 Im(p q*), |q|^2 - |p|^2) / (|p|^2 + |q|^2),

    which never requires dividing by q. Degree one reduces to the hedgehog.
    """
    poly, _sym = RATIONAL_MAPS[degree]
    X, Y, Z = (fs.to_numpy(g) for g in grid)
    x, y, z = X - centre[0], Y - centre[1], Z - centre[2]
    r = np.sqrt(x * x + y * y + z * z)

    Z1, Z0 = x + 1j * y, (r + z).astype(complex)
    nrm = np.sqrt(np.abs(Z1) ** 2 + np.abs(Z0) ** 2)
    bad = nrm < 1e-12                              # the ray z = -r, where R = infinity
    Z1 = np.where(bad, 1.0, Z1 / np.where(bad, 1.0, nrm))
    Z0 = np.where(bad, 0.0, Z0 / np.where(bad, 1.0, nrm))

    p, q = poly(Z1, Z0)
    d = np.abs(p) ** 2 + np.abs(q) ** 2
    d = np.where(d < 1e-300, 1.0, d)
    pq = p * np.conj(q)
    n1, n2 = 2.0 * pq.real / d, 2.0 * pq.imag / d
    n3 = (np.abs(q) ** 2 - np.abs(p) ** 2) / d

    f = np.pi * (1.0 - np.clip(r / R, 0.0, 1.0)) ** 2
    s = np.sin(f)
    out = np.stack([np.cos(f), s * n1, s * n2, s * n3])
    return fs.normalise(fs.to_backend(out))


def hedgehog_ode_energy(rmax: float = 60.0, nodes: int = 4000, tol: float = 1e-10):
    """Exact B = 1 energy from the spherically symmetric profile equation.

    For phi = (cos f, sin f * r_hat) the strain eigenvalues are f' (radial) and
    sin f / r (twice), so

        E = 4 pi Int [ f'^2 r^2 + 2 sin^2 f (1 + f'^2) + sin^4 f / r^2 ] dr ,

    whose Euler-Lagrange equation is solved here as a two-point boundary value
    problem with f(0) = pi, f(inf) = 0. This fixes the normalisation of the
    lattice computation independently of any published number: it returns
    E / 12 pi^2 = 1.2315, against the literature value 1.232.
    """
    from scipy.integrate import solve_bvp, simpson

    def rhs(r, y):
        f, fp = y
        s, c = np.sin(f), np.cos(f)
        return np.vstack([fp, (-2 * r * fp - 2 * s * c * fp**2 + 2 * s * c
                               + 2 * s**3 * c / r**2) / (r**2 + 2 * s**2)])

    def bc(ya, yb):
        return np.array([ya[0] - np.pi, yb[0]])

    r = np.geomspace(1e-3, rmax, nodes)
    g = np.pi * np.exp(-(r / 1.2) ** 2)
    sol = solve_bvp(rhs, bc, r, np.vstack([g, -2 * r / 1.2**2 * g]),
                    tol=tol, max_nodes=400000)
    rr = np.geomspace(1e-3, rmax, 200000)
    f, fp = sol.sol(rr)
    s2 = np.sin(f) ** 2
    dens = fp**2 * rr**2 + 2 * s2 * (1 + fp**2) + s2**2 / rr**2
    return float(4 * np.pi * simpson(dens, x=rr))


def _weight(phi, h, weight):
    if weight == "baryon":
        return fs.xp.abs(baryon_density(phi, h))
    e2, e4 = fs.energy_density(phi, h)
    return e2 + e4


def column_density(phi: np.ndarray, h: float, L: float, weight: str = "baryon"):
    """Column density C(b) sampled at every transverse lattice site.

    The integral along the beam axis is done exactly by summing over z, so there
    is no binning anywhere; the azimuthal average is left implicit and is
    performed by whatever fit consumes the samples. Returns (b, C, b_rms) with b
    measured from the transverse centroid.
    """
    N = phi.shape[1]
    (X, Y, _Z), _ = fs.make_grid(N, L)
    w = _weight(phi, h, weight)
    C = w.sum(axis=2) * h                      # (N, N) column density
    X2, Y2 = fs.to_numpy(X[:, :, 0]), fs.to_numpy(Y[:, :, 0])
    C = fs.to_numpy(C)

    tot = C.sum()
    cx, cy = (X2 * C).sum() / tot, (Y2 * C).sum() / tot
    b = np.sqrt((X2 - cx) ** 2 + (Y2 - cy) ** 2)
    b_rms = float(np.sqrt((b**2 * C).sum() / tot))
    return b.ravel(), C.ravel() / C.max(), b_rms


def transverse_profile(phi: np.ndarray, h: float, L: float, nbins: int = 0,
                       weight: str = "baryon"):
    """Azimuthally averaged column density, for reporting and plotting.

    Bins are uniform in b with width fixed to the lattice spacing unless
    `nbins` is given, which keeps every bin populated. Returns
    (b/b_rms, C/C(0), b_rms).
    """
    b, C, b_rms = column_density(phi, h, L, weight)
    if nbins <= 0:
        nbins = max(int(L / (2.0 * h)), 8)
    edges = np.linspace(0.0, L / 2, nbins + 1)
    idx = np.clip(np.digitize(b, edges) - 1, 0, nbins - 1)
    cnt = np.bincount(idx, minlength=nbins)
    tot = np.bincount(idx, weights=C, minlength=nbins)
    keep = cnt > 0
    centres = 0.5 * (edges[1:] + edges[:-1])[keep]
    T = (tot[keep] / cnt[keep])
    return centres / b_rms, T / T[0], b_rms

