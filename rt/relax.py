"""Energy minimisation and Lorentzian evolution for the Faddeev-Skyrme model.

The lattice is periodic by default and all configurations are vacuum in a collar
around the boundary, so no explicit boundary condition is imposed.  Setting
`fs_core.set_boundary("fixed")` switches to Dirichlet walls, which `minimise`
honours by freezing the collar; this reproduces the convention of the reference
computations and isolates the boundary's contribution to the energy.  For time
evolution an optional absorbing collar removes outgoing radiation, which lets
the radiated energy be measured cleanly.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

from . import fs_core as fs


def _ndimage():
    if fs.on_gpu():
        from cupyx.scipy import ndimage as gpu_ndimage
        return gpu_ndimage
    return ndimage


def rescale(n, mu, order=3):
    """Return the dilated field n_mu(x) = n(x / mu) on the same periodic lattice.

    Dilation is the exact zero mode of Derrick scaling, E(mu) = mu E2 + E4/mu, so
    applying it directly removes the slowest mode of the relaxation.
    """
    xp = fs.xp
    nd = _ndimage()
    N = n.shape[1]
    idx = xp.arange(N)
    c = (N - 1) / 2.0
    src = (idx - c) / mu + c
    coords = xp.stack(xp.meshgrid(src, src, src, indexing="ij"))
    out = xp.stack([
        nd.map_coordinates(n[a], coords, order=order, mode="nearest")
        for a in range(n.shape[0])
    ])
    return fs.normalise(out)


def frozen_mask(shape, width: int = 4):
    """Collar of cells clamped to their initial value under Dirichlet walls.

    The width must exceed the stencil reach (3 for the fourth-order scheme) so
    that no moving site ever reads a replicated edge value.
    """
    xp = fs.xp
    m = xp.zeros(shape[1:], dtype=bool)
    for axis in range(3):
        sl = [slice(None)] * 3
        sl[axis] = slice(0, width)
        m[tuple(sl)] = True
        sl[axis] = slice(-width, None)
        m[tuple(sl)] = True
    return m


def minimise(n, h, c2=1.0, c4=1.0, steps=6000, max_rot=0.01, momentum=0.9,
             report=500, tol=1e-12, verbose=True, track_charge=True,
             rescale_every=150, bc_width=4):
    """Adaptive damped flow towards a stationary point of E[n].

    The step length is capped so that no site rotates by more than `max_rot` per
    iteration; a heavy-ball term with energy arrest accelerates convergence, and
    every `rescale_every` steps the exact Derrick dilation mu = sqrt(E4/E2) is
    applied (accepted only if it lowers the energy).

    Returns (n, history) with rows (step, E2, E4, E2/E4, |grad|_inf, Q_H).
    """
    n = fs.normalise(n.copy())
    vel = fs.xp.zeros_like(n)

    clamp = None
    if fs.boundary() == "fixed":
        clamp = (frozen_mask(n.shape, bc_width), n.copy())

    def hold(f):
        if clamp is not None:
            mask, ref = clamp
            f[:, mask] = ref[:, mask]
        return f

    E2, E4 = fs.energy(n, h, c2, c4)
    E_prev = E2 + E4
    best = (E_prev, n.copy())
    history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        g = fs.variation(n, h, c2, c4)
        if clamp is not None:
            g[:, clamp[0]] = 0.0
        gmax = float(fs.xp.abs(g).max())
        if gmax < 1e-14:
            break
        tau = max_rot / gmax

        vel = momentum * vel - tau * g
        trial = hold(fs.normalise(n + vel))
        e2, e4 = fs.energy(trial, h, c2, c4)
        if e2 + e4 > E_prev:                       # arrest, then pure descent
            vel[...] = 0.0
            trial = hold(fs.normalise(n - tau * g))
            e2, e4 = fs.energy(trial, h, c2, c4)
            if e2 + e4 > E_prev:                   # still uphill: shorten the step
                max_rot *= 0.5
                if max_rot < 1e-8:
                    break
                continue
        n, E2, E4 = trial, e2, e4
        vel -= fs.xp.einsum("axyz,axyz->xyz", vel, n) * n     # keep momentum tangential
        E = E2 + E4
        rel = abs(E_prev - E) / max(E, 1e-300)
        E_prev = E
        if E < best[0]:
            best = (E, n.copy())

        if rescale_every and step % rescale_every == 0 and E4 > 0:
            mu = float(np.sqrt(E2 / E4)) ** -1.0     # mu = sqrt(E4/E2)
            mu = min(max(mu, 0.8), 1.25)
            if abs(mu - 1.0) > 1e-3:
                cand = hold(rescale(n, mu))
                f2, f4 = fs.energy(cand, h, c2, c4)
                # unbounded dilation samples only the core and clamps at the edges,
                # giving a uniform field with E = 0; the energy test alone accepts it
                leaked = fs.boundary_leakage(cand, h, c2=c2, c4=c4) > 0.05
                if f2 + f4 < E and not leaked:
                    n, E2, E4, E = cand, f2, f4, f2 + f4
                    E_prev = E
                    vel[...] = 0.0
                    if E < best[0]:
                        best = (E, n.copy())

        if step % report == 0 or step == steps:
            q = fs.hopf_charge(n, h) if track_charge else np.nan
            history.append((step, E2, E4, E2 / max(E4, 1e-300), gmax, q))
            if verbose:
                print(f"    {step:6d}  E={E:11.4f}  E2/E4={E2/E4:8.5f}  Q={q:+8.5f}  "
                      f"|g|inf={gmax:.3e}  dE/E={rel:.2e}  [{time.time()-t0:6.1f}s]")
        if rel < tol and step > 400:
            if verbose:
                print(f"    converged at step {step}, dE/E = {rel:.2e}")
            break

    n = best[1]
    E2, E4 = fs.energy(n, h, c2, c4)
    q = fs.hopf_charge(n, h) if track_charge else np.nan
    history.append((-1, E2, E4, E2 / max(E4, 1e-300), np.nan, q))
    return n, history


def absorbing_mask(grid, L, width=0.18, strength=1.0):
    """Damping profile: zero in the bulk, rising smoothly in a boundary collar."""
    xp = fs.xp
    X, Y, Z = grid
    r = xp.maximum(xp.maximum(xp.abs(X), xp.abs(Y)), xp.abs(Z)) / (L / 2)
    s = xp.clip((r - (1.0 - width)) / width, 0.0, 1.0)
    return strength * s**2


def evolve(n, h, c2=1.0, c4=1.0, dt=None, steps=1000, v0=None, callback=None,
           every=25, damping=None, verbose=False):
    """Constrained relativistic evolution of the director field.

    Equation of motion (c = 1, kinetic term c2 |n_t|^2; terms quartic in time
    derivatives are omitted, which is accurate at the sub-relativistic soliton
    velocities used here):

        n_tt = -(1/c2) P_n[ dE/dn ] - |n_t|^2 n - gamma(x) n_t ,

    with P_n the projector onto T_n S^2 and gamma an optional absorbing collar.
    Integrated by leapfrog with renormalisation, so |n| = 1 to machine precision.
    """
    n = fs.normalise(n.copy())
    vel = fs.xp.zeros_like(n) if v0 is None else v0.copy()
    if dt is None:
        dt = 0.05 * h
    records = []
    t0 = time.time()

    for step in range(steps + 1):
        if callback is not None and step % every == 0:
            records.append(callback(step, step * dt, n, vel))
            if verbose:
                print(f"      t={step*dt:8.3f}  [{time.time()-t0:6.1f}s]")
        acc = -fs.variation(n, h, c2, c4) / c2
        acc -= fs.xp.einsum("axyz,axyz->xyz", vel, vel) * n
        if damping is not None:
            acc -= damping * vel
        vel += dt * acc
        n = fs.normalise(n + dt * vel)
        vel -= fs.xp.einsum("axyz,axyz->xyz", vel, n) * n

    return n, vel, records


def kinetic_energy(vel, h, c2=1.0):
    return float(c2 * np.einsum("axyz,axyz->", vel, vel) * h**3)
