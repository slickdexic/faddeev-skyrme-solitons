"""Sign of the leading discretisation error, quadratic and quartic terms separately.

The symmetrised energy expands as  Int F - (h^2/24) Int F''[v,v] + O(h^4)  with
v = d^2 n, so the error is negative only where F is convex in dn.  That holds
term by term for the quadratic density, where F'' = 2*identity; for the quartic
density F'' carries a sign-indefinite piece, so the sign of its h^2 error is a
question for measurement rather than algebra.

A fixed analytic configuration is sampled at a sequence of spacings, which
isolates the discretisation error: no relaxation intervenes, so the continuum
value being approached is the same for every grid.

Outputs
-------
results/truncation_sign.json
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import ansatz, fs_core as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"

L = 12.0
NS = [64, 80, 96, 112, 128, 160, 192]


def measure(N, scheme):
    fs.set_scheme(scheme)
    grid, h = fs.make_grid(N, L)
    n = fs.normalise(ansatz.hedgehog_hopf(grid, degree=1, R=4.5))
    e2, e4 = fs.energy(n, h)
    return h, e2, e4


def richardson(hs, vals, p=2):
    """Fit v(h) = v0 + a h^p and return (v0, a)."""
    A = np.vstack([np.ones_like(hs), hs**p]).T
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return coef[0], coef[1]


def main():
    out = {"L": L, "ansatz": "hedgehog_hopf(degree=1, R=4.5)", "schemes": {}}

    for scheme in ("o2", "o4"):
        rows = [measure(N, scheme) for N in NS]
        hs = np.array([r[0] for r in rows])
        e2 = np.array([r[1] for r in rows])
        e4 = np.array([r[2] for r in rows])
        p = 2 if scheme == "o2" else 4

        rec = {"h": hs.tolist(), "E2": e2.tolist(), "E4": e4.tolist(), "power": p}
        print(f"== scheme {scheme}, fitting v0 + a h^{p} ==")
        for name, v in (("E2", e2), ("E4", e4), ("E", e2 + e4)):
            v0, a = richardson(hs, v, p)
            resid = float(np.abs(v - (v0 + a * hs**p)).max())
            sign = "above" if a > 0 else "below"
            rec[name] = {"v0": v0, "a": a, "max_resid": resid, "approaches_from": sign}
            print(f"  {name}: continuum {v0:11.4f}   coefficient a = {a:+10.4f}   "
                  f"finite h lies {sign:5s}   max resid {resid:.2e}")
        out["schemes"][scheme] = rec
        print()

    a2 = out["schemes"]["o2"]["E4"]["a"]
    print("quartic term at second order: finite spacing lies "
          f"{'above' if a2 > 0 else 'below'} the continuum value, "
          f"a = {a2:+.4f}")
    (RES / "truncation_sign.json").write_text(json.dumps(out, indent=2))
    print("wrote results/truncation_sign.json")


if __name__ == "__main__":
    main()
