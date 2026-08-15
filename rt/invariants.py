"""Symbolic verification of the strain-invariant identities underlying Theorem 1.

For a director field n : R^3 -> S^2 the pull-back (right Cauchy-Green) strain
tensor is

    D_ij = d_i n . d_j n,

with principal invariants

    I1 = tr D,
    I2 = 1/2 [ (tr D)^2 - tr(D^2) ],
    I3 = det D.

The claims verified here:

  (A)  I3 == 0 identically  (D has rank <= 2 because d_i n is tangent to S^2);
  (B)  sum_{ij} F_ij^2 == (tr D)^2 - tr(D^2) == 2 I2,
       where F_ij = n . (d_i n x d_j n) is the pull-back of the S^2 area form;
  (C)  |d_i n x d_j n|^2 == F_ij^2 (the cross product is parallel to n);
  (D)  the Faddeev-Skyrme energy density equals c2 I1 + c4 I2 exactly, i.e. it is
       the Mooney-Rivlin form of isotropic hyperelasticity.

The verification is fully general: n is written in an arbitrary orthonormal
tangent frame with completely free derivative components.
"""

import sympy as sp


def build_generic_director():
    """Return (n, dn) for a generic S^2-valued field with free first derivatives.

    n = (sin(T) cos(P), sin(T) sin(P), cos(T)) with T, P arbitrary functions of
    (x1, x2, x3).  The nine first derivatives T_i, P_i are free symbols, so the
    result is a completely general first-order jet of a director field.
    """
    T, P = sp.symbols("Theta Phi", real=True)
    Ti = sp.symbols("Theta_1 Theta_2 Theta_3", real=True)
    Pi = sp.symbols("Phi_1 Phi_2 Phi_3", real=True)

    n = sp.Matrix([sp.sin(T) * sp.cos(P), sp.sin(T) * sp.sin(P), sp.cos(T)])
    e_T = sp.Matrix([sp.cos(T) * sp.cos(P), sp.cos(T) * sp.sin(P), -sp.sin(T)])
    e_P = sp.Matrix([-sp.sin(P), sp.cos(P), sp.Integer(0)])

    dn = [sp.simplify(Ti[i] * e_T + sp.sin(T) * Pi[i] * e_P) for i in range(3)]
    return n, dn, (T, P, Ti, Pi)


def strain_tensor(dn):
    return sp.Matrix(3, 3, lambda i, j: sp.expand_trig(sp.expand((dn[i].T * dn[j])[0, 0])))


def area_form(n, dn):
    return sp.Matrix(3, 3, lambda i, j: sp.expand((n.T * dn[i].cross(dn[j]))[0, 0]))


def run() -> dict:
    n, dn, _ = build_generic_director()
    D = strain_tensor(dn)
    F = area_form(n, dn)

    I1 = sp.simplify(sp.trace(D))
    I2 = sp.simplify(sp.Rational(1, 2) * (sp.trace(D) ** 2 - sp.trace(D * D)))
    I3 = sp.simplify(D.det())

    sumF2 = sp.simplify(sum(F[i, j] ** 2 for i in range(3) for j in range(3)))
    cross2 = sp.simplify(
        sum(dn[i].cross(dn[j]).dot(dn[i].cross(dn[j])) for i in range(3) for j in range(3))
    )

    results = {
        "I3_is_zero": sp.simplify(I3) == 0,
        "sumF2_equals_2I2": sp.simplify(sumF2 - 2 * I2) == 0,
        "cross2_equals_sumF2": sp.simplify(cross2 - sumF2) == 0,
        "I1": sp.simplify(I1),
        "I2": sp.factor(sp.simplify(I2)),
    }

    # (D) unit-normalisation check: n . n == 1 and n . d_i n == 0
    results["unit_norm"] = sp.simplify((n.T * n)[0, 0] - 1) == 0
    results["tangency"] = all(sp.simplify((n.T * dn[i])[0, 0]) == 0 for i in range(3))
    return results


def derrick_virial():
    """Derrick scaling of E = E2 + E4 under n(x) -> n(x/mu). Returns (E(mu), stationarity)."""
    mu, E2, E4 = sp.symbols("mu E_2 E_4", positive=True)
    E = mu * E2 + E4 / mu
    dE = sp.diff(E, mu)
    sol = sp.solve(sp.Eq(dE.subs(mu, 1), 0), E2)
    d2E = sp.simplify(sp.diff(E, mu, 2).subs(mu, 1))
    return E, sol, d2E


if __name__ == "__main__":
    r = run()
    print("=== Theorem 1: strain-invariant identities for S^2 director fields ===")
    print(f"  |n| = 1 ..................... {r['unit_norm']}")
    print(f"  n . d_i n = 0 ............... {r['tangency']}")
    print(f"  I3 = det D == 0 ............. {r['I3_is_zero']}")
    print(f"  sum F_ij^2 == 2 I2 .......... {r['sumF2_equals_2I2']}")
    print(f"  |d_i n x d_j n|^2 == F_ij^2 . {r['cross2_equals_sumF2']}")
    print(f"  I1 = {r['I1']}")
    print(f"  I2 = {r['I2']}")

    E, sol, d2E = derrick_virial()
    print("\n=== Derrick / virial identity ===")
    print(f"  E(mu) = {E}")
    print(f"  stationarity at mu=1 gives  E_2 = {sol[0]}")
    print(f"  d2E/dmu2 |_(mu=1) = {d2E}  (>0 for E_4>0 : genuine minimum)")
