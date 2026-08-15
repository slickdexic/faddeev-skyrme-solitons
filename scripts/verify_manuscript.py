"""Cross-check the numbers quoted in the manuscripts against the stored results.

verify_claims.py re-derives physics from the fields; this checks that what the
LaTeX actually says matches what the drivers actually produced. Written after a
transcription error was found by hand, on the principle that a number typed twice
is a number that can differ.
"""

import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
C0 = 32 * np.pi**2 * np.sqrt(2)
FB = 12 * np.pi**2

ok = True


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def check(label, quoted, computed, tol):
    global ok
    good = abs(quoted - computed) <= tol
    ok &= good
    print(f"  [{'OK ' if good else 'FAIL'}] {label:<52} paper {quoted:>10.4f}  "
          f"code {computed:>10.4f}")


def tex(path):
    return (ROOT / path).read_text(encoding="utf8")


def table(src, label):
    """Body of the table carrying `label`; rows repeat across tables otherwise."""
    m = re.search(r"\\begin\{table\*?\}(?:(?!\\end\{table\*?\}).)*?"
                  r"\\label\{" + re.escape(label) + r"\}"
                  r"((?:(?!\\end\{table\*?\}).)*)", src, re.S)
    return m.group(1) if m else ""


print("=" * 78)
print("Paper A (papers/lattice/lattice.tex)")
print("=" * 78)
a = tex("papers/lattice/lattice.tex")

check("Ward constant c0", 446.65, C0, 0.01)
check("Faddeev-Bogomolny constant 12 pi^2", 118.435, FB, 0.01)

hres = load("hopfion_resolution.json")
if hres:
    t = table(a, "tab:hconv")
    print("  lattice-spacing table:")
    for row in hres["rows"]:
        m = re.search(rf"\n\s*{row['N']} & {row['h']:.4f} & ([0-9.]+) & ([0-9.]+)", t)
        if not m:
            print(f"    [WARN] no table row found for N={row['N']}")
            continue
        check(f"    N={row['N']}: E", float(m.group(1)), row["E"], 0.02)
        check(f"    N={row['N']}: E/c0", float(m.group(2)), row["E_over_c0"], 1e-4)

vol = load("volume_study.json")
if vol:
    quoted = [float(x) for x in
              re.search(r"E=540\.50,\\,([0-9.]+),\\,([0-9.]+)", a).groups()] \
        if re.search(r"E=540\.50,\\,([0-9.]+),\\,([0-9.]+)", a) else []
    for q, row in zip([540.50] + quoted, vol["rows"]):
        check(f"    volume study L={row['L']:.0f}", q, row["E"], 0.02)

o4 = load("order4_check.json")
if o4:
    t = table(a, "tab:o4")
    print("  two-scheme cross-check:")
    for r in o4["rows"]:
        pat = rf"{r['h']:.4f} & ([0-9.]+) &" if r["scheme"] == "o2" \
            else rf"{r['h']:.4f} & [0-9.]+ & [0-9.]+ & ([0-9.]+)"
        m = re.search(pat, t)
        if m:
            check(f"    {r['scheme']} h={r['h']:.4f}", float(m.group(1)),
                  r["E_over_c0"], 5e-4)

# a percentage a reader recomputes from the printed columns must match the
# printed percentage; 1.9795 shown as 1.980 crosses a rounding boundary and does not
t = table(a, "tab:validation")
print("  validation table is self-consistent as printed:")
for m in re.finditer(r"\n(\d) & ([0-9.]+) & ([0-9.]+) & \$([+-][0-9.]+)\\%\$", t):
    q, mine, ref, pct = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
    check(f"    Q={q}: percentage from the printed values",
          pct, round(100 * (mine - ref) / ref, 1), 0.05)

trunc = load("truncation_sign.json")
if trunc:
    print("  truncation-error signs:")
    o2, o4 = trunc["schemes"]["o2"], trunc["schemes"]["o4"]
    check("    o2 quadratic coefficient", -68.8, o2["E2"]["a"], 0.05)
    check("    o2 quartic coefficient", -214.4, o2["E4"]["a"], 0.05)
    check("    o4 quadratic coefficient (1e3)", 2.3, o4["E2"]["a"] / 1e3, 0.05)
    check("    o4 quartic coefficient (1e3)", 8.3, o4["E4"]["a"] / 1e3, 0.05)
    frac = o2["E4"]["a"] / o2["E"]["a"]
    check("    quartic share of the h^2 error", 0.75, frac, 0.02)
    for s, sc in (("o2", o2), ("o4", o4)):
        want = "below" if s == "o2" else "above"
        for term in ("E2", "E4"):
            got = sc[term]["approaches_from"]
            ok &= got == want
            print(f"  [{'OK ' if got == want else 'FAIL'}] "
                  f"    {s} {term} approaches from{'':<20} "
                  f"paper {want:>10}  code {got:>10}")

sky = load("skyrme.json")
if sky and "extrapolation" in sky:
    e = sky["extrapolation"]
    check("frame extrapolation E_inf/12pi^2", 1.2311, e["E_inf"], 5e-4)
    check("exact hedgehog ODE", 1.23145, e["exact"], 1e-5)

bnd = load("boundary_study.json")
if bnd and bnd.get("complete"):
    t = table(a, "tab:boundary")
    print("  paired boundary comparison:")
    per = [r for r in bnd["runs"] if r["bc"] == "periodic"]
    fix = [r for r in bnd["runs"] if r["bc"] == "fixed"]
    for p_, f_ in zip(per, fix):
        m = re.search(rf"\n\s*{p_['L']:.0f} & {p_['h']:.4f} & ([0-9.]+) & ([0-9.]+)"
                      rf" & \$\+([0-9.]+)\$", t)
        if not m:
            print(f"    [WARN] no row for L={p_['L']:.0f} h={p_['h']:.4f}")
            continue
        tag = f"L={p_['L']:.0f} h={p_['h']:.4f}"
        check(f"    {tag} periodic", float(m.group(1)), p_["E_over_bound"], 5e-5)
        check(f"    {tag} fixed", float(m.group(2)), f_["E_over_bound"], 5e-5)
        d = 100 * (f_["E_over_bound"] - p_["E_over_bound"]) / p_["E_over_bound"]
        check(f"    {tag} difference (%)", float(m.group(3)), d, 1e-3)
    check("    E_inf periodic", 1.2280, bnd["fit_periodic"]["E_inf"], 5e-5)
    check("    E_inf fixed", 1.2257, bnd["fit_fixed"]["E_inf"], 5e-5)
    shift = abs(100 * bnd["boundary_shift"] / bnd["fit_periodic"]["E_inf"])
    check("    shift quoted in text (%)", 0.18, shift, 5e-3)
    check("    excess over 1.204 under Dirichlet (%)", 1.81,
          100 * (bnd["fit_fixed"]["E_inf"] - 1.204) / 1.204, 5e-3)

spec = load("skyrme_spectrum.json")
if spec:
    t = table(a, "tab:frame")
    print("  frame-sector spectrum:")
    for B in ("1", "2", "3", "4"):
        r = spec["sectors"][B]
        m = re.search(rf"\n{B} & \w+\s+& ([0-9.]+) &", t)
        if m:
            check(f"    B={B}", float(m.group(1)), r["E_over_B"], 5e-5)
    for B in ("2", "3", "4"):
        r = spec["sectors"][B]
        m = re.search(rf"\n{B} & \w+\s+& [0-9.]+ & [0-9.]+ & \$([0-9.]+)", t)
        if m:
            check(f"    B={B} binding (%)", float(m.group(1)), 100 * r["binding"], 0.02)

stat = load("static_solitons.json")
if stat:
    t = table(a, "tab:spectrum")
    print("  Hopf spectrum:")
    for q in ("1", "2", "3", "4"):
        r = stat["sectors"][q]
        m = re.search(rf"\n{q} & [^&]+& ([0-9.]+)\s+&", t)
        if m:
            check(f"    Q={q}: E", float(m.group(1)), r["E"], 0.02)
    E = np.array([stat["sectors"][q]["E"] for q in "1234"])
    Q = np.arange(1, 5)
    slope = np.polyfit(np.log(Q), np.log(E), 1)[0]
    check("scaling exponent", 0.765, slope, 5e-4)
    for i, q in enumerate("234", start=1):
        bind = 100 * (1 - E[i] / ((i + 1) * E[0]))
        quoted = [18.5, 23.4, 28.0][i - 1]
        check(f"binding vs fission Q={q} (%)", quoted, bind, 0.05)

print()
print("=" * 78)
print("VERDICT:", "manuscript numbers match the results" if ok
      else "MANUSCRIPT DISAGREES WITH THE RESULTS")
print("=" * 78)
sys.exit(0 if ok else 1)
