"""Figures for the manuscript: soliton structure and the topological energy spectrum.

Reads results/static_solitons.json and results/field_Q*.npy.
Outputs figures/hopfion_structure.png and figures/energy_spectrum.png
"""

import json
import pathlib
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BOUND = 32 * np.pi**2 * np.sqrt(2)

# Published minimal energies, Sutcliffe, Proc. R. Soc. A 463 (2007) 3001, Table 3,
# in the normalisation E / (32 pi^2 sqrt 2). The integrand there is identical to
# the one used here, so these are directly comparable.
LITERATURE = {1: (1.204, "A_{1,1}"), 2: (1.967, "A_{2,1}"),
              3: (2.754, "A~_{3,1}"), 4: (3.445, "A_{2,2}")}


def structure_figure(data):
    sectors = sorted(int(k) for k in data["sectors"])
    fields = [(Q, np.load(RES / f"field_Q{Q}.npy")) for Q in sectors
              if (RES / f"field_Q{Q}.npy").exists()]
    if not fields:
        return
    L = data["sectors"][str(fields[0][0])]["L"]
    n_cols = len(fields)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.1 * n_cols, 6.0), squeeze=False)

    for col, (Q, n) in enumerate(fields):
        N = n.shape[1]
        h = L / N
        grid, _ = fs.make_grid(N, L)
        e2, e4 = fs.energy_density(n, h)
        e = e2 + e4
        AB, _B = fs.hopf_density(n, h)
        Qm = fs.hopf_charge(n, h)
        mid = N // 2
        ext = [-L / 2, L / 2, -L / 2, L / 2]

        im0 = axes[0][col].imshow(e[:, :, mid].T, origin="lower", extent=ext,
                                  cmap="inferno")
        axes[0][col].set_title(rf"$Q_H={Q}$   energy density", fontsize=10)
        plt.colorbar(im0, ax=axes[0][col], fraction=0.046)

        # symmetric scale so the vacuum is the neutral colour in every panel,
        # whatever the sign of the relaxed solution's charge
        lim = float(np.abs(AB[:, :, mid]).max())
        im1 = axes[1][col].imshow(AB[:, :, mid].T, origin="lower", extent=ext,
                                  cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[1][col].set_title(rf"$\mathbf{{A}}\cdot\mathbf{{B}}$  "
                               rf"($Q_H={Qm:+.2f}$)", fontsize=10)
        plt.colorbar(im1, ax=axes[1][col], fraction=0.046)
        for r in (0, 1):
            axes[r][col].set_xlabel("x")
            axes[r][col].set_ylabel("y")

    fig.suptitle("Relaxed Hopfions: mid-plane sections", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "hopfion_structure.png", dpi=150)
    plt.close(fig)


def spectrum_figure(data):
    sectors = sorted(int(k) for k in data["sectors"])
    Q = np.array(sectors, dtype=float)
    E = np.array([data["sectors"][str(int(q))]["E"] for q in Q])
    E1 = E[0]

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax[0].plot(Q, E, "o-", color="C3", label=r"lattice $E_Q$")
    ax[0].plot(Q, BOUND * Q**0.75, "k--", label=r"Ward bound $c_0Q^{3/4}$")
    ax[0].plot(Q, E1 * Q, ":", color="C0", label=r"$Q\,E_1$ (fission threshold)")
    ax[0].set_xlabel(r"$Q_H$")
    ax[0].set_ylabel(r"$E$  (units $c_2\ell_c$)")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].grid(alpha=0.25)

    ax[1].plot(Q, E / (E1 * Q), "o-", color="C3")
    ax[1].axhline(1.0, color="k", ls=":")
    ax[1].set_xlabel(r"$Q_H$")
    ax[1].set_ylabel(r"$E_Q / (Q\,E_1)$")
    ax[1].set_title("below 1 = bound against fission", fontsize=10)
    ax[1].grid(alpha=0.25)

    fig.suptitle("Topological energy spectrum of the hyperelastic vacuum", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "energy_spectrum.png", dpi=150)
    plt.close(fig)


def convergence_table(data):
    rows = data.get("resolution_study", [])
    if not rows:
        return ""
    out = ["| $N$ | $h$ | $\\hat E$ | $\\hat E/c_0$ | virial $E_2/E_4$ | $Q_H$ |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['N']} | {r['h']:.4f} | {r['E']:.2f} | "
                   f"{r['E']/BOUND:.4f} | {r['virial']:.4f} | {r['Q']:+.4f} |")
    return "\n".join(out)


def frame_figure():
    """Column density of the B = 1 frame-sector soliton against the two
    assumed opacity shapes of the scattering section, all at unit rms radius."""
    path = RES / "skyrme.json"
    if not path.exists():
        return False
    import matplotlib.pyplot as plt
    from rt import eikonal as ek

    d = json.loads(path.read_text())["profile"]
    u, T = np.asarray(d["u"]), np.asarray(d["T"])
    a, w = np.asarray(d["gauss_a"]), np.asarray(d["gauss_w"])

    x = np.linspace(0.0, 4.2, 400)
    sky = sum(ai * np.exp(-(x**2) / wi**2) for ai, wi in zip(a, w))
    dip = sum(ai * np.exp(-(x**2) / wi**2)
              for ai, wi in zip(ek._DIPOLE_A, ek._DIPOLE_W))

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    m = u <= 4.2
    ax.semilogy(u[m], T[m], "o", ms=3.5, color="C3", label="lattice $B=1$")
    ax.semilogy(x, sky, "-", color="C3", lw=1.2, label="Gaussian-sum fit")
    ax.semilogy(x, np.exp(-(x**2)), "--", color="C0", lw=1.2, label="Gaussian")
    ax.semilogy(x, dip, "-.", color="C2", lw=1.2, label="dipole form factor")
    ax.set_xlabel(r"$b/\langle b^2\rangle^{1/2}$")
    ax.set_ylabel("opacity shape (normalised)")
    ax.set_ylim(1e-5, 2.0)
    ax.set_xlim(0, 4.2)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "frame_profile.png", dpi=150)
    plt.close(fig)
    return True


def main():
    path = RES / "static_solitons.json"
    if not path.exists():
        print("run scripts/run_static.py first")
        return
    data = json.loads(path.read_text())
    structure_figure(data)
    spectrum_figure(data)
    if frame_figure():
        print("frame-sector profile figure written")

    print("== soliton spectrum ==")
    print("| Q | seed | E | E/(c0 Q^3/4) | R_rms | virial | Q measured | E_Q/(Q E_1) |")
    print("|---|---|---|---|---|---|---|---|")
    E1 = data["sectors"]["1"]["E"]
    for k in sorted(data["sectors"], key=int):
        r = data["sectors"][k]
        q = int(k)
        print(f"| {q} | {r['seed']} | {r['E']:.2f} | "
              f"{r['E']/(BOUND*q**0.75):.4f} | {r['R_rms']:.3f} | "
              f"{r['virial']:.4f} | {r['Q']:+.4f} | {r['E']/(E1*q):.4f} |")

    print("\n== convergence study (Q = 1) ==")
    print(convergence_table(data))

    print("\n== comparison with Sutcliffe (2007) ==")
    print("| Q | this work E/c0 | Sutcliffe E/c0 | difference | our ground state | published |")
    print("|---|---|---|---|---|---|")
    for k in sorted(data["sectors"], key=int):
        q = int(k)
        if q not in LITERATURE:
            continue
        ours = data["sectors"][k]["E"] / BOUND
        lit, typ = LITERATURE[q]
        print(f"| {q} | {ours:.4f} | {lit:.3f} | {100*(ours/lit-1):+.1f}% | "
              f"{data['sectors'][k]['seed']} | {typ} |")

    print(f"\nfigures written to {FIG}")


if __name__ == "__main__":
    main()
