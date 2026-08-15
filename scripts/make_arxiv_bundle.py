"""Assemble a self-contained arXiv upload for one of the manuscripts.

The repository keeps figures in a shared top-level directory, which arXiv cannot
resolve. This copies the source into a flat bundle, rewrites the figure paths,
and includes the .bbl, since arXiv does not run BibTeX.

Usage:  python scripts/make_arxiv_bundle.py lattice
        python scripts/make_arxiv_bundle.py framework
"""

import pathlib
import re
import shutil
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("lattice", "framework"):
        sys.exit(__doc__)
    name = sys.argv[1]
    src = ROOT / "papers" / name
    out = ROOT / "arxiv" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)

    tex = (src / f"{name}.tex").read_text(encoding="utf8")
    used = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for path in used:
        fig = (src / path).resolve()
        if not fig.exists():
            sys.exit(f"missing figure: {fig}")
        shutil.copy2(fig, out / "figures" / fig.name)
    tex = re.sub(r"\{[^}]*figures/([^}]+)\}", r"{figures/\1}", tex)
    (out / f"{name}.tex").write_text(tex, encoding="utf8")

    bbl = src / f"{name}.bbl"
    if not bbl.exists():
        sys.exit(f"no {bbl.name}; run build.sh first (arXiv does not run BibTeX)")
    shutil.copy2(bbl, out / bbl.name)

    tarball = ROOT / "arxiv" / f"{name}-arxiv.tar.gz"
    with tarfile.open(tarball, "w:gz") as t:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                t.add(f, arcname=str(f.relative_to(out)))

    print(f"  bundle: {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(out)}  ({f.stat().st_size/1024:.0f} kB)")
    print(f"  tarball: {tarball}  ({tarball.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
