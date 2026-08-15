"""Rasterise manuscript pages to PNG for visual proofing.

Usage:  python scripts/render_pdf.py [page ...]     (1-based; default: all)
Output: paper/preview/page_NN.png
"""

import pathlib
import sys

import pypdfium2 as pdfium

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF = ROOT / "paper" / "resonant_topology.pdf"
OUT = ROOT / "paper" / "preview"
SCALE = 2.0


def main(pages):
    OUT.mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(str(PDF))
    todo = pages or range(1, len(doc) + 1)
    for p in todo:
        img = doc[p - 1].render(scale=SCALE).to_pil()
        path = OUT / f"page_{p:02d}.png"
        img.save(path)
        print(f"  {path.relative_to(ROOT)}  {img.width}x{img.height}")
    print(f"{len(doc)} pages in {PDF.name}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
