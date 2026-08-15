#!/usr/bin/env bash
# Build the manuscript. Requires MiKTeX (or TeX Live) on PATH.
set -e
cd "$(dirname "$0")"
export PATH="/c/Users/Paul/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"

NAME=lattice
pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex"
bibtex "$NAME" || true
pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex"
pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex"
echo
echo "built: $(pwd)/$NAME.pdf"
