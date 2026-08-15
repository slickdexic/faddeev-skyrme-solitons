#!/usr/bin/env bash
# Build the manuscript. Requires MiKTeX (or TeX Live) on PATH.
set -e
cd "$(dirname "$0")"

# MiKTeX's default Windows install is not on PATH under Git Bash, and
# LOCALAPPDATA is a Windows path that bash cannot use as a PATH entry.
if ! command -v pdflatex >/dev/null 2>&1 && [ -n "$LOCALAPPDATA" ]; then
  miktex="$LOCALAPPDATA"
  command -v cygpath >/dev/null 2>&1 && miktex="$(cygpath -u "$miktex")"
  export PATH="$miktex/Programs/MiKTeX/miktex/bin/x64:$PATH"
fi
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found; install TeX Live or MiKTeX and put it on PATH" >&2
  exit 1
fi

NAME=framework
pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex"
bibtex "$NAME" || true
pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex"
pdflatex -interaction=nonstopmode -halt-on-error "$NAME.tex"
echo
echo "built: $(pwd)/$NAME.pdf"
