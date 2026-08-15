"""Regenerate results/skyrme_field_B1.npy only.

The B = 1 frame-sector field is consumed by verify_claims.py, by the profile
figure and by the eikonal's computed opacity shape, but is not itself recorded in
results/skyrme.json. This reruns exactly the B = 1, L = 12 ladder of
run_skyrme.py so the stored field matches the recorded sector, without repeating
the box study or the B = 2 sector.
"""

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rt import fs_core as fs, skyrme as sk  # noqa: E402
from run_skyrme import ladder  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"


def main():
    if "--gpu" in sys.argv:
        fs.use_gpu()
        print("  lattice backend: GPU (float64)")
    target = json.loads((RES / "skyrme.json").read_text())["sectors"]["1"]
    t0 = time.time()
    phi, h, recs = ladder(1, target["L"], [64, 80, 96], 1500, t0)

    got = recs[-1]["E_over_bound"]
    print(f"\n  recorded E/(12 pi^2) = {target['E_over_bound']:.10f}")
    print(f"  regenerated          = {got:.10f}   (rel. {abs(got/target['E_over_bound']-1):.2e})")
    np.save(RES / "skyrme_field_B1.npy", fs.to_numpy(phi))
    print(f"  saved shape {fs.to_numpy(phi).shape}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
