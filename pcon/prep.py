"""Stitch every photo (train + test) into a half-resolution canvas with reflective padding.
Saved as work/canvas/<photo>.npy so training can crop each tile with surrounding context."""
import sys

import numpy as np
import pandas as pd
from PIL import Image

from common import WORK, T, load_test, load_train, stitch

SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
PAD = 128  # full-res px of reflective margin around the photo

out = WORK / ("canvas" if SCALE == 0.5 else f"canvas_s{SCALE:g}")
out.mkdir(exist_ok=True)
df = pd.concat([load_train().assign(split="train"), load_test().assign(split="test")])
print(df.groupby("split").photo.nunique())
dups = df[df.duplicated(["photo", "idx"], keep=False)]
if len(dups):
    print("duplicate tiles:\n", dups)
for ph, d in df.groupby("photo"):
    cv, filled = stitch(d, pad=PAD)
    h, w = cv.shape[:2]
    small = cv if SCALE == 1 else np.asarray(Image.fromarray(cv).resize((int(w * SCALE), int(h * SCALE)), Image.BILINEAR))
    np.save(out / f"{ph}.npy", small)
    if not filled.all():
        print(ph, "missing tiles:", (~filled).sum())
print("done")
