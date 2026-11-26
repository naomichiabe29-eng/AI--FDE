"""Shared utilities: paths, label loading, photo stitching (18 cols x 14 rows grid)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DIST = Path(os.environ.get("PCON_DIST", ROOT / "uc?export=download&id=1f9imDAg2nZYwhzL09AkzbIZD3iDpFjCA&name=distribution"))
IMG_ROOT = Path(os.environ.get("PCON_IMG", ROOT / "DATA_Maguire_20180517_ALL" / "SDNET2018"))
WORK = ROOT / "pcon" / "work"
WORK.mkdir(parents=True, exist_ok=True)

NCOL, NROW, T = 18, 14, 256


def parse(path):
    """'W/UW/7131-22.jpg' -> (photo '7131', index 22). Handles odd names like '99_2'."""
    stem = Path(path).stem
    photo, idx = stem.split("-", 1)
    idx = idx.split("_")[0]  # 'N_2' files are re-saved versions of position N
    return photo, int(idx) if idx.isdigit() else -1


def load_train():
    df = pd.read_csv(DIST / "train_label.csv", header=None, names=["path", "label"])
    return _annotate(df)


def load_test():
    df = pd.read_csv(DIST / "test.txt", header=None, names=["path"])
    return _annotate(df)


def _annotate(df):
    p = df["path"].map(parse)
    df["photo"] = [a for a, _ in p]
    df["idx"] = [b for _, b in p]
    df["row"] = [(i - 1) // NCOL if i > 0 else -1 for i in df["idx"]]
    df["col"] = [(i - 1) % NCOL if i > 0 else -1 for i in df["idx"]]
    return df


def stitch(df_photo, pad=T):
    """Stitch all tiles of one photo into a canvas with `pad` px reflective margin.
    Returns uint8 array (H, W, 3); tile (r, c) occupies [pad + r*T, pad + c*T]."""
    canvas = np.zeros((NROW * T, NCOL * T, 3), np.uint8)
    filled = np.zeros((NROW, NCOL), bool)
    for path, r, c in zip(df_photo["path"], df_photo["row"], df_photo["col"]):
        if r < 0:
            continue
        im = np.asarray(Image.open(IMG_ROOT / path).convert("RGB"))
        canvas[r * T:(r + 1) * T, c * T:(c + 1) * T] = im
        filled[r, c] = True
    if pad:
        canvas = np.pad(canvas, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    return canvas, filled
