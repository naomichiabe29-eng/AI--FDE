"""Draw OOF errors on stitched photos: green=TP, red=FN (missed), yellow=FP. Also saves full-res
crops (tile + 128px context) of every error for close inspection.
usage: python viz_errors.py <oof_name> [threshold]"""
import sys

import numpy as np
from PIL import Image, ImageDraw

from common import NCOL, NROW, T, WORK, load_train, stitch

name = sys.argv[1]
th = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
tr = load_train()
tr["p"] = np.load(WORK / f"oof_{name}.npy")
tr["pred"] = (tr.p > th).astype(int)
err = tr[tr.pred != tr.label]
print(f"errors: FN {((tr.label == 1) & (tr.pred == 0)).sum()}  FP {((tr.label == 0) & (tr.pred == 1)).sum()}")
out = WORK / f"err_{name}"
out.mkdir(exist_ok=True)
S = 64
for ph in err.photo.unique():
    d = tr[tr.photo == ph]
    cv, _ = stitch(d, pad=0)
    im = Image.fromarray(cv).resize((NCOL * S, NROW * S))
    dr = ImageDraw.Draw(im)
    for _, x in d.iterrows():
        col = {(1, 1): (0, 200, 0), (1, 0): (255, 0, 0), (0, 1): (255, 200, 0)}.get((x.label, x.pred))
        if col:
            dr.rectangle([x.col * S, x.row * S, x.col * S + S - 1, x.row * S + S - 1], outline=col, width=2)
            if x.label != x.pred:
                dr.text((x.col * S + 3, x.row * S + 3), f"{x.p:.2f}", fill=col)
    im.save(out / f"photo_{ph}.jpg", quality=85)
    big = np.pad(cv, ((128, 128), (128, 128), (0, 0)), mode="reflect")
    for _, x in d[d.label != d.pred].iterrows():
        y0, x0 = x.row * T, x.col * T
        c = Image.fromarray(big[y0:y0 + T + 256, x0:x0 + T + 256].copy())
        ImageDraw.Draw(c).rectangle([128, 128, 128 + T - 1, 128 + T - 1], outline=(255, 0, 0) if x.label else (255, 200, 0), width=2)
        c.save(out / f"{'FN' if x.label else 'FP'}_{ph}-{x.idx}_p{x.p:.2f}.jpg", quality=90)
print(out)
print(err[["path", "label", "p"]].sort_values("p").to_string())
