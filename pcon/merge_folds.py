"""Merge per-fold jobs <prefix>0..4 into oof_<out>.npy / test_<out>.npy (test = mean over folds).
usage: python merge_folds.py <prefix> <out>"""
import sys
import numpy as np
from common import WORK
prefix, out = sys.argv[1], sys.argv[2]
oofs = [np.load(WORK / f"oof_{prefix}{f}.npy") for f in range(5)]
oof = np.full(len(oofs[0]), np.nan)
for o in oofs:
    m = ~np.isnan(o)
    oof[m] = o[m]
assert not np.isnan(oof).any()
np.save(WORK / f"oof_{out}.npy", oof)
np.save(WORK / f"test_{out}.npy", np.mean([np.load(WORK / f"test_{prefix}{f}.npy") for f in range(5)], 0))
print("merged", out)
