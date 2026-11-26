"""Merge the per-fold small_filled jobs (smallp0 = fold-0 model re-predicted, smallf1..4) into
oof_small_filled.npy / test_small_filled.npy (test = mean of the 5 fold models)."""
import numpy as np
from common import WORK
parts = ["smallp0", "smallf1", "smallf2", "smallf3", "smallf4"]
oofs = [np.load(WORK / f"oof_{p}.npy") for p in parts]
oof = np.full(len(oofs[0]), np.nan)
for o in oofs:
    m = ~np.isnan(o)
    assert np.isnan(oof[m]).all()
    oof[m] = o[m]
assert not np.isnan(oof).any()
test = np.mean([np.load(WORK / f"test_{p}.npy") for p in parts], 0)
np.save(WORK / "oof_small_filled.npy", oof)
np.save(WORK / "test_small_filled.npy", test)
print("merged", oof.shape, test.shape)
