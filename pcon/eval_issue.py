"""Compare ensembles on the two remaining issues (OOF, labels of training photos only).

  small   : P-cons in photo 7131 (far, small, elliptical) -> misses
  void/fit: false positives on the training tiles known to be voids, dark spots or pipe fittings
Also prints, for the test set (no labels), how many tiles of the rough photo 7112 fall in the
uncertain band, as a rough sign of how a model behaves on void-heavy surfaces.

usage: python eval_issue.py "name=runA,runB" "name2=runC" ...   (threshold 0.50 = the LB-chosen one)"""
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from common import WORK, load_test, load_train

tr, te = load_train(), load_test()
y = tr.label.values
VOID_FIT = {"7111-90", "7107-2", "7113-19", "7113-20", "7123-221", "7111-143", "7134-145", "7134-164", "7134-182",
            "7134-183", "7134-199", "7133-236", "7133-215", "7123-229"}
key = (tr.photo + "-" + tr.idx.astype(str)).values
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def ens(names, split):
    return 1 / (1 + np.exp(-np.mean([lg(np.load(WORK / f"{split}_{n}.npy")) for n in names], 0)))


print(f"{'candidate':14s} {'AP':>6s} {'bestF1':>7s} {'F1@.5':>6s} | 7131 missed@.5 (of {int(y[tr.photo == '7131'].sum())}) | "
      f"void/fitting FP@.5 (of {np.isin(key, list(VOID_FIT)).sum()}) | total FN/FP@.5 | test 7112: pos@.5, 0.2-0.8 band")
for arg in sys.argv[1:]:
    name, runs = arg.split("=")
    o, t = ens(runs.split(","), "oof"), ens(runs.split(","), "test")
    best = max(f1_score(y, o > th) for th in np.arange(0.3, 0.9, 0.01))
    p = o > 0.5
    m7131 = int(((tr.photo == "7131").values & (y == 1) & ~p).sum())
    vf = int((np.isin(key, list(VOID_FIT)) & p).sum())
    t12 = t[(te.photo == "7112").values]
    print(f"{name:14s} {average_precision_score(y, o):6.4f} {best:7.4f} {f1_score(y, p):6.4f} | {m7131:>3d} | {vf:>3d} | "
          f"{int(((y == 1) & ~p).sum())}/{int(((y == 0) & p).sum())} | {int((t12 > 0.5).sum())}, {int(((t12 > 0.2) & (t12 < 0.8)).sum())}")
