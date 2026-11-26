"""Stage 2 + F1-oriented threshold selection, then write the submission.

1. Ensemble stage-1 runs (mean of logits).
2. Neighbour refinement: logistic regression on [own logit, max 4-neighbour logit, 2nd 4-neighbour,
   max diagonal] (a hole straddling a border makes adjacent tiles positive together).
   Evaluated with photo-grouped CV so OOF stays honest.
3. Threshold for F1: the F1-optimal cut on pooled OOF is noisy with ~190 positives, so we resample
   whole photos (bootstrap, B=1000), find the F1-optimal threshold in each resample and use the
   median. This is the threshold that is optimal "on a typical set of unseen photos", which is
   what the test set is.

usage: python stage2.py run1 [run2 ...] [--stage1]"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import NCOL, NROW, ROOT, WORK, load_test, load_train

names = [a for a in sys.argv[1:] if not a.startswith("--")]
tr, te = load_train(), load_test()
ytr = tr.label.values


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


oof = sigmoid(np.mean([logit(np.load(WORK / f"oof_{n}.npy")) for n in names], 0))
pte = sigmoid(np.mean([logit(np.load(WORK / f"test_{n}.npy")) for n in names], 0))


def feats(df, p):
    grid = {ph: np.full((NROW + 2, NCOL + 2), 1e-6) for ph in df.photo.unique()}
    for ph, r, c, v in zip(df.photo, df.row, df.col, p):
        grid[ph][r + 1, c + 1] = max(grid[ph][r + 1, c + 1], v)
    F = []
    for ph, r, c, v in zip(df.photo, df.row, df.col, p):
        g = grid[ph][r:r + 3, c:c + 3]
        four = np.sort([g[0, 1], g[2, 1], g[1, 0], g[1, 2]])
        diag = [g[0, 0], g[0, 2], g[2, 0], g[2, 2]]
        F.append([logit(v), logit(four[-1]), logit(four[-2]), logit(max(diag))])
    return np.array(F)


TH = np.arange(0.01, 0.99, 0.005)


def f1_curve(y, p):
    """F1 at every threshold in TH, vectorised."""
    pred = p[None, :] > TH[:, None]
    tp = (pred & (y[None, :] == 1)).sum(1)
    fp = (pred & (y[None, :] == 0)).sum(1)
    fn = ((~pred) & (y[None, :] == 1)).sum(1)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def bootstrap_threshold(y, p, groups, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in ug}
    ths, f1s = [], []
    for _ in range(B):
        idx = np.concatenate([rows[g] for g in rng.choice(ug, len(ug))])
        if y[idx].sum() == 0:
            continue
        c = f1_curve(y[idx], p[idx])
        ths.append(TH[np.argmax(c)])
    th = float(np.median(ths))
    # F1 of the chosen threshold on each bootstrap set (spread = expected score uncertainty)
    for _ in range(B):
        idx = np.concatenate([rows[g] for g in rng.choice(ug, len(ug))])
        f1s.append(f1_score(y[idx], p[idx] > th))
    return th, np.percentile(ths, [5, 95]), np.percentile(f1s, [5, 50, 95])


Ftr, Fte = feats(tr, oof), feats(te, pte)
oof2 = np.zeros(len(tr))
for tri, vai in StratifiedGroupKFold(5, shuffle=True, random_state=1).split(Ftr, ytr, tr.photo):
    oof2[vai] = LogisticRegression(max_iter=1000).fit(Ftr[tri], ytr[tri]).predict_proba(Ftr[vai])[:, 1]
lr = LogisticRegression(max_iter=1000).fit(Ftr, ytr)
pte2 = lr.predict_proba(Fte)[:, 1]
print("runs:", names)
print("stage2 coef [self, n4max, n4second, diagmax]:", np.round(lr.coef_[0], 3), "b", np.round(lr.intercept_, 3))

results = {}
for tag, p in [("stage1", oof), ("stage2", oof2)]:
    c = f1_curve(ytr, p)
    th, th_ci, f1_ci = bootstrap_threshold(ytr, p, tr.photo.values)
    pred = p > th
    fn, fp = int(((ytr == 1) & ~pred).sum()), int(((ytr == 0) & pred).sum())
    results[tag] = (th, f1_score(ytr, pred))
    print(f"{tag}: auc {roc_auc_score(ytr, p):.5f} | F1@0.5 {f1_score(ytr, p > 0.5):.4f} | oracle best {c.max():.4f}@{TH[c.argmax()]:.3f}"
          f" | bootstrap th {th:.3f} (90% {th_ci[0]:.2f}-{th_ci[1]:.2f}) -> OOF F1 {results[tag][1]:.4f} (FN {fn}, FP {fp})"
          f" | F1 over resampled photo sets 5/50/95%: {f1_ci.round(3)}")

use = "stage1" if "--stage1" in sys.argv else "stage2" if "--stage2" in sys.argv else \
    max(results, key=lambda k: results[k][1])  # default: whichever scores higher on OOF
th = results[use][0]
p_final = pte2 if use == "stage2" else pte
pred = (p_final > th).astype(int)
out = ROOT / "submission.csv"
pd.DataFrame({"path": te.path, "label": pred}).to_csv(out, header=False, index=False)
tr.assign(p=oof2 if use == "stage2" else oof).to_csv(WORK / "oof_final.csv", index=False)
te.assign(p=p_final).to_csv(WORK / "test_final.csv", index=False)
print(f"[{use}] threshold {th:.3f}; test positives {pred.sum()} / {len(pred)} ({pred.mean():.3%}); train rate {ytr.mean():.3%}")
print("wrote", out)
