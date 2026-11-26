"""Evaluate logit-average ensembles of finished runs on pooled OOF and optionally write a submission.

usage:
  python ens.py eval run1 run2 ...          # every single run + every subset of size >= 2
  python ens.py write out_name run1 run2 ...  # write submissions/submission_<out_name>.csv (+ root submission.csv with --root)
Threshold = F1-optimal threshold on the pooled OOF of the same ensemble (threshold tuning deferred)."""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import DIST, ROOT, WORK, load_test, load_train

tr, te = load_train(), load_test()
y = tr.label.values
TH = np.arange(0.05, 0.96, 0.01)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load(names):
    o = 1 / (1 + np.exp(-np.mean([logit(np.load(WORK / f"oof_{n}.npy")) for n in names], 0)))
    t = 1 / (1 + np.exp(-np.mean([logit(np.load(WORK / f"test_{n}.npy")) for n in names], 0)))
    return o, t


def score(o):
    f = np.array([f1_score(y, o > t) for t in TH])
    i = int(f.argmax())
    return dict(auc=roc_auc_score(y, o), ap=average_precision_score(y, o), f1_05=f1_score(y, o > 0.5),
                best_f1=f[i], th=TH[i], fn=int(((y == 1) & (o <= TH[i])).sum()), fp=int(((y == 0) & (o > TH[i])).sum()))


mode = sys.argv[1]
if mode == "eval":
    names = sys.argv[2:]
    rows = []
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            o, _ = load(combo)
            rows.append(dict(runs="+".join(combo), n=k, **score(o)))
    df = pd.DataFrame(rows).sort_values("ap", ascending=False)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
else:
    out_name, names = sys.argv[2], [a for a in sys.argv[3:] if not a.startswith("--")]
    o, t = load(names)
    s = score(o)
    pred = (t > s["th"]).astype(int)
    txt = [l.strip() for l in open(DIST / "test.txt") if l.strip()]
    out = ROOT / "submissions" / f"submission_{out_name}.csv"
    pd.DataFrame({"path": te.path, "label": pred}).to_csv(out, header=False, index=False)
    chk = pd.read_csv(out, header=None)
    assert chk[0].tolist() == txt and set(chk[1]) <= {0, 1} and len(chk) == len(txt)
    if "--root" in sys.argv:
        pd.DataFrame({"path": te.path, "label": pred}).to_csv(ROOT / "submission.csv", header=False, index=False)
    print(f"{out.name}: runs={names} OOF auc {s['auc']:.4f} ap {s['ap']:.4f} best_f1 {s['best_f1']:.4f}@{s['th']:.2f} "
          f"(FN {s['fn']}, FP {s['fp']}) | test positives {pred.sum()}/{len(pred)} ({pred.mean():.2%})"
          + (" | also written to submission.csv" if "--root" in sys.argv else ""))
