"""Build the final submission: logit-mean of the 4 MIL models' test predictions, threshold 0.50.

Inputs : work/test_{v4_mil_focal,v6_lse,v6_filled,v6_both}.npy  (written by train2.py)
Output : ../submission.csv  (no header, "path,label", same order as test.txt)
"""
import numpy as np
import pandas as pd

from common import DIST, ROOT, WORK, load_test

RUNS = ["v4_mil_focal", "v6_lse", "v6_filled", "v6_both"]
THRESHOLD = 0.50

te = load_test()
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
prob = 1 / (1 + np.exp(-np.mean([logit(np.load(WORK / f"test_{r}.npy")) for r in RUNS], 0)))
pred = (prob > THRESHOLD).astype(int)

out = ROOT / "submission.csv"
pd.DataFrame({"path": te.path, "label": pred}).to_csv(out, header=False, index=False)

txt = [l.strip() for l in open(DIST / "test.txt") if l.strip()]
chk = pd.read_csv(out, header=None)
assert chk[0].tolist() == txt and set(chk[1]) <= {0, 1}
print(f"wrote {out}: {len(chk)} rows, {pred.sum()} positives")
