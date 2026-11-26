"""Write many submission candidates (ensemble x threshold) into submissions/cands/ with a summary.

Every candidate: logit-mean of the listed runs' fold-averaged test probabilities, cut at a fixed
threshold (the LB showed 0.50 best for ens4; neighbours 0.45-0.55 are included). Runs whose files
are missing are skipped, so the script can be re-run as more runs finish."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from common import DIST, ROOT, WORK, load_test, load_train

tr, te = load_train(), load_test()
y = tr.label.values
txt = [l.strip() for l in open(DIST / "test.txt") if l.strip()]
out_dir = ROOT / "submissions" / "cands"
out_dir.mkdir(parents=True, exist_ok=True)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

BASE = ["v4_mil_focal", "v6_lse", "v6_filled", "v6_both"]
MS = ["tta_v4_ms", "ms_v6_lse", "ms_v6_filled", "ms_v6_both"]
ENS = {
    "ens4": BASE,
    "ens5": BASE + ["b1_lse2"],
    "ens7": BASE + ["b1_lse2", "s1_lse", "s1_filled"],
    "ens8": BASE + ["b1_lse2", "s1_lse", "s1_filled", "fit_filled"],
    "ens4ms": MS,
    "ens5ms": MS + ["b1_lse2"],
    "ens8ms": MS + ["b1_lse2", "s1_lse", "s1_filled", "fit_filled"],
    "ensSmall": BASE + ["small_filled"],
    "ens8msSmall": MS + ["b1_lse2", "s1_lse", "s1_filled", "fit_filled", "small_filled"],
    "filledFamily": ["v6_filled", "s1_filled", "fit_filled", "ms_v6_filled", "small_filled"],
    "lseFamily": ["v6_lse", "s1_lse", "b1_lse2", "ms_v6_lse"],
    "msLse": ["ms_v6_lse"],
    "msFilled": ["ms_v6_filled"],
    "small": ["small_filled"],
    "hnV4": ["hn_v4"],
    "hnLse": ["hn_lse"],
    "hn2": ["hn_v4", "hn_lse"],
    "ens4hn": ["hn_v4", "hn_lse", "v6_filled", "v6_both"],
    "ens6mshn": MS + ["hn_v4", "hn_lse"],
}
THS = {"ens4": [0.45, 0.475, 0.5, 0.525, 0.55], "ens5": [0.45, 0.5, 0.55], "ens7": [0.45, 0.5, 0.55],
       "ens8": [0.45, 0.5, 0.55], "ens4ms": [0.45, 0.5, 0.55], "ens5ms": [0.45, 0.5, 0.55],
       "ens8ms": [0.45, 0.5, 0.55], "hnV4": [0.35, 0.4, 0.45, 0.5], "hnLse": [0.35, 0.4, 0.45, 0.5],
       "hn2": [0.35, 0.4, 0.45, 0.5], "ens4hn": [0.4, 0.45, 0.5, 0.55], "ens6mshn": [0.4, 0.45, 0.5, 0.55], "ensSmall": [0.45, 0.5, 0.55], "ens8msSmall": [0.45, 0.5, 0.55]}
ref = pd.read_csv(ROOT / "submissions" / "submission_ens4_mil_th50.csv", header=None)[1].values
rows = []
for name, runs in ENS.items():
    if not all((WORK / f"test_{r}.npy").exists() and (WORK / f"oof_{r}.npy").exists() for r in runs):
        continue
    o = 1 / (1 + np.exp(-np.mean([lg(np.load(WORK / f"oof_{r}.npy")) for r in runs], 0)))
    t = 1 / (1 + np.exp(-np.mean([lg(np.load(WORK / f"test_{r}.npy")) for r in runs], 0)))
    if np.isnan(o).any() or np.isnan(t).any():
        continue
    ap = average_precision_score(y, o)
    best = max(f1_score(y, o > th) for th in np.arange(0.3, 0.9, 0.01))
    for th in THS.get(name, [0.5]):
        pred = (t > th).astype(int)
        fn = out_dir / f"sub_{name}_th{int(round(th * 1000)):03d}.csv"
        pd.DataFrame({"path": te.path, "label": pred}).to_csv(fn, header=False, index=False)
        chk = pd.read_csv(fn, header=None)
        assert chk[0].tolist() == txt and set(chk[1]) <= {0, 1} and len(chk) == len(txt)
        rows.append(dict(file=fn.name, runs="+".join(runs), threshold=th, oof_ap=round(ap, 4), oof_best_f1=round(best, 4),
                         oof_f1_at_th=round(f1_score(y, o > th), 4), test_pos=int(pred.sum()),
                         added_vs_lb909=int(((pred == 1) & (ref == 0)).sum()), removed_vs_lb909=int(((pred == 0) & (ref == 1)).sum())))
df = pd.DataFrame(rows)
df.to_csv(out_dir / "summary.csv", index=False)
pd.set_option("display.width", 250)
print(df.drop(columns="runs").to_string(index=False))
