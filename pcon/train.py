"""Stage 1: from-scratch CNN on each tile + surrounding context (no pretrained weights, no external data).

Input = crop of the stitched half-res canvas centered on the tile, with `margin` px of context on
each side, plus a 4th channel marking the tile region, so the net can judge whether a P-con hole
that straddles the tile border actually reaches into the tile.

Outputs OOF probabilities (GroupKFold by source photo) and fold-averaged test probabilities."""
import argparse
import json
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import WORK, load_test, load_train

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="r18_m32")
ap.add_argument("--arch", default="resnet18")
ap.add_argument("--margin", type=int, default=32)  # half-res px of context per side
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--pos_frac", type=float, default=0.25)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--only_fold", type=int, default=-1)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

HT, HP = 128, 64  # half-res tile size and canvas pad
M = args.margin
S = HT + 2 * M

tr, te = load_train(), load_test()
canv = {}


def crops(df):
    out = np.zeros((len(df), S, S, 3), np.uint8)
    for i, (ph, r, c) in enumerate(zip(df.photo, df.row, df.col)):
        if ph not in canv:
            canv[ph] = np.load(WORK / "canvas" / f"{ph}.npy")
        y0, x0 = HP + r * HT - M, HP + c * HT - M
        out[i] = canv[ph][y0:y0 + S, x0:x0 + S]
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW


Xtr, Xte = crops(tr), crops(te)
ytr = tr.label.values.astype(np.float32)
mask = torch.zeros(1, 1, S, S)
mask[..., M:M + HT, M:M + HT] = 1
mask = mask.to(dev)
MEAN = Xtr[:2000].float().mean((0, 2, 3)).view(1, 3, 1, 1).to(dev)
STD = Xtr[:2000].float().std((0, 2, 3)).view(1, 3, 1, 1).to(dev)


def prep(xb, k=0):
    x = (xb.to(dev).float() - MEAN) / STD
    x = torch.cat([x, mask.expand(len(x), -1, -1, -1) * 2 - 1], 1)
    return dihedral(x, k)


def dihedral(x, k):
    if k >= 4:
        x = x.flip(3)
    return torch.rot90(x, k % 4, (2, 3))


def augment(x):
    n = len(x)
    # per-sample dihedral (mask channel is symmetric, so labels are preserved)
    ks = torch.randint(0, 8, (n,))
    x = torch.stack([dihedral(x[i:i + 1], int(ks[i]))[0] for i in range(n)])
    rgb = x[:, :3]
    gain = 1 + 0.25 * (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1)
    bias = 0.3 * (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1)
    chan = 1 + 0.05 * (torch.rand(n, 3, 1, 1, device=dev) * 2 - 1)
    rgb = rgb * gain * chan + bias
    rgb = rgb + 0.05 * torch.randn_like(rgb) * (torch.rand(n, 1, 1, 1, device=dev) < 0.3)
    # mild scale jitter about the center (keeps tile boundary close to correct)
    if torch.rand(1).item() < 0.5:
        s = 1 + 0.08 * (torch.rand(1).item() * 2 - 1)
        theta = torch.tensor([[s, 0, 0], [0, s, 0]], device=dev).unsqueeze(0).expand(n, -1, -1)
        grid = F.affine_grid(theta, rgb.shape, align_corners=False)
        rgb = F.grid_sample(rgb, grid, padding_mode="reflection", align_corners=False)
    return torch.cat([rgb, x[:, 3:]], 1)


def make_model():
    return timm.create_model(args.arch, pretrained=False, num_classes=1, in_chans=4).to(dev)


@torch.no_grad()
def predict(model, X, tta=8):
    model.eval()
    out = []
    for i in range(0, len(X), 256):
        xb = X[i:i + 256]
        p = sum(torch.sigmoid(model(prep(xb, k))).squeeze(1) for k in range(tta)) / tta
        out.append(p.cpu())
    return torch.cat(out).numpy()


def train_fold(tri):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = len(tri) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    pos, neg = tri[ytr[tri] == 1], tri[ytr[tri] == 0]
    npos = int(args.bs * args.pos_frac)
    for ep in range(args.epochs):
        model.train()
        tl = 0
        for _ in range(steps_per_epoch):
            idx = np.concatenate([np.random.choice(pos, npos), np.random.choice(neg, args.bs - npos)])
            x = augment(prep(Xtr[idx]))
            y = torch.from_numpy(ytr[idx]).to(dev)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"  ep{ep + 1} loss {tl / steps_per_epoch:.4f}", flush=True)
    return model


def best_f1(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    fs = [f1_score(y, p > t) for t in ths]
    i = int(np.argmax(fs))
    return fs[i], ths[i]


oof = np.full(len(tr), np.nan)
pte = np.zeros(len(te))
gkf = GroupKFold(args.folds)
nf = 0
for f, (tri, vai) in enumerate(gkf.split(tr, ytr, tr.photo)):
    if args.only_fold >= 0 and f != args.only_fold:
        continue
    t0 = time.time()
    model = train_fold(tri)
    oof[vai] = predict(model, Xtr[vai])
    pte += predict(model, Xte)
    nf += 1
    fb, tb = best_f1(ytr[vai], oof[vai])
    print(f"fold {f}: auc {roc_auc_score(ytr[vai], oof[vai]):.4f} f1@0.5 {f1_score(ytr[vai], oof[vai] > 0.5):.4f} "
          f"best f1 {fb:.4f}@{tb:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), WORK / f"{args.name}_f{f}.pt")
pte /= nf

ok = ~np.isnan(oof)
fb, tb = best_f1(ytr[ok], oof[ok])
res = dict(name=args.name, auc=roc_auc_score(ytr[ok], oof[ok]), f1_05=f1_score(ytr[ok], oof[ok] > 0.5), best_f1=fb, best_th=tb)
print(json.dumps(res))
np.save(WORK / f"oof_{args.name}.npy", oof)
np.save(WORK / f"test_{args.name}.npy", pte)
