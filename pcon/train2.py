"""Stage 1 v2: same as train.py, but geometric augmentation is applied jointly to the RGB context
crop AND the tile-mask channel, so the label stays exact under zoom / rotation / translation.

Why: OOF errors showed far-away (small, elliptical) P-con holes in one photo scored ~0, i.e. the
model was not scale-robust. Zoom-out needs real surrounding pixels, so a large source window is
cut from the stitched canvas and resampled with affine_grid.

From-scratch weights only; data = SDNET W tiles only."""
import argparse
import json
import math
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from common import WORK, load_test, load_train

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="v2_r18_m48")
ap.add_argument("--arch", default="resnet18")
ap.add_argument("--margin", type=int, default=48)  # half-res px of context per side (output)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--pos_frac", type=float, default=0.25)
ap.add_argument("--zmin", type=float, default=0.6)
ap.add_argument("--zmax", type=float, default=1.6)
ap.add_argument("--rot", type=float, default=20)
ap.add_argument("--shift", type=int, default=16)
ap.add_argument("--tta_scales", default="1.0")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--only_fold", type=int, default=-1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--debug_aug", action="store_true")
ap.add_argument("--head", default="cls", choices=["cls", "mil"])
ap.add_argument("--res", default="half", choices=["half", "full"])
ap.add_argument("--pool", default="max", choices=["max", "lse"])  # MIL pooling over in-tile cells
ap.add_argument("--lse_r", type=float, default=5.0)  # LSE sharpness: r->0 mean, r->inf max
ap.add_argument("--aug_filled", action="store_true")  # appearance aug aimed at mortar-filled (light) P-cons  # margin/shift args stay in half-res px
ap.add_argument("--split", default="sgkf", choices=["gkf", "sgkf"])  # sgkf: photo-grouped + positive-rate balanced
ap.add_argument("--loss", default="focal", choices=["bce", "focal"])
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--alpha", type=float, default=0.5)  # weight on positives (0.5 = none)
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

R = 2 if args.res == "full" else 1  # resolution factor relative to the half-res canvas
HT, HP, EXTRA = 128 * R, 64 * R, 192 * R  # tile, canvas pad from prep.py, extra reflect pad here
M = args.margin * R
S = HT + 2 * M  # network input size
MS = M + 128 * R  # source margin
SS = HT + 2 * MS  # source window size
assert MS <= HP + EXTRA

tr, te = load_train(), load_test()
canv = {}
for ph in set(tr.photo) | set(te.photo):
    c = np.load(WORK / ("canvas_s1" if args.res == "full" else "canvas") / f"{ph}.npy")
    canv[ph] = np.pad(c, ((EXTRA, EXTRA), (EXTRA, EXTRA), (0, 0)), mode="reflect")


def src_crops(df, idx):
    out = np.empty((len(idx), SS, SS, 3), np.uint8)
    for j, i in enumerate(idx):
        ph, r, c = df.photo.iat[i], df.row.iat[i], df.col.iat[i]
        y0, x0 = HP + EXTRA + r * HT - MS, HP + EXTRA + c * HT - MS
        out[j] = canv[ph][y0:y0 + SS, x0:x0 + SS]
    return torch.from_numpy(out).to(dev).permute(0, 3, 1, 2).float()


ytr = tr.label.values.astype(np.float32)
_s = src_crops(tr, np.random.choice(len(tr), 500, replace=False))
MEAN, STD = _s.mean((0, 2, 3)).view(1, 3, 1, 1), _s.std((0, 2, 3)).view(1, 3, 1, 1)
del _s
MASK = torch.zeros(1, 1, SS, SS, device=dev)
MASK[..., MS:MS + HT, MS:MS + HT] = 1


def to_input(src, theta):
    """src: (n,3,SS,SS) float raw pixels; theta: (n,2,3) mapping output->source normalized coords."""
    x = torch.cat([(src - MEAN) / STD, MASK.expand(len(src), -1, -1, -1)], 1)
    grid = F.affine_grid(theta, (len(src), 4, S, S), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    x[:, 3] = x[:, 3] * 2 - 1
    return x


def make_theta(n, train, k=0, z=1.0):
    base = S / SS  # identity crop of the central S window
    if train:
        z = torch.exp(torch.empty(n).uniform_(math.log(args.zmin), math.log(args.zmax)))
        aniso = torch.exp(torch.empty(n).uniform_(-0.15, 0.15))
        ang = torch.empty(n).uniform_(-args.rot, args.rot) * math.pi / 180 + torch.randint(0, 4, (n,)) * math.pi / 2
        flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
        t = torch.empty(n, 2).uniform_(-args.shift, args.shift) * R * 2 / SS
    else:
        z = torch.full((n,), z)
        aniso = torch.ones(n)
        ang = torch.full((n,), (k % 4) * math.pi / 2)
        flip = torch.full((n,), -1.0 if k >= 4 else 1.0)
        t = torch.zeros(n, 2)
    sx, sy = base / z * aniso, base / z / aniso
    cos, sin = torch.cos(ang), torch.sin(ang)
    th = torch.zeros(n, 2, 3)
    th[:, 0, 0], th[:, 0, 1] = cos * sx * flip, -sin * sy
    th[:, 1, 0], th[:, 1, 1] = sin * sx * flip, cos * sy
    th[:, :, 2] = t
    return th.to(dev)


def photometric(x):
    n = len(x)
    rgb = x[:, :3]
    gain = 1 + 0.25 * (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1)
    bias = 0.3 * (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1)
    chan = 1 + 0.05 * (torch.rand(n, 3, 1, 1, device=dev) * 2 - 1)
    rgb = rgb * gain * chan + bias
    rgb = rgb + 0.05 * torch.randn_like(rgb) * (torch.rand(n, 1, 1, 1, device=dev) < 0.3)
    if torch.rand(1).item() < 0.3:  # mild blur
        rgb = F.avg_pool2d(rgb, 3, 1, 1)
    if args.aug_filled:
        # Mortar-filled P-cons are light, low-contrast discs, so "dark hole" alone is not enough.
        # Per-sample: luminance inversion, strong contrast reduction, or grayscale, to push the net
        # toward shape cues (circular rim / ellipse outline) rather than darkness.
        mu = rgb.mean((1, 2, 3), keepdim=True)
        inv = torch.rand(n, 1, 1, 1, device=dev) < 0.2
        rgb = torch.where(inv, 2 * mu - rgb, rgb)
        low = torch.rand(n, 1, 1, 1, device=dev) < 0.3
        c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.3, 1.0)
        rgb = torch.where(low, mu + (rgb - mu) * c, rgb)
        gray = torch.rand(n, 1, 1, 1, device=dev) < 0.2
        rgb = torch.where(gray, rgb.mean(1, keepdim=True).expand_as(rgb), rgb)
    return torch.cat([rgb, x[:, 3:]], 1)


def loss_fn(logit, y):
    if args.loss == "bce":
        return F.binary_cross_entropy_with_logits(logit, y)
    # Focal loss: down-weights easy (mostly background) tiles so hard ones drive the gradient
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = y * p + (1 - y) * (1 - p)
    a = y * args.alpha + (1 - y) * (1 - args.alpha)
    return (a * (1 - pt) ** args.gamma * ce).mean()


class MILNet(torch.nn.Module):
    """Dense P-con map + max over cells inside the tile (multiple-instance pooling).

    Encodes the label definition directly: a tile is positive iff some part of a P-con lies
    inside it. The backbone sees RGB only (translation-equivariant detector); the transformed
    tile mask is used only to choose which cells of the map are pooled, so the tile border is
    handled geometrically instead of being inferred from a mask channel."""

    def __init__(self):
        super().__init__()
        self.body = timm.create_model(args.arch, pretrained=False, features_only=True, out_indices=(3, 4), in_chans=3)
        c3, c4 = self.body.feature_info.channels()
        self.head = torch.nn.Sequential(torch.nn.Conv2d(c3 + c4, 128, 3, padding=1), torch.nn.BatchNorm2d(128),
                                        torch.nn.ReLU(inplace=True), torch.nn.Conv2d(128, 1, 1))

    def forward(self, x, return_map=False):
        f3, f4 = self.body(x[:, :3])
        f4 = F.interpolate(f4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        logit_map = self.head(torch.cat([f3, f4], 1))  # stride 16
        stride = x.shape[-1] // logit_map.shape[-1]
        inside = F.avg_pool2d((x[:, 3:4] > 0).float(), stride) > 0.5
        lm = logit_map.float().flatten(1)
        inside = inside.flatten(1)
        if args.pool == "max":
            out = lm.masked_fill(~inside, -1e4).max(1).values
        else:
            # LSE pooling, normalised by cell count so it lies between mean and max of in-tile cells;
            # every in-tile cell gets gradient and the tile score is less over-confident than a hard max
            r = args.lse_r
            n = inside.sum(1).clamp(min=1).float()
            out = (torch.logsumexp(r * lm.masked_fill(~inside, -1e4), 1) - torch.log(n)) / r
        out = out.unsqueeze(1)
        return (out, logit_map) if return_map else out


def make_model():
    if args.head == "mil":
        return MILNet().to(dev)
    return timm.create_model(args.arch, pretrained=False, num_classes=1, in_chans=4).to(dev)


@torch.no_grad()
def predict(model, df, idx):
    model.eval()
    scales = [float(s) for s in args.tta_scales.split(",")]
    out = []
    for i in range(0, len(idx), 128):
        bi = idx[i:i + 128]
        src = src_crops(df, bi)
        p, cnt = 0, 0
        for z in scales:
            for k in range(8):
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    p = p + torch.sigmoid(model(to_input(src, make_theta(len(bi), False, k, z))).float()).squeeze(1)
                cnt += 1
        out.append((p / cnt).cpu())
    return torch.cat(out).numpy()


def train_fold(tri):
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    spe = len(tri) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * spe, pct_start=0.1)
    pos, neg = tri[ytr[tri] == 1], tri[ytr[tri] == 0]
    npos = int(args.bs * args.pos_frac)  # pos_frac <= 0: plain uniform sampling (natural imbalance)
    for ep in range(args.epochs):
        model.train()
        tl = 0
        for _ in range(spe):
            if args.pos_frac > 0:
                idx = np.concatenate([np.random.choice(pos, npos), np.random.choice(neg, args.bs - npos)])
            else:
                idx = np.random.choice(tri, args.bs)
            x = photometric(to_input(src_crops(tr, idx), make_theta(len(idx), True)))
            y = torch.from_numpy(ytr[idx]).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1).float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"  ep{ep + 1} loss {tl / spe:.4f}", flush=True)
    return model


def best_f1(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    fs = [f1_score(y, p > t) for t in ths]
    i = int(np.argmax(fs))
    return fs[i], ths[i]


if args.debug_aug:  # save augmented positives (mask outline in red) for a visual check, then exit
    from PIL import Image
    pi = np.where(ytr == 1)[0][:8]
    rows = []
    for train in (False, True, True, True):
        x = to_input(src_crops(tr, pi), make_theta(len(pi), train))
        rgb = (x[:, :3] * STD + MEAN).clamp(0, 255)
        edge = (x[:, 3:] > 0).float()
        edge = edge - F.max_pool2d(-edge, 3, 1, 1).neg()  # inner boundary
        rgb[:, 0:1] = torch.where(edge.abs() > 0, torch.full_like(rgb[:, :1], 255), rgb[:, :1])
        rows.append(torch.cat(list(rgb), 2))
    img = torch.cat(rows, 1).permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(img).save(WORK / "debug_aug.jpg")
    raise SystemExit

oof = np.full(len(tr), np.nan)
pte = np.zeros(len(te))
nf = 0
cv = GroupKFold(args.folds) if args.split == "gkf" else StratifiedGroupKFold(args.folds, shuffle=True, random_state=0)
fold_th = []
for f, (tri, vai) in enumerate(cv.split(tr, ytr, tr.photo)):
    if args.only_fold >= 0 and f != args.only_fold:
        continue
    t0 = time.time()
    model = train_fold(tri)
    oof[vai] = predict(model, tr, vai)
    pte += predict(model, te, np.arange(len(te)))
    nf += 1
    fb, tb = best_f1(ytr[vai], oof[vai])
    fold_th.append(tb)
    print(f"fold {f}: auc {roc_auc_score(ytr[vai], oof[vai]):.4f} f1@0.5 {f1_score(ytr[vai], oof[vai] > 0.5):.4f} "
          f"best f1 {fb:.4f}@{tb:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), WORK / f"{args.name}_f{f}.pt")
pte /= nf

ok = ~np.isnan(oof)
fb, tb = best_f1(ytr[ok], oof[ok])
res = dict(fold_best_th=[round(t, 2) for t in fold_th], name=args.name, auc=roc_auc_score(ytr[ok], oof[ok]), f1_05=f1_score(ytr[ok], oof[ok] > 0.5), best_f1=fb, best_th=tb)
print(json.dumps(res))
np.save(WORK / f"oof_{args.name}.npy", oof)
np.save(WORK / f"test_{args.name}.npy", pte)
