"""Batch-1 experiments on the MIL model (half resolution, from scratch, SDNET W only).

Base = v4_mil_focal: ResNet18 dense P-con map, max over in-tile cells, Focal loss, positives 25% of
each batch, joint geometric aug (zoom 0.4-1.6, aniso, rot, flip, shift), photometric aug,
StratifiedGroupKFold by photo. New switches:

  --pad mean           photo border filled with the photo's mean colour instead of mirroring
                       (mirroring turned P-cons cut by the photo border into symmetric slits)
  --cell_cover c       in-tile cells = coverage > c (0.5 before; 0 = any overlap)
  --cell_weight soft   pool over (cell logit + log coverage) instead of a hard in/out cut
  --stride8            add layer2 so the map has 8-px cells (finer tile-border decisions)
  --pool lse --lse_r   LSE pooling sharpness
  --ema d              evaluate an exponential moving average of the weights
  --retile f           fraction of each batch drawn from re-tiled windows (see build_retile)
  --predict_only       reuse saved fold models of --load (e.g. for multi-scale TTA)
"""
import argparse
import copy
import json
import math
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import NCOL, NROW, WORK, load_test, load_train

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--arch", default="resnet18")
ap.add_argument("--margin", type=int, default=48)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--pos_frac", type=float, default=0.25)
ap.add_argument("--zmin", type=float, default=0.4)
ap.add_argument("--zmax", type=float, default=1.6)
ap.add_argument("--rot", type=float, default=20)
ap.add_argument("--shift", type=int, default=16)
ap.add_argument("--tta_scales", default="1.0")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--only_fold", type=int, default=-1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pool", default="max", choices=["max", "lse"])
ap.add_argument("--lse_r", type=float, default=5.0)
ap.add_argument("--aug_filled", action="store_true")
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--pad", default="mean", choices=["reflect", "mean"])
ap.add_argument("--cell_cover", type=float, default=0.5)
ap.add_argument("--cell_weight", default="hard", choices=["hard", "soft"])
ap.add_argument("--stride8", action="store_true")
ap.add_argument("--ema", type=float, default=0.0)
ap.add_argument("--retile", type=float, default=0.0)
ap.add_argument("--retile_from", default="v4_mil_focal")
ap.add_argument("--predict_only", action="store_true")
ap.add_argument("--load", default="")
ap.add_argument("--debug_retile", action="store_true")
ap.add_argument("--ohem", action="store_true")
ap.add_argument("--finetune", action="store_true")  # start from --load fold models, short low-lr training
ap.add_argument("--ft_lr", type=float, default=1e-4)
ap.add_argument("--hard_pool", type=int, default=300)  # top-scoring training negatives of the fold = hard pool
ap.add_argument("--n_hard", type=int, default=16)  # per batch: pos_frac*bs positives, n_hard hard negs, rest random negs  # half of the negatives = hardest of a 3x random pool (scored by the current model)
ap.add_argument("--fit_tile", action="store_true")  # limit zoom so the whole (rotated, shifted) tile stays in the input
ap.add_argument("--no_context", action="store_true")  # single-image model: pixels outside the tile replaced by the tile mean
args = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

HT, HP, EXTRA = 128, 64, 192  # half-res tile, pad baked into canvas npy, extra pad added here
OFF = HP + EXTRA  # photo origin inside the padded canvas
PH_H, PH_W = NROW * HT, NCOL * HT
M = args.margin
S = HT + 2 * M
MS = M + 128
SS = HT + 2 * MS
assert MS <= OFF

tr, te = load_train(), load_test()
ytr = tr.label.values.astype(np.float32)
raw = {ph: np.load(WORK / "canvas" / f"{ph}.npy") for ph in set(tr.photo) | set(te.photo)}  # reflect-padded by HP


def padded(c, mode):
    if mode == "reflect":
        return np.pad(c, ((EXTRA, EXTRA), (EXTRA, EXTRA), (0, 0)), mode="reflect")
    core = c[HP:-HP, HP:-HP]
    mean = core.reshape(-1, 3).mean(0).astype(np.uint8)
    out = np.empty((core.shape[0] + 2 * OFF, core.shape[1] + 2 * OFF, 3), np.uint8)
    out[:] = mean
    out[OFF:OFF + core.shape[0], OFF:OFF + core.shape[1]] = core
    return out


canv = {ph: padded(c, args.pad) for ph, c in raw.items()}


def crops_xy(photos, ys, xs):
    """Source windows for tiles whose top-left (photo half-res coords) is (y, x)."""
    out = np.empty((len(photos), SS, SS, 3), np.uint8)
    for j, (ph, y, x) in enumerate(zip(photos, ys, xs)):
        y0, x0 = OFF + y - MS, OFF + x - MS
        out[j] = canv[ph][y0:y0 + SS, x0:x0 + SS]
    return torch.from_numpy(out).to(dev).permute(0, 3, 1, 2).float()


def src_crops(df, idx):
    return crops_xy(df.photo.values[idx], df.row.values[idx] * HT, df.col.values[idx] * HT)


_s = src_crops(tr, np.random.choice(len(tr), 500, replace=False))
MEAN, STD = _s.mean((0, 2, 3)).view(1, 3, 1, 1), _s.std((0, 2, 3)).view(1, 3, 1, 1)
del _s
MASK = torch.zeros(1, 1, SS, SS, device=dev)
MASK[..., MS:MS + HT, MS:MS + HT] = 1


def to_input(src, theta):
    if args.no_context:  # only the 256x256 tile itself is visible (train and test alike)
        tile_mean = src[..., MS:MS + HT, MS:MS + HT].mean((2, 3), keepdim=True)
        src = torch.where(MASK > 0, src, tile_mean)
    x = torch.cat([(src - MEAN) / STD, MASK.expand(len(src), -1, -1, -1)], 1)
    grid = F.affine_grid(theta, (len(src), 4, S, S), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    x[:, 3] = x[:, 3] * 2 - 1
    return x


def make_theta(n, train, k=0, z=1.0):
    base = S / SS
    if train:
        z = torch.exp(torch.empty(n).uniform_(math.log(args.zmin), math.log(args.zmax)))
        aniso = torch.exp(torch.empty(n).uniform_(-0.15, 0.15))
        ang = torch.empty(n).uniform_(-args.rot, args.rot) * math.pi / 180 + torch.randint(0, 4, (n,)) * math.pi / 2
        flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
        t_src = torch.empty(n, 2).uniform_(-args.shift, args.shift)
        if args.fit_tile:
            # corner extent of the tile in output px = 64*z*aniso*(|cos|+|sin|) + z*|shift|; keep it inside S/2
            k = ang.cos().abs() + ang.sin().abs()
            zfit = (S / 2 - 2) / (64 * torch.maximum(aniso, 1 / aniso) * k + t_src.abs().max(1).values)
            z = torch.minimum(z, zfit)
        t = t_src * 2 / SS
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
    if torch.rand(1).item() < 0.3:
        rgb = F.avg_pool2d(rgb, 3, 1, 1)
    if args.aug_filled:
        mu = rgb.mean((1, 2, 3), keepdim=True)
        rgb = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.2, 2 * mu - rgb, rgb)
        c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.3, 1.0)
        rgb = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.3, mu + (rgb - mu) * c, rgb)
        rgb = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.2, rgb.mean(1, keepdim=True).expand_as(rgb), rgb)
    return torch.cat([rgb, x[:, 3:]], 1)


def loss_fn(logit, y):
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = y * p + (1 - y) * (1 - p)
    a = y * args.alpha + (1 - y) * (1 - args.alpha)
    return (a * (1 - pt) ** args.gamma * ce).mean()


class MILNet(torch.nn.Module):
    def __init__(self, stride8=False, pool="max", cell_cover=0.5, cell_weight="hard", lse_r=5.0):
        super().__init__()
        self.stride8, self.pool, self.cover, self.weight, self.r = stride8, pool, cell_cover, cell_weight, lse_r
        idx = (2, 3, 4) if stride8 else (3, 4)
        self.body = timm.create_model(args.arch, pretrained=False, features_only=True, out_indices=idx, in_chans=3)
        ch = sum(self.body.feature_info.channels())
        self.head = torch.nn.Sequential(torch.nn.Conv2d(ch, 128, 3, padding=1), torch.nn.BatchNorm2d(128),
                                        torch.nn.ReLU(inplace=True), torch.nn.Conv2d(128, 1, 1))

    def dense(self, rgb):
        fs = self.body(rgb)
        size = fs[0].shape[-2:]
        fs = [fs[0]] + [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in fs[1:]]
        return self.head(torch.cat(fs, 1))

    def forward(self, x):
        lm = self.dense(x[:, :3])
        stride = x.shape[-1] // lm.shape[-1]
        cov = F.avg_pool2d((x[:, 3:4] > 0).float(), stride).flatten(1)
        lm = lm.float().flatten(1)
        if self.weight == "soft":  # cells partly inside the tile count, discounted by their coverage
            inside = cov > 0
            lm = lm + torch.log(cov.clamp(min=1e-4))
        else:
            inside = cov > self.cover
        if self.pool == "max":
            out = lm.masked_fill(~inside, -1e4).max(1).values
        else:
            n = inside.sum(1).clamp(min=1).float()
            out = (torch.logsumexp(self.r * lm.masked_fill(~inside, -1e4), 1) - torch.log(n)) / self.r
        return out.unsqueeze(1)


def make_model():
    return MILNet(args.stride8, args.pool, args.cell_cover, args.cell_weight, args.lse_r).to(dev)


# ----------------------------------------------------------------------------------------------
# Re-tiling: estimate where each P-con is (pixel mask) on training photos, then cut windows at
# arbitrary offsets and label them by whether they overlap a P-con. Creates many new "P-con only
# partly inside" examples, which grid tiles alone provide only a few of.
#   * Map = stage model trained on THIS fold's training photos only (no validation leakage),
#     run fully-convolutionally over the whole photo (4 flip TTA).
#   * Mask = (map logit > 0) AND inside a positive GT tile (labels are hard constraints).
#   * Positive GT tiles the map does not cover are "unknown": windows touching them are skipped.
#   * Window label: overlap >= 32 px -> 1; no overlap with the mask dilated by 8 px -> 0; else skip.
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def build_retile(photos, load_name, fold, n_per_photo=1500):
    src = MILNet().to(dev)
    src.load_state_dict(torch.load(WORK / f"{load_name}_f{fold}.pt", map_location=dev))
    src.eval()
    cand = []
    masks = {}
    for ph in photos:
        c = torch.from_numpy(raw[ph]).to(dev).permute(2, 0, 1).float()[None]
        x = (c - MEAN) / STD
        acc = 0
        for fh in (False, True):
            for fw in (False, True):
                xi = x.flip(2) if fh else x
                xi = xi.flip(3) if fw else xi
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    m = src.dense(xi).float()
                m = m.flip(2) if fh else m
                acc = acc + (m.flip(3) if fw else m)
        lm = F.interpolate(acc / 4, size=c.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        lm = lm[HP:HP + PH_H, HP:HP + PH_W].cpu().numpy()
        d = tr[tr.photo == ph]
        pos = np.zeros((PH_H, PH_W), bool)
        known = np.zeros((PH_H, PH_W), bool)
        for r, col, lab in zip(d.row, d.col, d.label):
            sl = (slice(r * HT, (r + 1) * HT), slice(col * HT, (col + 1) * HT))
            known[sl] = True
            if lab == 1:
                pos[sl] = True
        R = (lm > 0) & pos
        unk = ~known
        for r, col, lab in zip(d.row, d.col, d.label):
            sl = (slice(r * HT, (r + 1) * HT), slice(col * HT, (col + 1) * HT))
            if lab == 1 and not R[sl].any():
                unk[sl] = True
        Rd = F.max_pool2d(torch.from_numpy(R.astype(np.float32))[None, None], 17, 1, 8)[0, 0].numpy() > 0
        masks[ph] = (R, unk)

        def integral(a):
            ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
            ii[1:, 1:] = a.astype(np.int64).cumsum(0).cumsum(1)
            return ii

        iR, iD, iU = integral(R), integral(Rd), integral(unk)

        def box(ii, y0, x0, y1, x1):
            y0, x0 = np.clip(y0, 0, PH_H), np.clip(x0, 0, PH_W)
            y1, x1 = np.clip(y1, 0, PH_H), np.clip(x1, 0, PH_W)
            return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]

        ys = np.random.randint(0, PH_H - HT + 1, n_per_photo)
        xs = np.random.randint(0, PH_W - HT + 1, n_per_photo)
        # bias half of the windows toward P-con borders: start near a mask pixel
        py, px = np.nonzero(R)
        if len(py):
            k = n_per_photo // 2
            j = np.random.randint(0, len(py), k)
            ys[:k] = np.clip(py[j] - np.random.randint(0, HT, k), 0, PH_H - HT)
            xs[:k] = np.clip(px[j] - np.random.randint(0, HT, k), 0, PH_W - HT)
        ov = box(iR, ys, xs, ys + HT, xs + HT)
        dl = box(iD, ys, xs, ys + HT, xs + HT)
        un = box(iU, ys, xs, ys + HT, xs + HT)
        outer = box(iR, ys - 48, xs - 48, ys + HT + 48, xs + HT + 48)
        lab = np.full(n_per_photo, -1)
        lab[ov >= 32] = 1
        lab[(dl == 0) & (un == 0)] = 0
        edge = (lab == 1) & (outer > ov * 1.5)
        keep = lab >= 0
        cand.append(pd_frame(ph, ys[keep], xs[keep], lab[keep], edge[keep]))
    del src
    return np.concatenate(cand), masks


def pd_frame(ph, ys, xs, lab, edge):
    a = np.empty(len(ys), dtype=[("ph", "U8"), ("y", "i4"), ("x", "i4"), ("lab", "i1"), ("edge", "?")])
    a["ph"], a["y"], a["x"], a["lab"], a["edge"] = ph, ys, xs, lab, edge
    return a


def train_fold(tri, rt):
    model = make_model()
    ema = copy.deepcopy(model).eval() if args.ema > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    spe = len(tri) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * spe, pct_start=0.1)
    pos, neg = tri[ytr[tri] == 1], tri[ytr[tri] == 0]
    n_rt = int(round(args.bs * args.retile)) if rt is not None else 0
    n_or = args.bs - n_rt
    if n_rt:
        rpos_e = np.where((rt["lab"] == 1) & rt["edge"])[0]
        rpos_c = np.where((rt["lab"] == 1) & ~rt["edge"])[0]
        rneg = np.where(rt["lab"] == 0)[0]
        np_rt = int(n_rt * args.pos_frac)
    for ep in range(args.epochs):
        model.train()
        tl = 0
        for _ in range(spe):
            npos = int(n_or * args.pos_frac)
            nneg = n_or - npos
            if args.ohem and ep >= 2:
                # Online hard-negative mining: score a 3x pool of random negatives with the current model
                # (no grad, same augmentation) and keep the highest-scoring half -> voids, fittings, stains
                cand = np.random.choice(neg, 3 * nneg)
                with torch.no_grad():
                    model.eval()
                    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                        sc = model(to_input(src_crops(tr, cand), make_theta(len(cand), True))).float().squeeze(1)
                    model.train()
                hard = cand[torch.argsort(sc, descending=True)[: nneg // 2].cpu().numpy()]
                negs = np.concatenate([hard, np.random.choice(neg, nneg - nneg // 2)])
            else:
                negs = np.random.choice(neg, nneg)
            idx = np.concatenate([np.random.choice(pos, npos), negs])
            src = src_crops(tr, idx)
            y = ytr[idx]
            if n_rt:
                ne = np_rt // 2
                j = np.concatenate([np.random.choice(rpos_e, ne), np.random.choice(rpos_c, np_rt - ne),
                                    np.random.choice(rneg, n_rt - np_rt)])
                src = torch.cat([src, crops_xy(rt["ph"][j], rt["y"][j], rt["x"][j])])
                y = np.concatenate([y, rt["lab"][j].astype(np.float32)])
            x = photometric(to_input(src, make_theta(len(src), True)))
            yt = torch.from_numpy(y).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1).float(), yt)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
                        else:
                            pe.copy_(pm)
            tl += loss.item()
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"  ep{ep + 1} loss {tl / spe:.4f}", flush=True)
    return ema if ema is not None else model


def finetune_fold(model, tri):
    """Hard-negative fine-tuning. Hard pool = training negatives of THIS fold that the loaded fold
    model scores highest (no TTA, no aug); no validation photo is touched."""
    neg_all = tri[ytr[tri] == 0]
    pos = tri[ytr[tri] == 1]
    sc = predict_raw(model, tr, neg_all)
    hard = neg_all[np.argsort(-sc)[: args.hard_pool]]
    print(f"  hard pool: {len(hard)} negs, score range {np.sort(sc)[-args.hard_pool]:.3f}-{sc.max():.3f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    spe = len(tri) // args.bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * spe)
    npos = int(args.bs * args.pos_frac)
    nrand = args.bs - npos - args.n_hard
    for ep in range(args.epochs):
        model.train()
        tl = 0
        for _ in range(spe):
            idx = np.concatenate([np.random.choice(pos, npos), np.random.choice(neg_all, nrand), np.random.choice(hard, args.n_hard)])
            x = photometric(to_input(src_crops(tr, idx), make_theta(len(idx), True)))
            yt = torch.from_numpy(ytr[idx]).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1).float(), yt)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        print(f"  ft ep{ep + 1} loss {tl / spe:.4f}", flush=True)
    return model


@torch.no_grad()
def predict_raw(model, df, idx):
    model.eval()
    out = []
    for i in range(0, len(idx), 256):
        bi = idx[i:i + 256]
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(model(to_input(src_crops(df, bi), make_theta(len(bi), False))).float()).squeeze(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def predict(model, df, idx):
    model.eval()
    scales = [float(s) for s in args.tta_scales.split(",")]
    out = []
    for i in range(0, len(idx), 128):
        bi = idx[i:i + 128]
        src = src_crops(df, bi)
        p = 0
        for z in scales:
            for k in range(8):
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    p = p + torch.sigmoid(model(to_input(src, make_theta(len(bi), False, k, z))).float()).squeeze(1)
        out.append((p / (8 * len(scales))).cpu())
    return torch.cat(out).numpy()


def best_f1(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    fs = [f1_score(y, p > t) for t in ths]
    i = int(np.argmax(fs))
    return fs[i], ths[i]


cv = StratifiedGroupKFold(args.folds, shuffle=True, random_state=0)
splits = list(cv.split(tr, ytr, tr.photo))

if args.debug_retile:
    from PIL import Image, ImageDraw
    tri, _ = splits[0]
    photos = sorted(set(tr.photo.values[tri]))
    rt, masks = build_retile(photos, args.retile_from, 0, n_per_photo=1500)
    print("candidates:", len(rt), "pos", (rt["lab"] == 1).sum(), "edge pos", rt["edge"].sum(),
          "neg", (rt["lab"] == 0).sum())
    for ph in [p for p in ["7131", "7078", "7121", "7099", "7135", "7082"] if p in masks][:4]:
        R, unk = masks[ph]
        img = raw[ph][HP:HP + PH_H, HP:HP + PH_W].copy()
        img[R] = (0.5 * img[R] + [127, 0, 0]).astype(np.uint8)
        img[unk & ~R] = (0.6 * img[unk & ~R] + [0, 0, 100]).astype(np.uint8)
        im = Image.fromarray(img).resize((PH_W // 2, PH_H // 2))
        dr = ImageDraw.Draw(im)
        for r, cc in zip(*np.nonzero(np.isin(np.arange(NROW * NCOL).reshape(NROW, NCOL) + 1,
                                             tr[(tr.photo == ph) & (tr.label == 1)].idx.values))):
            dr.rectangle([cc * 64, r * 64, cc * 64 + 63, r * 64 + 63], outline=(0, 255, 0))
        im.save(WORK / f"debug_retile_mask_{ph}.jpg")
    sel = np.concatenate([np.where((rt["lab"] == 1) & rt["edge"])[0][:12], np.where(rt["lab"] == 0)[0][:4]])
    x = to_input(crops_xy(rt["ph"][sel], rt["y"][sel], rt["x"][sel]), make_theta(len(sel), False))
    rgb = (x[:, :3] * STD + MEAN).clamp(0, 255)
    ed = (x[:, 3:] > 0).float()
    ed = ed - F.max_pool2d(-ed, 3, 1, 1).neg()
    rgb[:, 1:2] = torch.where(ed.abs() > 0, torch.full_like(rgb[:, :1], 255), rgb[:, 1:2])
    rows = [torch.cat(list(rgb[i:i + 8]), 2) for i in range(0, len(sel), 8)]
    Image.fromarray(torch.cat(rows, 1).permute(1, 2, 0).byte().cpu().numpy()).save(WORK / "debug_retile_windows.jpg")
    raise SystemExit

oof = np.full(len(tr), np.nan)
pte = np.zeros(len(te))
nf = 0
for f, (tri, vai) in enumerate(splits):
    if args.only_fold >= 0 and f != args.only_fold:
        continue
    t0 = time.time()
    if args.predict_only or args.finetune:
        model = make_model()
        model.load_state_dict(torch.load(WORK / f"{args.load}_f{f}.pt", map_location=dev))
        if args.finetune:
            model = finetune_fold(model, tri)
            torch.save(model.state_dict(), WORK / f"{args.name}_f{f}.pt")
    else:
        rt = None
        if args.retile > 0:
            rt, _ = build_retile(sorted(set(tr.photo.values[tri])), args.retile_from, f)
            print(f"  retile fold {f}: {len(rt)} windows, pos {(rt['lab'] == 1).sum()} (edge {rt['edge'].sum()}), "
                  f"neg {(rt['lab'] == 0).sum()}", flush=True)
        model = train_fold(tri, rt)
        torch.save(model.state_dict(), WORK / f"{args.name}_f{f}.pt")
    oof[vai] = predict(model, tr, vai)
    pte += predict(model, te, np.arange(len(te)))
    nf += 1
    fb, tb = best_f1(ytr[vai], oof[vai])
    print(f"fold {f}: auc {roc_auc_score(ytr[vai], oof[vai]):.4f} ap {average_precision_score(ytr[vai], oof[vai]):.4f} "
          f"f1@0.5 {f1_score(ytr[vai], oof[vai] > 0.5):.4f} best f1 {fb:.4f}@{tb:.2f}  ({time.time() - t0:.0f}s)", flush=True)
pte /= nf

ok = ~np.isnan(oof)
fb, tb = best_f1(ytr[ok], oof[ok])
res = dict(name=args.name, auc=roc_auc_score(ytr[ok], oof[ok]), ap=average_precision_score(ytr[ok], oof[ok]),
           f1_05=f1_score(ytr[ok], oof[ok] > 0.5), best_f1=fb, best_th=tb)
print(json.dumps(res))
np.save(WORK / f"oof_{args.name}.npy", oof)
import sklearn
meta = dict(args=vars(args), result=res, sklearn=sklearn.__version__, torch=torch.__version__,
            mean=MEAN.flatten().tolist(), std=STD.flatten().tolist(),
            folds=[sorted(set(tr.photo.values[v])) for _, v in splits])
(WORK / f"meta_{args.name}.json").write_text(json.dumps(meta, indent=1, default=str))
np.save(WORK / f"test_{args.name}.npy", pte)
