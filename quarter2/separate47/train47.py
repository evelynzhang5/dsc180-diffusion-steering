"""Round 47 (GPU): train the layer separator. FROZEN before training.

Backgrounds (training only; none of round 46's six test backgrounds, no transparent-object RealBokeh scene):
  round 40 inputs (outputs/q2_compare40/data/input), round 45 inputs except the 6 round 46 backgrounds
  (outputs/q2_confirm45/data/input), and RealBokeh_3MP test-split f/22 sources; RealBokeh ids 48, 49, 51, 116, 129,
  131 excluded everywhere. Each background is resized to a 512-px short side.
Samples: random scale (crop side 256/s, s in [0.5, 1]) and flip, random pane, 1-2 random TRAIN families (synth47), composite
  in linear light, quantised to 8-bit sRGB (as a real input).
Model: U-Net (32-64-128-256-256 channels, GroupNorm, SiLU, 4 down/up levels).
  full    input sRGB composite + pane mask (privileged region) -> alpha (sigmoid), F, B (sigmoid, linear light)
  oracle  input sRGB composite + pane mask + true alpha     -> F, B
Loss: |alpha_hat - alpha| + |B_hat - B| (weight 1 in the pane, 0.1 outside) + alpha-weighted |F_hat - F|
      + 0.5 |alpha F_hat + (1 - alpha) B_hat - I| (alpha_hat for full, true alpha for oracle); means over the batch.
Optimiser: AdamW 2e-4, cosine to 0, batch 16, STEPS steps (8,000; SMOKE 30), seed 0.
usage: python train47.py <variant full|oracle> <out .pt>      env: SMOKE, R40_IN, R45_IN, RB_IN (dir of f/22 jpgs)
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth47 as S  # noqa: E402

TEST_BGS = {"f_fur_0", "f_hair_3", "f_highlights_0", "f_highlights_2", "f_thin_1", "f_hair_5"}
RB_EXCLUDE = {48, 49, 51, 116, 129, 131}
CROP = 256


def background_paths():
    p = [f for f in glob.glob(f"{os.environ['R40_IN']}/*.png")
         if not (os.path.basename(f).startswith("rb_") and int(os.path.basename(f)[3:-4]) in RB_EXCLUDE)]
    p += [f for f in glob.glob(f"{os.environ['R45_IN']}/*.png") if os.path.basename(f)[:-4] not in TEST_BGS]
    p += [f for f in glob.glob(f"{os.environ['RB_IN']}/*_f22.JPG") if int(os.path.basename(f).split("_")[0]) not in RB_EXCLUDE]
    return sorted(p)


def load_bg(path):
    im = Image.open(path).convert("RGB")
    s = 512 / min(im.size)
    return np.asarray(im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS))


class Synth(torch.utils.data.IterableDataset):
    def __init__(self, bgs, seed):
        self.bgs, self.seed = bgs, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        while True:
            bg = self.bgs[rng.integers(len(self.bgs))]
            side = int(CROP / rng.uniform(0.5, 1.0))
            side = min(side, *bg.shape[:2])
            y, x = rng.integers(0, bg.shape[0] - side + 1), rng.integers(0, bg.shape[1] - side + 1)
            b = Image.fromarray(bg[y:y + side, x:x + side]).resize((CROP, CROP), Image.LANCZOS)
            b = np.asarray(b) / 255.0
            if rng.random() < 0.5:
                b = b[:, ::-1]
            B = S.to_lin(b)
            pane = S.random_pane(rng, CROP, CROP)
            alpha, Fs = S.random_surface(rng, pane, S.TRAIN_FAMILIES)
            I = alpha[..., None] * Fs + (1 - alpha[..., None]) * B
            Isrgb = np.round(S.to_srgb(I) * 255) / 255.0
            I = S.to_lin(Isrgb)
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
            yield (t(Isrgb).permute(2, 0, 1), t(pane)[None], t(alpha)[None], t(Fs).permute(2, 0, 1),
                   t(B).permute(2, 0, 1), t(I).permute(2, 0, 1))


def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin, cout, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.enc = nn.ModuleList()
        c = cin
        for co in ch:
            self.enc.append(block(c, co))
            c = co
        self.dec = nn.ModuleList()
        for k in range(len(ch) - 1, 0, -1):
            self.dec.append(block(ch[k] + ch[k - 1], ch[k - 1]))
        self.head = nn.Conv2d(ch[0], cout, 1)

    def forward(self, x):
        skips = []
        for k, e in enumerate(self.enc):
            x = e(x if k == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for k, d in enumerate(self.dec):
            s = skips[-2 - k]
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.head(x)


def make_model(variant):
    return UNet(5 if variant == "oracle" else 4, 6 if variant == "oracle" else 7)


def predict(model, variant, Isrgb, pane, alpha=None):
    """Isrgb (N,3,H,W) in [0,1]; returns alpha_hat, F_hat, B_hat (linear light)."""
    x = torch.cat([Isrgb, pane] + ([alpha] if variant == "oracle" else []), 1)
    o = model(x)
    if variant == "oracle":
        return alpha, torch.sigmoid(o[:, 0:3]), torch.sigmoid(o[:, 3:6])
    return torch.sigmoid(o[:, 0:1]), torch.sigmoid(o[:, 1:4]), torch.sigmoid(o[:, 4:7])


if __name__ == "__main__":
    variant, out = sys.argv[1], sys.argv[2]
    smoke = os.environ.get("SMOKE", "0") == "1"
    steps = 30 if smoke else 8000
    torch.manual_seed(0)
    paths = background_paths()
    print(f"{len(paths)} training backgrounds", flush=True)
    bgs = [load_bg(p) for p in (paths[:20] if smoke else paths)]
    dl = torch.utils.data.DataLoader(Synth(bgs, 0), batch_size=16, num_workers=6)
    model = make_model(variant).cuda()
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0, run = time.time(), []
    for it, (Isrgb, pane, alpha, Fs, B, I) in enumerate(dl):
        if it >= steps:
            break
        Isrgb, pane, alpha, Fs, B, I = [v.cuda(non_blocking=True) for v in (Isrgb, pane, alpha, Fs, B, I)]
        ah, Fh, Bh = predict(model, variant, Isrgb, pane, alpha)
        la = (ah - alpha).abs().mean() if variant == "full" else torch.zeros((), device="cuda")
        lb = ((Bh - B).abs() * (pane + 0.1)).mean()
        lf = ((Fh - Fs).abs() * alpha).sum() / alpha.sum().clamp_min(1.0) / 3
        lr = (ah * Fh + (1 - ah) * Bh - I).abs().mean()
        loss = la + lb + lf + 0.5 * lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run.append([loss.item(), la.item(), lb.item(), lf.item(), lr.item()])
        if (it + 1) % (10 if smoke else 500) == 0:
            m = np.mean(run[-500:], 0)
            print(f"{variant} step {it + 1} loss {m[0]:.4f} alpha {m[1]:.4f} B {m[2]:.4f} F {m[3]:.4f} rec {m[4]:.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
    torch.save(dict(variant=variant, state=model.state_dict(), steps=steps, n_backgrounds=len(paths),
                    final=np.mean(run[-200:], 0).tolist()), out)
    print("saved", out, flush=True)
