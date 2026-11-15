"""Q2 round 14: two first tests after the steering approach stopped (round 13). Run on a GPU pod:

    python quarter2/optical_operator/optical.py operator     # hypothesis 1: predict a blur operator on source pixels
    python quarter2/optical_operator/optical.py stereotype   # hypothesis 2: does a blur request change the scene type?

operator    Synthetic layered scenes with known geometry, built from real photos (round 4's depth-of-field set; photos
            split 80/20 into train and test). A foreground layer (a smooth blob plus thin hair-like strands, alpha
            known, texture from another photo) in focus over a background plane whose blur radius varies smoothly
            (0.6-1.0 x the level). The target is rendered independently with the full background layer, which no
            method sees: T = a F + (1 - a) DiskBlur(B, r). Every method gets only the composite I, the blur-radius map
            and the protected mask m = (a > 0.5), and is identity on m:
              gather      classical depth-dependent disk blur of I (foreground colours leak into the background)
              normconv    classical blur that excludes protected pixels (normalised convolution)
              image       small U-Net predicting the output pixels (residual on I)
              operator    the same U-Net predicting a 17 x 17 softmax kernel per pixel, applied to I's own pixels
            The two learned models share the backbone and training (L1 outside m, 4000 steps); only the head differs.
            Scored on held-out photos at blur levels 2, 4, 8 px by region of the true geometry: protected, thin
            strands outside m, boundary band (within level + 2 px of m), background interior.
stereotype  SD 1.5 (round 4's sampler) on four scenes that break photographic conventions, 4 seeds, three conditions:
            none, prompt (", shallow depth of field, blurred bokeh background" as a third branch, s 2) and the paper's
            RFM direction (global, s 4). Measures: background blur reached; size of the small subject; a person
            appearing in an empty street; CLIP preference for 'a close-up object in the foreground'; DINOv2 to the
            unsteered image.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "quarter2" / "diagnostics"))
import diag as G  # noqa: E402  (round 12: Seg, overlay; puts rounds 11, 10 and 4 on the path)

D, M, P = G.D, G.M, G.P
OUT = ROOT / "outputs" / "q2_optical_operator"
SZ, K, RMAX = 128, 17, 8
LEVELS = [2.0, 4.0, 8.0]
log = G.log
dev = "cuda"


# ---------------------------------------------------------------- synthetic layered scenes
def disk(r):
    """Anti-aliased disk kernel of radius r (K x K), normalised; r = 0 is the identity."""
    y, x = torch.meshgrid(torch.arange(K) - K // 2, torch.arange(K) - K // 2, indexing="ij")
    w = (r + 0.5 - (x ** 2 + y ** 2).float().sqrt()).clamp(0, 1)
    return w / w.sum()


DISKS = None


def blur_var(x, r):
    """Spatially varying disk blur: x (n, c, h, w), r (n, 1, h, w) in px; interpolates integer-radius blurs."""
    global DISKS
    if DISKS is None:
        DISKS = torch.stack([disk(float(k)) for k in range(RMAX + 1)]).to(x.device)[:, None]
    n, c, h, w = x.shape
    xp = F.pad(x.reshape(n * c, 1, h, w), (K // 2,) * 4, mode="reflect")
    st = F.conv2d(xp, DISKS).reshape(n, c, RMAX + 1, h, w)
    r = r.clamp(0, RMAX)
    i0 = r.floor().long().clamp(max=RMAX - 1)
    fr = (r - i0).unsqueeze(2)
    g0 = st.gather(2, i0.unsqueeze(2).expand(n, c, 1, h, w))
    g1 = st.gather(2, (i0 + 1).unsqueeze(2).expand(n, c, 1, h, w))
    return ((1 - fr) * g0 + fr * g1).squeeze(2)


def gblur(x, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-(torch.arange(k, device=x.device) - k // 2).float() ** 2 / (2 * s * s))
    g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))


def scene(photos, n, level, gen):
    """n scenes at a blur level (None: random 1-8). Returns I, T, r, m, alpha, strands (all (n, ., SZ, SZ))."""
    N, _, H, W = photos.shape
    pick = lambda: torch.randint(0, N, (n,), generator=gen).tolist()

    def crop(idx):
        ys = torch.randint(0, H - SZ + 1, (n,), generator=gen).tolist()
        xs = torch.randint(0, W - SZ + 1, (n,), generator=gen).tolist()
        return torch.stack([photos[i, :, y:y + SZ, x:x + SZ] for i, y, x in zip(idx, ys, xs)])
    B, Fg = crop(pick()), crop(pick())
    # blob: smoothed noise plus a centre bias, thresholded to 12-35% of the area
    noise = gblur(torch.randn(n, 1, SZ, SZ, generator=gen).to(dev), 10.0)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, SZ), torch.linspace(-1, 1, SZ), indexing="ij")
    cy, cx = (torch.rand(n, generator=gen) - 0.5) * 0.6, (torch.rand(n, generator=gen) - 0.5) * 0.6
    rad = ((yy[None] - cy.view(-1, 1, 1)) ** 2 + (xx[None] - cx.view(-1, 1, 1)) ** 2).sqrt().to(dev)[:, None]
    f = noise / noise.flatten(1).std(1).view(-1, 1, 1, 1) - 2.5 * rad
    frac = 0.12 + 0.23 * torch.rand(n, generator=gen).to(dev)
    thr = torch.stack([torch.quantile(f[i].flatten(), 1 - frac[i]) for i in range(n)]).view(-1, 1, 1, 1)
    blob = gblur((f > thr).float(), 0.7)
    # strands: thin curves leaving the blob, ~1 px wide (points drawn on the CPU, one scatter on the GPU)
    blob_c = (blob[:, 0] > 0.5).cpu()
    idx = []
    for i in range(n):
        ys, xs = torch.nonzero(blob_c[i], as_tuple=True)
        if len(ys) == 0:
            continue
        k = int(torch.randint(6, 20, (1,), generator=gen))
        j = torch.randint(0, len(ys), (k,), generator=gen)
        p0 = torch.stack([ys[j], xs[j]], 1).float()                       # (k, 2)
        ang = torch.rand(k, generator=gen) * 6.283
        L = 15 + 30 * torch.rand(k, generator=gen)
        p2 = p0 + L[:, None] * torch.stack([torch.sin(ang), torch.cos(ang)], 1)
        p1 = (p0 + p2) / 2 + (torch.rand(k, 2, generator=gen) - 0.5) * L[:, None]
        tt = torch.linspace(0, 1, 200).view(1, -1, 1)
        pts = ((1 - tt) ** 2 * p0[:, None] + 2 * (1 - tt) * tt * p1[:, None] + tt ** 2 * p2[:, None]).round().long()
        pts = pts.reshape(-1, 2)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < SZ) & (pts[:, 1] >= 0) & (pts[:, 1] < SZ)]
        idx.append(torch.cat([torch.full((len(pts), 1), i), pts], 1))
    canvas = torch.zeros(n, 1, SZ, SZ, device=dev)
    if idx:
        ix = torch.cat(idx).to(dev)
        canvas[ix[:, 0], 0, ix[:, 1], ix[:, 2]] = 1.0
    strands = (gblur(canvas, 0.5) * 2.5).clamp(0, 1)
    alpha = torch.maximum(blob, strands)
    # background blur: a tilted plane, radius in [0.6, 1.0] x level
    lv = (1 + 7 * torch.rand(n, generator=gen)) if level is None else torch.full((n,), level)
    ang = torch.rand(n, generator=gen) * 6.283
    ramp = (torch.cos(ang).view(-1, 1, 1) * xx[None] + torch.sin(ang).view(-1, 1, 1) * yy[None] + 1.5) / 3
    r = (lv.view(-1, 1, 1) * (0.6 + 0.4 * ramp)).to(dev)[:, None].float()
    I = alpha * Fg + (1 - alpha) * B
    T = alpha * Fg + (1 - alpha) * blur_var(B, r)        # uses the hidden background: independent of every method
    m = (alpha > 0.5).float()
    return I, T, r, m, alpha, strands


# ---------------------------------------------------------------- methods
def gather(I, r, m):
    return m * I + (1 - m) * blur_var(I, r)


def normconv(I, r, m):
    num, den = blur_var(I * (1 - m), r), blur_var(1 - m, r)
    return m * I + (1 - m) * torch.where(den > 1e-3, num / den.clamp(min=1e-3), I)


class UNet(nn.Module):
    def __init__(self, out_ch, w=32):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        self.e1, self.e2, self.e3 = c(5, w), c(w, 2 * w), c(2 * w, 4 * w)
        self.d2, self.d1 = c(6 * w, 2 * w), c(3 * w, w)
        self.head = nn.Conv2d(w, out_ch, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        b = self.d2(torch.cat([b, F.interpolate(c, scale_factor=2)], 1))
        return self.head(self.d1(torch.cat([a, F.interpolate(b, scale_factor=2)], 1)))


class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(3)

    def forward(self, I, r, m):
        out = I + self.net(torch.cat([I, r / RMAX, m], 1))
        return m * I + (1 - m) * out


class OperatorNet(nn.Module):
    """Per-pixel K x K softmax weights applied to the source's own pixels: a convex combination, no new colours."""

    def __init__(self):
        super().__init__()
        self.net = UNet(K * K)

    def forward(self, I, r, m, return_w=False):
        n, _, h, w = I.shape
        W = self.net(torch.cat([I, r / RMAX, m], 1)).softmax(1)                 # (n, K*K, h, w)
        patches = F.unfold(F.pad(I, (K // 2,) * 4, mode="reflect"), K).view(n, 3, K * K, h, w)
        out = (patches * W[:, None]).sum(2)
        out = m * I + (1 - m) * out
        return (out, W) if return_w else out


def load_photos():
    imgs, labels, names = D.dof_photos(n_per_class=300)
    x = torch.stack([torch.from_numpy(np.asarray(im.resize((256, 256), Image.BICUBIC))).permute(2, 0, 1)
                     for im in imgs]).float() / 255
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    k = int(0.8 * len(x))
    return x[perm[:k]].to(dev), x[perm[k:]].to(dev)


def regions(m, alpha, strands, level):
    """Masks from the true geometry: protected, thin strands outside m, boundary band, background interior."""
    from scipy.ndimage import distance_transform_edt
    out = {k: [] for k in ("protected", "thin", "boundary", "background")}
    for i in range(len(m)):
        mm = m[i, 0].cpu().numpy() > 0.5
        dist = distance_transform_edt(~mm)
        thin = (strands[i, 0].cpu().numpy() > 0.05) & ~mm
        out["protected"].append(mm)
        out["thin"].append(thin)
        out["boundary"].append(~mm & ~thin & (dist <= level + 2))
        out["background"].append(~mm & ~thin & (dist > level + 2) & (alpha[i, 0].cpu().numpy() < 0.01))
    return {k: np.stack(v) for k, v in out.items()}


def operator(cfg):
    out = OUT / "operator"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train, test = load_photos()
    log(f"photos: {len(train)} train, {len(test)} test")
    models = {"image": ImageNet().to(dev), "operator": OperatorNet().to(dev)}
    params = {k: sum(p.numel() for p in v.parameters()) for k, v in models.items()}
    hist = {}
    for name, net in models.items():
        torch.manual_seed(0)
        gen = torch.Generator().manual_seed(1)              # same training scenes for both
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
        ts, hist[name] = time.time(), []
        for step in range(cfg.steps):
            with torch.no_grad():
                I, T, r, m, _, _ = scene(train, cfg.bs, None, gen)
            loss = ((net(I, r, m) - T).abs() * (1 - m)).sum() / ((1 - m).sum() * 3)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == cfg.steps - 1:
                hist[name].append((step, float(loss)))
                log(f"{name} step {step}: L1 {float(loss):.4f}")
        hist[name + "_seconds"] = time.time() - ts
        net.eval()

    # evaluation on held-out photos, fixed scenes per level
    res, sheet_rows, speed = {}, [], {}
    for L in LEVELS:
        gen = torch.Generator().manual_seed(100 + int(L))
        acc = {k: {q: [] for q in ("protected", "thin", "boundary", "background", "all")} for k in
               ["gather", "normconv", "image", "operator"]}
        gaps = {k: [] for k in acc}
        for b in range(cfg.test_batches):
            I, T, r, m, alpha, strands = scene(test, 16, L, gen)
            with torch.no_grad():
                outs = {"gather": gather(I, r, m), "normconv": normconv(I, r, m),
                        "image": models["image"](I, r, m), "operator": models["operator"](I, r, m)}
            R = regions(m, alpha, strands, L)
            err = {k: (o - T).abs().mean(1).cpu().numpy() for k, o in outs.items()}
            for k, e in err.items():
                for q, mk in R.items():
                    if mk.any():
                        acc[k][q].append(float(e[mk].mean()))
                acc[k]["all"].append(float(e.mean()))
                # matched blur: background-interior log Laplacian variance vs the target's
                lap = lambda x: F.conv2d(x.mean(1, keepdim=True), torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                         dtype=torch.float32, device=dev).view(1, 1, 3, 3), padding=1)[:, 0].cpu().numpy()
                lo, lt = lap(outs[k]), lap(T)
                for i in range(len(I)):
                    bg = R["background"][i]
                    if bg.sum() > 100:
                        gaps[k].append(float(np.log(lo[i][bg].var() + 1e-8) - np.log(lt[i][bg].var() + 1e-8)))
            if b == 0:
                for i in range(2 if L == LEVELS[-1] else 1):
                    to = lambda x: Image.fromarray((x[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round()
                                                   .astype(np.uint8)).resize((256, 256), Image.NEAREST)
                    sheet_rows.append([to(I), to(T)] + [to(outs[k]) for k in ("gather", "normconv", "image", "operator")])
        res[L] = {k: {q: float(np.mean(v)) * 255 for q, v in acc[k].items()} | {"blur_gap": float(np.mean(gaps[k]))}
                  for k in acc}
    # speed at 512 x 512
    I, T, r, m, _, _ = scene(test, 1, 4.0, torch.Generator().manual_seed(9))
    I, r, m = (F.interpolate(x, size=512) for x in (I, r, m))
    for k, fn in {"gather": gather, "normconv": normconv, "image": models["image"], "operator": models["operator"]}.items():
        with torch.no_grad():
            fn(I, r, m)
            torch.cuda.synchronize()
            ts = time.time()
            for _ in range(5):
                fn(I, r, m)
            torch.cuda.synchronize()
        speed[k] = (time.time() - ts) / 5 * 1000
    n_sheet = P.sheet(sheet_rows, out / "operator_examples.png", W=128)
    (out / "operator.json").write_text(json.dumps(dict(res={str(k): v for k, v in res.items()}, params=params,
                                                       hist=hist, speed_ms=speed), indent=1))
    meths = ["gather", "normconv", "image", "operator"]
    Lines = ["# Round 14 hypothesis 1: predicting a blur operator on the source pixels (synthetic layered scenes; "
             "decisive first test)",
             f"Train {len(train)} / test {len(test)} photos; {cfg.steps} steps x batch {cfg.bs}; test "
             f"{cfg.test_batches * 16} scenes per level. Mean absolute error x 255 against the independently rendered "
             "target, by region of the true geometry. Blur gap: background log Laplacian variance minus the "
             "target's (0 = matched blur, > 0 = not blurred enough). Parameters: "
             + ", ".join(f"{k} {v:,}" for k, v in params.items()) + ". Inference at 512 x 512 (ms): "
             + ", ".join(f"{k} {v:.1f}" for k, v in speed.items()) + ". `operator_examples.png` columns: input, target, "
             + ", ".join(meths) + ".", "",
             "| level (px) | method | protected | thin strands | boundary band | background | all | blur gap |",
             "|---|---|---|---|---|---|---|---|"]
    for L in LEVELS:
        for k in meths:
            a = res[L][k]
            Lines.append(f"| {L:g} | {k} | {a['protected']:.2f} | {a['thin']:.2f} | {a['boundary']:.2f} | "
                         f"{a['background']:.2f} | {a['all']:.2f} | {a['blur_gap']:+.3f} |")
    wins, ok_thin, ok_bg = 0, 0, 0
    for L in LEVELS:
        best = min(res[L][k]["boundary"] for k in ("gather", "normconv", "image"))
        wins += res[L]["operator"]["boundary"] <= 0.9 * best
        ok_thin += res[L]["operator"]["thin"] <= 1.05 * min(res[L][k]["thin"] for k in ("gather", "normconv", "image"))
        ok_bg += res[L]["operator"]["background"] <= 1.1 * min(res[L][k]["background"] for k in ("gather", "normconv",
                                                                                                  "image"))
    go = wins >= 2 and ok_thin >= 2 and ok_bg >= 2
    Lines += ["", "## Decision (frozen in README)", "",
              f"- Operator boundary error <= 0.9 x the best other method: {wins}/3 levels.",
              f"- Thin strands no worse than the best other (within 5%): {ok_thin}/3; background within 10%: {ok_bg}/3.",
              f"- **{'Continue: the operator improves boundaries at matched blur without losing detail.' if go else 'Do not continue as framed: no boundary advantage over the classical renderers and the image model.'}**",
              f"\nTotal time {time.time() - t0:.0f} s."]
    checks = [
        ("identity on the protected region: every method has the same protected error",
         all(max(res[L][k]["protected"] for k in meths) - min(res[L][k]["protected"] for k in meths) < 1e-3
             for L in LEVELS), ", ".join(f"{res[L]['gather']['protected']:.3f}" for L in LEVELS)),
        ("classical renderers leave a matched blur (|blur gap| of normconv < 0.5 at every level)",
         all(abs(res[L]["normconv"]["blur_gap"]) < 0.5 for L in LEVELS),
         ", ".join(f"{res[L]['normconv']['blur_gap']:+.2f}" for L in LEVELS)),
        ("training loss fell for both models", all(hist[k][-1][1] < hist[k][0][1] for k in models),
         ", ".join(f"{k} {hist[k][0][1]:.4f} -> {hist[k][-1][1]:.4f}" for k in models)),
        ("parameter counts within 5%", abs(params["operator"] - params["image"]) / params["image"] < 0.05,
         ", ".join(f"{k} {v:,}" for k, v in params.items())),
        ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}"),
    ]
    if not P.write(out / "operator_checks.md", Lines, checks):
        sys.exit("operator: some checks failed")


# ---------------------------------------------------------------- hypothesis 2: stereotype check on SD 1.5
PROMPTS = [
    ("landscape", "A wide mountain landscape with a calm lake", None),
    ("small_bird", "A small bird standing far away on a long empty beach", "small bird"),
    ("sharp_wildlife", "A deer standing in a forest, the whole forest in sharp focus", "deer"),
    ("empty_street", "An empty city street at dawn with no people", "person"),
]
SEEDS = [42, 123, 256, 789]


def stereotype():
    out = OUT / "stereotype"
    (out / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sd, meter, dino, seg = D.SD("cuda"), D.DofMeter("cuda"), M.Dino("cuda"), G.Seg("cuda")
    d = torch.load(M.DIRS)["rfm_eig1"]
    conds = [("none", "none", 0.0), ("prompt", "prompt", 2.0), ("rfm", "rfm", 4.0)]
    from transformers import CLIPModel, CLIPProcessor
    texts = ["a photo with a prominent close-up object in the foreground", "a wide photo with nothing in the foreground"]

    @torch.no_grad()
    def fg_pref(img):
        x = meter.cproc(text=texts, images=img, return_tensors="pt", padding=True).to(dev)
        lg = meter.clip(**x).logits_per_image[0]
        return float(lg[0] - lg[1])

    def area(img, noun):
        if noun is None:
            return float("nan")
        G.SEG_THR = 0.3
        m, _ = seg(img, noun)
        G.SEG_THR = 0.4
        return float(m.mean())

    rows, sheet_rows = [], []
    for kind, cap, noun in PROMPTS:
        row_imgs = []
        for seed in SEEDS:
            ims = {c: sd.generate([cap], [seed], mode, s, d if mode == "rfm" else None, bs=1)[0] for c, mode, s in conds}
            base = meter(ims["none"], cap)
            e0 = dino(ims["none"])
            for c, im in ims.items():
                im.save(out / "images" / f"{kind}_{seed}_{c}.png")
                mm = meter(im, cap)
                mm.pop("fg_mask")
                rows.append(dict(kind=kind, seed=seed, cond=c, bg_change=mm["log_bg"] - base["log_bg"],
                                 clip_blur=mm["clip_blur"], fg_pref=fg_pref(im), area=area(im, noun),
                                 dino=float(e0 @ dino(im))))
            if seed == SEEDS[0]:
                row_imgs = [ims[c] for c, _, _ in conds]
        sheet_rows.append(row_imgs)
        log(f"{kind} done")
    n_sheet = P.sheet(sheet_rows, out / "stereotype_examples.png")
    (out / "stereotype.json").write_text(json.dumps(rows, indent=1))
    L = ["# Round 14 hypothesis 2: does a blur request change the kind of scene? (SD 1.5, 4 scenes x 4 seeds; "
         "existence check)",
         "Changes vs the unsteered image of the same seed. area = CLIPSeg area of the named subject (person for the "
         "empty street: should stay 0). fg pref = CLIP logit 'close-up object in the foreground' minus 'nothing in "
         "the foreground'. `stereotype_examples.png`: rows = scenes, columns = none, prompt, rfm (seed 42).", "",
         "| scene | condition | bg sharpness change | CLIP blur score | fg pref | subject area (x1000) | DINOv2 |",
         "|---|---|---|---|---|---|---|"]
    for kind, _, _ in PROMPTS:
        for c, _, _ in conds:
            rs = [r for r in rows if r["kind"] == kind and r["cond"] == c]
            mean = lambda k: float(np.nanmean([r[k] for r in rs])) if any(np.isfinite(r[k]) for r in rs) else float("nan")
            L.append(f"| {kind} | {c} | {mean('bg_change'):+.3f} | {mean('clip_blur'):+.2f} | {mean('fg_pref'):+.2f} | "
                     f"{1000 * mean('area'):.1f} | {mean('dino'):.3f} |")
    L.append(f"\nTotal time {time.time() - t0:.0f} s.")
    checks = [("all generations measured", len(rows) == len(PROMPTS) * len(SEEDS) * len(conds), f"{len(rows)}"),
              ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}")]
    if not P.write(out / "stereotype_checks.md", L, checks):
        sys.exit("stereotype: some checks failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["operator", "stereotype"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--test_batches", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    operator(a) if a.stage == "operator" else stereotype()


if __name__ == "__main__":
    main()
