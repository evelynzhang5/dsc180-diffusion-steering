"""Q2 round 15: where do single-depth and layered bokeh renderers fail on real photos? Run on a GPU pod:

    python quarter2/real_bokeh/bench.py setup     # EBB! zip, MPIB code + checkpoints (patched: no cupy / numba)
    python quarter2/real_bokeh/bench.py select    # CLIP picks thin-structure and glass scenes, plus random controls
    python quarter2/real_bokeh/bench.py subsetA   # real pairs: four renderers vs the real f/1.8 photo, by region
    python quarter2/real_bokeh/bench.py subsetB   # additive glass reflection: synthetic exact truth + EBB! glass scenes

Renderers (all use MPIB's renderer, kernels, gamma 2.2, and the same focus disparity and blur K per scene):
  gather   classical single-depth blur: each pixel takes the blur of its own depth bin (no occlusion handling)
  hard     layered scatter with one depth per pixel (one-hot layers from the disparity map; 2.5D)
  soft     MPIB's learned soft layers (alpha extends behind occluders), no inpainted background
  mpib     full MPIB: soft layers plus LaMa-inpainted hidden background
Focus disparity and K are fitted per scene to the real target with `hard` at quarter resolution (a grid of 8 x 6),
then used for every renderer, so the comparison is about the representation, not the settings.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof"))
import sd_dof as D  # noqa: E402  (round 4: model names, log)

DL = ROOT / "downloads"
EBB_URL = "https://download.ai-benchmark.com/s/ZwN65gKQnFe5LnR/download/Bokeh_Simulation_Dataset.zip"
EBB = DL / "ebb" / "Bokeh_Simulation_Dataset.zip"
MPIB = DL / "mpib"
MPIB_CKPT = "https://drive.google.com/drive/folders/13t9M4s9CnAYrjArhrLfJtQAH80Cnn2kF"
OUT = ROOT / "outputs" / "q2_real_bokeh"
SHORT = 512
DFS = np.linspace(0.3, 1.0, 8)
KS = [8, 16, 24, 32, 48, 64]           # at SHORT; halved for the quarter-resolution fit
GAMMA = 2.2
N_SCAN = 1500
PROMPTS = {"hair": "a portrait of a person with fine flyaway hair",
           "branches": "thin bare tree branches and twigs",
           "fence": "a wire fence or metal mesh",
           "foliage": "thin grass stems and small plant leaves",
           "glass": "a reflection in a glass window or shop window"}
NEUTRAL = "a plain photo of large smooth simple objects"
PER_CAT, N_CTRL = 8, 20
log = D.log
dev = "cuda"


# ---------------------------------------------------------------- setup
def setup():
    (DL / "ebb").mkdir(parents=True, exist_ok=True)
    if not EBB.exists() or EBB.stat().st_size < 3.7e9:
        log("downloading EBB! (3.8 GB)")
        subprocess.run(["curl", "-sL", "-o", str(EBB), EBB_URL], check=True)
    log(f"EBB! zip {EBB.stat().st_size / 1e9:.2f} GB")
    if not MPIB.exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/JuewenPeng/MPIB.git", str(MPIB)], check=True)
    enc = MPIB / "models" / "encoder.py"
    t = enc.read_text()
    if "resnets[num_layers](pretrained)" in t:              # torchvision >= 0.13 has keyword-only weights
        enc.write_text(t.replace("resnets[num_layers](pretrained)", "resnets[num_layers]()"))
    ck = MPIB / "checkpoints"
    if not all((ck / f).exists() for f in ("mpinet.pth", "aupnet.pth", "inpnet.pth")):
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q", "gdown"], check=True)
        subprocess.run([sys.executable, "-m", "gdown", "--folder", MPIB_CKPT, "-O", str(ck)], check=True)
        for f in ck.rglob("*.pth"):                          # gdown may nest a subfolder
            if f.parent != ck:
                f.rename(ck / f.name)
    log("checkpoints: " + ", ".join(f"{f.name} {f.stat().st_size / 1e6:.0f} MB" for f in sorted(ck.glob("*.pth"))))
    with zipfile.ZipFile(EBB) as z:
        names = z.namelist()
    dirs = sorted({"/".join(n.split("/")[:-1]) for n in names})
    log(f"zip: {len(names)} entries; folders: {dirs[:20]}")


def pairs():
    """(original, bokeh) member names in the EBB! zip, matched by file name."""
    with zipfile.ZipFile(EBB) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".png"))]
    orig = {Path(n).stem: n for n in names if "/original/" in n.lower()}
    bok = {Path(n).stem: n for n in names if "/bokeh/" in n.lower()}
    keys = sorted(set(orig) & set(bok), key=lambda s: (len(s), s))
    if not keys:
        sys.exit(f"no original/bokeh pairs found; sample names: {names[:10]}")
    return [(k, orig[k], bok[k]) for k in keys]


def read(z, name):
    return Image.open(io.BytesIO(z.read(name))).convert("RGB")


def prep(im, short=SHORT):
    w, h = im.size
    s = short / min(w, h)
    W, H = int(w * s) // 32 * 32, int(h * s) // 32 * 32
    im = im.resize((int(round(w * s)), int(round(h * s))), Image.BICUBIC)
    return im.crop((0, 0, W, H))


# ---------------------------------------------------------------- selection
def select():
    from transformers import CLIPModel, CLIPProcessor
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    proc = CLIPProcessor.from_pretrained(D.CLIP)
    clip = CLIPModel.from_pretrained(D.CLIP).to(dev).eval()
    texts = list(PROMPTS.values()) + [NEUTRAL]
    P = pairs()[:N_SCAN]
    log(f"{len(pairs())} pairs; scanning {len(P)}")
    probs = []
    with zipfile.ZipFile(EBB) as z, torch.no_grad():
        for i in range(0, len(P), 32):
            ims = [read(z, o) for _, o, _ in P[i:i + 32]]
            x = proc(text=texts, images=ims, return_tensors="pt", padding=True).to(dev)
            probs.append(clip(**x).logits_per_image.softmax(-1).cpu().numpy())
    probs = np.concatenate(probs)
    chosen, used = {}, set()
    for c, cat in enumerate(PROMPTS):
        order = np.argsort(-probs[:, c])
        picks = [int(j) for j in order if int(j) not in used][:PER_CAT]
        used |= set(picks)
        chosen[cat] = [dict(key=P[j][0], orig=P[j][1], bokeh=P[j][2], p=float(probs[j, c])) for j in picks]
    rng = np.random.default_rng(0)
    rest = [j for j in range(len(P)) if j not in used]
    chosen["control"] = [dict(key=P[j][0], orig=P[j][1], bokeh=P[j][2], p=float("nan"))
                         for j in rng.choice(rest, N_CTRL, replace=False)]
    (out / "selection.json").write_text(json.dumps(chosen, indent=1))
    # check sheet: 5 per thin/glass category (25 images)
    sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof_masked"))
    import smoke_paired as SP
    with zipfile.ZipFile(EBB) as z:
        rows = [[read(z, s["orig"]) for s in chosen[cat][:5]] for cat in PROMPTS]
    SP.sheet(rows, out / "selection_check.png")
    log("selection: " + ", ".join(f"{k} {len(v)}" for k, v in chosen.items()))


# ---------------------------------------------------------------- MPIB, patched helpers
def softsplat(x, flow):
    """Forward summation splatting with bilinear weights (replaces MPIB's cupy kernel)."""
    n, c, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    fx, fy = gx[None].float() + flow[:, 0], gy[None].float() + flow[:, 1]
    x0, y0 = fx.floor(), fy.floor()
    out = torch.zeros(n, c, h * w, device=x.device, dtype=x.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            xx, yy = x0 + dx, y0 + dy
            wgt = (1 - (fx - xx).abs()) * (1 - (fy - yy).abs())
            valid = ((xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)).float()
            idx = (yy.clamp(0, h - 1) * w + xx.clamp(0, w - 1)).long().view(n, -1)
            v = (x * (wgt * valid)[:, None]).view(n, c, -1)
            for b in range(n):
                out[b].index_add_(1, idx[b], v[b])
    return out.view(n, c, h, w)


def dilate(x, k=3):
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


def resize(x, short_size, ensure_multiple_of=32):
    h, w = x.shape[-2:]
    hr = min(h, max(short_size, short_size * h / w))
    wr = min(w, max(short_size, short_size * w / h))
    hr = int((hr + ensure_multiple_of - 1) // ensure_multiple_of * ensure_multiple_of)
    wr = int((wr + ensure_multiple_of - 1) // ensure_multiple_of * ensure_multiple_of)
    return F.interpolate(x, size=(hr, wr), mode="bilinear", align_corners=True)


def mask_generator(disp_grad, mask, iteration, k=5):
    disp_grad = disp_grad * mask
    for _ in range(iteration):
        disp_grad = disp_grad / (disp_grad.norm(dim=1, keepdim=True) + 1e-10)
        disp_grad = disp_grad * mask + softsplat(disp_grad, disp_grad) * (1 - mask)
        mask = (disp_grad.abs().sum(dim=1, keepdim=True) > 0).float()
    return dilate(mask, k)


class MPIBModel:
    def __init__(self):
        import yaml
        from munch import DefaultMunch
        sys.path.insert(0, str(MPIB))
        from models import AUPNet, FFCResNetGenerator, MPIBokehRenderer, MPINet
        cfg = DefaultMunch.fromDict(yaml.safe_load((MPIB / "configs" / "default.yaml").read_text()), "C")
        cfg.mpinet.encoder_pretrained = False
        ck = MPIB / "checkpoints"
        load = lambda m, f: (m.load_state_dict(torch.load(ck / f, map_location=dev)["model"]), m)[1]
        dp = torch.nn.DataParallel
        self.inp = load(FFCResNetGenerator(**cfg.inpnet).to(dev), "inpnet.pth").eval()
        self.mpi = load(dp(MPINet(**cfg.mpinet).to(dev)), "mpinet.pth").eval()
        self.aup = load(dp(AUPNet(**cfg.aupnet).to(dev)), "aupnet.pth").eval()
        self.ren = MPIBokehRenderer(**cfg.renderer).to(dev)
        self.bins = cfg.renderer.bins

    @torch.no_grad()
    def layers(self, image, disp, K, inpaint_size=512):
        """MPIB's demo pipeline: returns (mpi_rgb with inpainted background, mpi_alpha, hard one-hot alpha)."""
        import cv2
        d = disp[0, 0].cpu().numpy()
        dre = cv2.resize(d, (resize(disp, inpaint_size, 8).shape[-1], resize(disp, inpaint_size, 8).shape[-2]))
        gy = cv2.GaussianBlur(cv2.Sobel(dre, -1, 0, 1), (7, 7), 0)
        gx = cv2.GaussianBlur(cv2.Sobel(dre, -1, 1, 0), (7, 7), 0)
        grad = np.stack((gx, gy), -1)
        mask = (np.abs(grad).sum(-1, keepdims=True) > 0.2).astype(np.float32)
        _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for i in range(1, lab.max() + 1):
            if (lab == i).sum() < lab.size / 10000:
                mask[lab == i] = 0
        H, W = image.shape[-2:]
        grad_t = torch.from_numpy(grad).permute(2, 0, 1)[None].to(dev)
        mask_t = torch.from_numpy(mask).permute(2, 0, 1)[None].to(dev)
        image_re = resize(image, inpaint_size, 8)
        mask_t = mask_generator(grad_t, mask_t, int(2 * K * min(1, inpaint_size / min(H, W))))
        bg_re = self.inp(image_re * (1 - mask_t), mask_t)
        im_mask = F.interpolate(mask_t, size=(H, W), mode="bilinear", align_corners=True)
        bg = im_mask * F.interpolate(bg_re, size=(H, W), mode="bilinear", align_corners=True) + (1 - im_mask) * image
        base = min(512, int(min(H, W) * 30 / max(K, 1)))
        image_re, disp_re = resize(image, base, 32), resize(disp, base, 32)
        bw_re, alpha_re, _ = self.mpi(image_re, disp_re)
        bw_re = dilate(bw_re[0], 5).unsqueeze(0)
        bw = F.interpolate(bw_re[0], size=(H, W), mode="bilinear", align_corners=True).unsqueeze(0)
        rgb = bw * bg.unsqueeze(1) + (1 - bw) * image.unsqueeze(1)
        scale = max(H / image_re.shape[2], 1)
        for _ in range(int(np.log2(scale))):
            image_re = F.interpolate(image, size=(image_re.shape[2] * 2, image_re.shape[3] * 2), mode="bilinear",
                                     align_corners=True)
            disp_re = F.interpolate(disp, size=(disp_re.shape[2] * 2, disp_re.shape[3] * 2), mode="bilinear",
                                    align_corners=True)
            alpha_re = self.aup(image_re.clamp(0, 1), disp_re, alpha_re)
        image_re, disp_re = resize(image, min(H, W), 2), resize(disp, min(H, W), 2)
        alpha_re = F.interpolate(alpha_re[0], size=(image_re.shape[2] // 2, image_re.shape[3] // 2), mode="bilinear",
                                 align_corners=True).unsqueeze(0)
        alpha_re = self.aup(image_re.clamp(0, 1), disp_re, alpha_re)
        alpha = F.interpolate(alpha_re[0], size=(H, W), mode="bilinear", align_corners=True).unsqueeze(0)
        return rgb, alpha, bg

    def hard(self, image, disp):
        b = self.bins
        c = torch.linspace(0.5 / b, 1 - 0.5 / b, b, device=dev).view(1, b, 1, 1)
        oh = ((c - 0.5 / b <= disp) & (disp <= c + 0.5 / b)).float()
        oh = oh / oh.sum(1, keepdim=True).clamp(min=1)       # a pixel exactly on a bin edge is split, not doubled
        return image.unsqueeze(1).expand(-1, b, -1, -1, -1), oh.unsqueeze(2)

    @torch.no_grad()
    def render(self, rgb, alpha, K, df):
        return self.ren(rgb.clamp(1e-6, 1), alpha, K, df, GAMMA, norm=True, increase_focus_range=True).clamp(0, 1)

    @torch.no_grad()
    def gather(self, image, disp, K, df):
        _, oh = self.hard(image, disp)
        x = image.clamp(1e-6, 1) ** GAMMA
        out = torch.zeros_like(image)
        for i in range(self.bins):
            dfo = max(abs((i + 0.5) / self.bins - df) - 0.5 / self.bins, 0)
            out += oh[:, i] * self.ren.blur_kernels[int(round(dfo * K))](x)
        return out.clamp(1e-10, 1) ** (1 / GAMMA)


# ---------------------------------------------------------------- measures
_lp = None


def lpips_map(a, b):
    global _lp
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev)
    with torch.no_grad():
        return _lp(a * 2 - 1, b * 2 - 1)[0, 0].cpu().numpy()


def regions(disp, df, K):
    """From the input's disparity and the fitted focus: in-focus area, its inner and outer boundary bands (outer =
    where foreground colour can bleed into the defocused background), thin near structures, background interior."""
    from scipy.ndimage import distance_transform_edt, grey_opening
    d = disp[0, 0].cpu().numpy()
    R = K * np.maximum(np.abs(d - df) - 0.5 / 32, 0)
    focus = R < 1.5
    din, dout = distance_transform_edt(focus), distance_transform_edt(~focus)
    thin = (d - grey_opening(d, size=9)) > 0.05
    return dict(outer=(~focus) & (dout <= 12), inner=focus & (din <= 6), thin=thin & ~((~focus) & (dout > 24)),
                interior_bg=(~focus) & (dout > 24), focus=focus)


def score(o, T, R):
    e = (o - T).abs().mean(1)[0].cpu().numpy() * 255
    lp = lpips_map(o, T)
    res = {f"mae_{k}": float(e[m].mean()) if m.sum() > 50 else float("nan") for k, m in R.items()}
    band = R["outer"] | R["inner"]
    res["lpips_band"] = float(lp[band].mean()) if band.sum() > 50 else float("nan")
    res["lpips_all"] = float(lp.mean())
    res["psnr"] = float(10 * np.log10(1 / max(((o - T) ** 2).mean().item(), 1e-10)))
    return res


def to_t(im):
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)


def to_im(x):
    return Image.fromarray((x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))


class Depth:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.p = AutoImageProcessor.from_pretrained(D.DEPTH)
        self.m = AutoModelForDepthEstimation.from_pretrained(D.DEPTH).to(dev).eval()

    @torch.no_grad()
    def __call__(self, im):
        x = self.p(images=im, return_tensors="pt").to(dev)
        d = self.m(**x).predicted_depth[:, None]
        d = F.interpolate(d, size=im.size[::-1], mode="bicubic", align_corners=False)
        lo, hi = torch.quantile(d.flatten(), 0.01), torch.quantile(d.flatten(), 0.99)
        return ((d - lo) / (hi - lo)).clamp(0, 1)


def fit(model, image, disp, T):
    """Grid-fit focus disparity and K to the real target with the hard renderer at half resolution."""
    s = lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    im2, d2, T2 = s(image), s(disp), s(T)
    rgb, oh = model.hard(im2, d2)
    best = None
    for df in DFS:
        for K in KS:
            e = (model.render(rgb, oh, K / 2, float(df)) - T2).abs().mean().item()
            if best is None or e < best[0]:
                best = (e, float(df), K)
    return best[1], best[2]


METHODS = ["input", "gather", "hard", "soft", "mpib"]


def run_scene(model, depth, orig, bok):
    image, T = to_t(prep(orig)), to_t(prep(bok))
    if T.shape != image.shape:
        T = F.interpolate(T, size=image.shape[-2:], mode="bilinear", align_corners=False)
    disp = depth(to_im(image))
    df, K = fit(model, image, disp, T)
    rgb, alpha, _ = model.layers(image, disp, K)
    hrgb, oh = model.hard(image, disp)
    outs = dict(input=image, gather=model.gather(image, disp, K, df), hard=model.render(hrgb, oh, K, df),
                soft=model.render(image.unsqueeze(1).expand_as(rgb), alpha, K, df), mpib=model.render(rgb, alpha, K, df))
    R = regions(disp, df, K)
    res = {k: score(o, T, R) for k, o in outs.items()}
    res["floor_shift1"] = score(torch.roll(T, 1, dims=-1), T, R)      # misalignment sensitivity of the measures
    area = {k: float(m.mean()) for k, m in R.items()}
    return res, dict(df=df, K=K, area=area), outs, T


def subsetA():
    out = OUT
    sel = json.loads((out / "selection.json").read_text())
    model, depth = MPIBModel(), Depth()
    t0 = time.time()
    rows, sheet = [], []
    with zipfile.ZipFile(EBB) as z:
        for cat, items in sel.items():
            for s in items:
                try:
                    res, info, outs, T = run_scene(model, depth, read(z, s["orig"]), read(z, s["bokeh"]))
                except Exception as e:                       # record and continue
                    log(f"{cat} {s['key']}: FAILED {e}")
                    rows.append(dict(cat=cat, key=s["key"], error=str(e)))
                    continue
                rows.append(dict(cat=cat, key=s["key"], res=res, **info))
                if cat in ("hair", "fence", "branches", "glass") and sum(r.get("cat") == cat and "res" in r
                                                                         for r in rows) == 1:
                    sheet.append([to_im(outs["input"]), to_im(T)] + [to_im(outs[k]) for k in ("gather", "hard", "mpib")])
                log(f"{cat} {s['key']}: df {info['df']:.2f} K {info['K']} | outer MAE " + ", ".join(
                    f"{k} {res[k]['mae_outer']:.2f}" for k in METHODS))
                (out / "subsetA.json").write_text(json.dumps(rows, indent=1))
    sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof_masked"))
    import smoke_paired as SP
    n_sheet = SP.sheet(sheet, out / "subsetA_examples.png", W=200) if sheet else 0
    reportA(rows, n_sheet, time.time() - t0)


def reportA(rows, n_sheet, secs):
    ok = [r for r in rows if "res" in r]
    groups = {"thin (hair, branches, fence, foliage)": ["hair", "branches", "fence", "foliage"], "glass": ["glass"],
              "control": ["control"]}
    keys = ["mae_outer", "mae_inner", "mae_thin", "mae_interior_bg", "lpips_band", "lpips_all", "psnr"]
    L = ["# Round 15 subset A: single-depth vs layered rendering on real EBB! pairs (diagnostic)",
         "Error against the real f/1.8 photo (MAE x 255; LPIPS AlexNet). outer = defocused side of the in-focus "
         "boundary (where foreground colour bleeds: the layer-leakage measure); inner = in-focus side; thin = thin "
         "near structures (disparity top-hat); interior = defocused background far from boundaries. floor = the "
         "target against itself shifted 1 px (how much a 1 px misalignment alone costs). Focus and K fitted per "
         "scene with `hard`, then shared.", ""]
    summ = {}
    for g, cats in groups.items():
        rs = [r for r in ok if r["cat"] in cats]
        if not rs:
            continue
        L += [f"## {g} ({len(rs)} scenes)", "", "| method | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for m in METHODS + ["floor_shift1"]:
            vals = {k: float(np.nanmean([r["res"][m][k] for r in rs])) for k in keys}
            summ[f"{g}|{m}"] = vals
            L.append(f"| {m} | " + " | ".join(f"{vals[k]:.3f}" if "lpips" in k else f"{vals[k]:.2f}" for k in keys) + " |")
        wins = lambda a, b, k: sum(r["res"][a][k] <= 0.9 * r["res"][b][k] for r in rs
                                   if np.isfinite(r["res"][a][k]) and np.isfinite(r["res"][b][k]))
        L += ["", f"Per scene, outer-band MAE at least 10% lower: mpib vs hard {wins('mpib', 'hard', 'mae_outer')}/{len(rs)}, "
              f"mpib vs gather {wins('mpib', 'gather', 'mae_outer')}/{len(rs)}, soft vs hard "
              f"{wins('soft', 'hard', 'mae_outer')}/{len(rs)}, mpib vs soft (inpainting) {wins('mpib', 'soft', 'mae_outer')}/{len(rs)}; "
              f"thin-structure MAE mpib vs hard {wins('mpib', 'hard', 'mae_thin')}/{len(rs)}.", ""]
    thin = [r for r in ok if r["cat"] in groups["thin (hair, branches, fence, foliage)"]]
    n = max(len(thin), 1)
    w_out = sum(r["res"]["mpib"]["mae_outer"] <= 0.9 * min(r["res"]["hard"]["mae_outer"], r["res"]["gather"]["mae_outer"])
                for r in thin)
    resid = float(np.nanmean([r["res"]["mpib"]["mae_outer"] / max(r["res"]["mpib"]["mae_interior_bg"], 1e-6)
                              for r in thin]))
    floor = float(np.nanmean([r["res"]["floor_shift1"]["mae_outer"] for r in thin]))
    gap = float(np.nanmean([min(r["res"]["hard"]["mae_outer"], r["res"]["gather"]["mae_outer"]) - r["res"]["mpib"]["mae_outer"]
                            for r in thin]))
    if w_out < 0.6 * n:
        dec = ("Layered rendering does not clearly beat single-depth rendering at boundaries on these real pairs: the "
               "2.5D failure is not the dominant error here (depth error and misalignment are candidates).")
    elif resid >= 1.5:
        dec = ("Layered rendering fixes part of the boundary error, and a clear residual remains at boundaries "
               "(outer band >= 1.5 x background interior): room beyond MPIB on thin structures.")
    else:
        dec = "Layered rendering already fixes the boundary error on these scenes: Idea A has no gap here."
    L += ["## Decision (frozen in README)", "",
          f"- Thin-structure scenes where MPIB's outer-band error is >= 10% below the better single-depth renderer: {w_out}/{len(thin)}.",
          f"- Mean outer-band gain of MPIB over the better single-depth renderer: {gap:+.2f} (MAE x 255); 1 px shift floor {floor:.2f}.",
          f"- MPIB residual: outer band / background interior = {resid:.2f}.",
          f"- **{dec}**", "", f"`subsetA_examples.png` columns: input, real target, gather, hard, mpib (one hair, branches, fence, glass scene).",
          f"\nTotal time {secs:.0f} s."]
    checks = [("every scene ran", len(ok) == len(rows), f"{len(ok)}/{len(rows)}"),
              ("measures finite (outer band, all methods)", all(np.isfinite(r["res"][m]["mae_outer"]) for r in ok for m in METHODS), ""),
              ("renderers differ from the input (blur applied)", np.mean([r["res"]["hard"]["psnr"] != r["res"]["input"]["psnr"] for r in ok]) == 1, ""),
              ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}")]
    (OUT / "subsetA_summary.json").write_text(json.dumps(summ, indent=1))
    sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof_masked"))
    import smoke_paired as SP
    if not SP.write(OUT / "subsetA_checks.md", L, checks):
        sys.exit("subsetA: some checks failed")


# ---------------------------------------------------------------- subset B: additive reflections
def subsetB():
    """Synthetic glass: I = T + R inside a window (T: transmitted scene, R: reflection, dimmed). Exact truth for two
    focus targets: focus on the scene behind the glass (R blurred) and focus on the reflection (T blurred). Every
    renderer gets I and Depth Anything's disparity of I; the additive oracle gets T and R separately."""
    out = OUT
    sel = json.loads((out / "selection.json").read_text())
    model, depth = MPIBModel(), Depth()
    with zipfile.ZipFile(EBB) as z:
        src = [prep(read(z, s["orig"]), 384) for s in sel["control"]]
    rng = np.random.default_rng(0)
    rows, sheet = [], []
    for i in range(0, len(src) - 1, 2):
        Tm, Rf = to_t(src[i]), to_t(src[i + 1])
        H = min(Tm.shape[-2], Rf.shape[-2])
        W = min(Tm.shape[-1], Rf.shape[-1])
        Tm, Rf = Tm[..., :H, :W], Rf[..., :H, :W]
        win = torch.zeros(1, 1, H, W, device=dev)
        y0, x0 = int(H * 0.15), int(W * 0.15)
        win[..., y0:H - y0, x0:W - x0] = 1
        a = float(rng.uniform(0.25, 0.45))
        Rw = a * Rf * win
        I = (Tm * (1 - 0.3 * win) + Rw).clamp(0, 1)
        Kb = 24
        blur = lambda x, r: model.ren.blur_kernels[r](x.clamp(1e-6, 1) ** GAMMA) ** (1 / GAMMA)
        T_scene = (Tm * (1 - 0.3 * win) + blur(Rw, Kb)).clamp(0, 1)      # focus behind the glass: reflection blurred
        T_refl = (blur(Tm * (1 - 0.3 * win), Kb) + Rw).clamp(0, 1)       # focus on the reflection: scene blurred
        disp = depth(to_im(I))
        res = {}
        for tname, T in (("focus_scene", T_scene), ("focus_reflection", T_refl)):
            best = {}
            for mname in ("gather", "hard", "mpib"):
                b = None
                for df in DFS:
                    for K in [8, 16, 24, 32, 48]:
                        if mname == "gather":
                            o = model.gather(I, disp, K, float(df))
                        elif mname == "hard":
                            o = model.render(*model.hard(I, disp), K, float(df))
                        else:
                            continue
                        e = ((o - T).abs() * win).sum().item() / (win.sum().item() * 3) * 255
                        if b is None or e < b[0]:
                            b = (e, float(df), K, o)
                if mname == "mpib":                           # MPIB at hard's best settings (its layers cost more)
                    e0, df, K, _ = best["hard"]
                    rgb, alpha, _ = model.layers(I, disp, K)
                    o = model.render(rgb, alpha, K, df)
                    b = (((o - T).abs() * win).sum().item() / (win.sum().item() * 3) * 255, df, K, o)
                best[mname] = b
            ident = ((I - T).abs() * win).sum().item() / (win.sum().item() * 3) * 255
            res[tname] = {k: dict(mae_window=v[0], df=v[1], K=v[2]) for k, v in best.items()}
            res[tname]["input"] = dict(mae_window=ident)
            res[tname]["additive_oracle"] = dict(mae_window=0.0)
            if i == 0:
                sheet.append([to_im(I), to_im(T)] + [to_im(best[k][3]) for k in ("gather", "hard", "mpib")])
        rows.append(dict(pair=i // 2, refl_strength=a, res=res))
        log(f"glass pair {i // 2}: " + "; ".join(f"{t}: " + ", ".join(f"{k} {v['mae_window']:.2f}" for k, v in r.items())
                                                 for t, r in res.items()))
    sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof_masked"))
    import smoke_paired as SP
    n_sheet = SP.sheet(sheet, out / "subsetB_examples.png", W=200)
    (out / "subsetB.json").write_text(json.dumps(rows, indent=1))
    meths = ["input", "gather", "hard", "mpib", "additive_oracle"]
    L = [f"# Round 15 subset B: additive glass reflections ({len(rows)} synthetic scenes from EBB! photos; diagnostic)",
         "I = transmitted scene (dimmed 30% inside the window) + reflection (strength 0.25-0.45) inside a window. Exact "
         "targets: focus on the scene (reflection blurred, K 24) and focus on the reflection (scene blurred). gather / "
         "hard: focus and K fitted to each target on the window (oracle settings); mpib: hard's settings. The additive "
         "oracle renders T and R separately with their own blur, so it is exact by construction: it shows what a "
         "two-depth additive representation could reach, not a method. MAE x 255 inside the window.", "",
         "| target | " + " | ".join(meths) + " |", "|---|" + "---|" * len(meths)]
    for t in ("focus_scene", "focus_reflection"):
        L.append(f"| {t} | " + " | ".join(f"{np.mean([r['res'][t][m]['mae_window'] for r in rows]):.2f}" for m in meths) + " |")
    best_single = {t: np.mean([min(r["res"][t][m]["mae_window"] for m in ("gather", "hard", "mpib")) for r in rows])
                   for t in ("focus_scene", "focus_reflection")}
    inp = {t: np.mean([r["res"][t]["input"]["mae_window"] for r in rows]) for t in best_single}
    L += ["", "## Decision (frozen in README)", "",
          "- Best single-image renderer as a fraction of doing nothing (input): " + ", ".join(
              f"{t} {best_single[t] / max(inp[t], 1e-6):.2f}" for t in best_single),
          "- **" + ("No single-depth or layered renderer can produce the two focus targets from one image: they barely beat "
                    "the unedited input (>= 0.7 of its error) on at least one target." if any(best_single[t] / max(inp[t], 1e-6) >= 0.7 for t in best_single)
                    else "Existing renderers get substantially closer than the input on both targets: the additive case is "
                         "less clear-cut than predicted.") + "**",
          "- Real data with a focus shift between reflection and scene is still needed: EBB! has one fixed focus per "
          "pair. Its glass scenes are scored in subset A.", "",
          "`subsetB_examples.png` rows: focus on scene, focus on reflection; columns: input, target, gather, hard, mpib."]
    checks = [("all scenes ran", len(rows) == len(src) // 2, f"{len(rows)}"),
              ("targets differ (focus actually moves)", all(r["res"]["focus_scene"]["input"]["mae_window"] > 0 for r in rows), ""),
              ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}")]
    if not SP.write(out / "subsetB_checks.md", L, checks):
        sys.exit("subsetB: some checks failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["setup", "select", "subsetA", "subsetB"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    {"setup": setup, "select": select, "subsetA": subsetA, "subsetB": subsetB}[a.stage]()


if __name__ == "__main__":
    main()
