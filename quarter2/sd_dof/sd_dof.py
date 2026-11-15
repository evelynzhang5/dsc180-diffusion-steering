"""Depth-of-field steering on Stable Diffusion 1.5: reproduce the NA-RFM paper's setup (Wang, Belkin and Wang 2026,
Appendix G.2 and Table 24), then compare it with two other ways of asking for the same blur.

    python quarter2/sd_dof/sd_dof.py collect train reproduce compare [--smoke] [--out outputs/q2_sd_dof]

Paper's setup (Table 24), as implemented here:
- direction data: svnfs/depth-of-field (1,200 Unsplash photos, ~600 shallow / ~600 deep depth of field);
- activations of unet.down_blocks[2].resnets[-1] (1280 x 16 x 16 = 327,680 values) for the VAE latents of the
  photos, forward-noised to sigma ~ 0.344;
- RFM with the paper's own code (na_rfm.rfm_training): Laplace kernel, bandwidth 5000, 5 iterations;
- sampling: 50 DDIM steps, text CFG 7.5; inside sigma in [0.041, 13.12] a third branch eps_s (the conditional
  prediction with the edit h <- h + w |h| d / |d| at that layer, w = 0.7) enters as
  eps = eps_u + 7.5 (eps_c - eps_u) + 4.0 (eps_s - eps_c);
- evaluation: first 25 COCO-Karpathy validation captions x seeds 42, 123, 256, 789; Depth Anything V2 depth, the
  nearest 40% of pixels as foreground, Laplacian variance per region. Paper: background sharpness 1085 -> 786
  (-27.6%), foreground / background ratio 2.25 -> 2.58.
Choices the paper does not state (ours): the photos are resized and center-cropped to 512 px; activations are
collected with the empty prompt and one fixed noise draw; the RFM keeps its code's defaults (reg 1e-3, standardized
features, 20% validation split) and the direction is get_direction(topk=1); captions are the first caption of the
first 25 validation images of yerevann/coco-karpathy; Depth Anything V2 Small.

Comparison (stage `compare`), same three-branch formula, only eps_s changes:
- meandiff: the same layer and data, but the direction is the shallow-minus-deep mean of the activations (no RFM);
- prompt: eps_s is the conditional prediction for the caption + ", shallow depth of field, blurred bokeh
  background", with no activation edit.
Each one's amplification s is picked on development captions (validation images 25-32, seeds 1-2) to match the
RFM setting's background-sharpness change there, then all are evaluated on the 25 test captions x 4 seeds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NARFM = ROOT / "downloads" / "narfm" / "na-rfm"   # the paper's code (cloned by cifar10_replication/replicate.py setup)
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEPTH = "depth-anything/Depth-Anything-V2-Small-hf"
CLIP = "openai/clip-vit-large-patch14"
STEPS, CFG = 50, 7.5
W_RFM, S_RFM = 0.7, 4.0
SIG_LO, SIG_HI = 0.041, 13.12
SIG_COLLECT = 0.344
BANDWIDTH = 5000.0
SEEDS = [42, 123, 256, 789]
DEV_SEEDS = [1, 2]
BLUR_SUFFIX = ", shallow depth of field, blurred bokeh background"
SWEEP = {"meandiff": [1.0, 2.0, 4.0, 8.0, 16.0], "prompt": [0.5, 1.0, 2.0, 4.0]}


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


class Hook:
    """Forward hook: on rows [lo, hi) of the batch, h <- h + w |h| d / |d| (the paper's additive, adaptive edit)."""

    def __init__(self, module):
        self.d, self.w, self.rows, self.last = None, 0.0, None, None
        module.register_forward_hook(self.fn)

    def fn(self, mod, inp, out):
        self.last = out
        if self.d is None or self.rows is None:
            return out
        lo, hi = self.rows
        h = out[lo:hi].float()
        norm = h.flatten(1).norm(dim=1).view(-1, 1, 1, 1)   # float32: fp16 squares overflow
        out = out.clone()
        out[lo:hi] = (h + self.w * norm * self.d).to(out.dtype)
        return out


class SD:
    def __init__(self, device):
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        self.dt = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32   # fp16 slow on Pascal
        pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=self.dt, safety_checker=None,
                                                       requires_safety_checker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(STEPS)
        pipe.set_progress_bar_config(disable=True)
        self.pipe, self.dev = pipe.to(device), device
        self.hook = Hook(pipe.unet.down_blocks[2].resnets[-1])
        ac = pipe.scheduler.alphas_cumprod.double()
        self.sig_all = ((1 - ac) / ac).sqrt()                      # sigma of x_t for every training timestep
        self.window = [k for k, t in enumerate(pipe.scheduler.timesteps)
                       if SIG_LO - 5e-3 <= float(self.sig_all[int(t)]) <= SIG_HI + 5e-3]   # table values are rounded

    @torch.no_grad()
    def embed(self, texts):
        p = self.pipe
        ids = p.tokenizer(texts, padding="max_length", max_length=p.tokenizer.model_max_length, truncation=True,
                          return_tensors="pt").input_ids
        return p.text_encoder(ids.to(self.dev))[0]

    @torch.no_grad()
    def acts(self, images, seed=0, bs=8):
        """Layer activations (n, 327,680, fp16 on CPU) of the photos' latents forward-noised to sigma ~ 0.344."""
        p = self.pipe
        t = int((self.sig_all - SIG_COLLECT).abs().argmin())
        log(f"collecting at t = {t}, sigma = {float(self.sig_all[t]):.4f}")
        empty = self.embed([""])
        g = torch.Generator().manual_seed(seed)
        out = []
        for i in range(0, len(images), bs):
            x = torch.stack([torch.from_numpy(np.asarray(im)).permute(2, 0, 1) for im in images[i:i + bs]])
            x = (x.float() / 127.5 - 1).to(self.dev, self.dt)
            z0 = p.vae.encode(x).latent_dist.mean * p.vae.config.scaling_factor
            noise = torch.randn(z0.shape, generator=g).to(self.dev, self.dt)
            tt = torch.full((len(z0),), t, device=self.dev, dtype=torch.long)
            zt = p.scheduler.add_noise(z0, noise, tt)
            p.unet(zt, tt, encoder_hidden_states=empty.expand(len(z0), -1, -1))
            out.append(self.hook.last.float().flatten(1).half().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def generate(self, prompts, seeds, mode="none", s=0.0, direction=None, bs=4):
        """Images for (prompt, seed) pairs. mode none: plain CFG. rfm / meandiff: third branch with the activation
        edit. prompt: third branch conditioned on caption + BLUR_SUFFIX."""
        p, imgs = self.pipe, []
        uncond = self.embed([""])
        d = None
        if direction is not None:
            d = (direction.float() / direction.float().norm()).view(1, 1280, 16, 16).to(self.dev)
        for i in range(0, len(prompts), bs):
            pr, sd = prompts[i:i + bs], seeds[i:i + bs]
            n = len(pr)
            cond = self.embed(pr)
            third = self.embed([q + BLUR_SUFFIX for q in pr]) if mode == "prompt" else cond
            z = torch.stack([torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(x)) for x in sd])
            z = (z * p.scheduler.init_noise_sigma).to(self.dev, self.dt)
            for k, t in enumerate(p.scheduler.timesteps):
                on = mode != "none" and k in self.window
                emb = torch.cat([uncond.expand(n, -1, -1), cond] + ([third] if on else []))
                zz = torch.cat([z] * (3 if on else 2))
                if on and mode in ("rfm", "meandiff"):
                    self.hook.d, self.hook.w, self.hook.rows = d, W_RFM, (2 * n, 3 * n)
                out = p.unet(zz, t, encoder_hidden_states=emb).sample
                self.hook.d, self.hook.rows = None, None
                eu, ec = out[:n], out[n:2 * n]
                eps = eu + CFG * (ec - eu)
                if on:
                    eps = eps + s * (out[2 * n:] - ec)
                z = p.scheduler.step(eps, t, z).prev_sample
            x = torch.cat([p.vae.decode(zi[None] / p.vae.config.scaling_factor).sample.float() for zi in z])
            x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            imgs += [Image.fromarray(v) for v in x]
        return imgs


class DofMeter:
    """Paper's metric: Depth Anything V2 depth, nearest 40% of pixels = foreground, Laplacian variance per region.
    Plus CLIP caption score and CLIP 'blurry background' minus 'sharp background'."""

    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, CLIPModel, CLIPProcessor
        self.dproc = AutoImageProcessor.from_pretrained(DEPTH)
        self.depth = AutoModelForDepthEstimation.from_pretrained(DEPTH).to(device).eval()
        self.cproc = CLIPProcessor.from_pretrained(CLIP)
        self.clip = CLIPModel.from_pretrained(CLIP).to(device).eval()
        self.dev = device

    @torch.no_grad()
    def fg_mask(self, img):
        inp = self.dproc(images=img, return_tensors="pt").to(self.dev)
        pred = self.depth(**inp).predicted_depth                      # relative inverse depth: larger = nearer
        pred = torch.nn.functional.interpolate(pred[:, None], size=img.size[::-1], mode="bicubic",
                                               align_corners=False)[0, 0].cpu().numpy()
        return pred >= np.quantile(pred, 0.6)

    @torch.no_grad()
    def clip_scores(self, img, caption):
        texts = [caption, "a photo with a blurry, out-of-focus background (bokeh)",
                 "a photo with a sharp, in-focus background"]
        inp = self.cproc(text=texts, images=img, return_tensors="pt", padding=True, truncation=True).to(self.dev)
        lg = self.clip(**inp).logits_per_image[0]
        return float(lg[0]), float(lg[1] - lg[2])

    def __call__(self, img, caption):
        from scipy.ndimage import laplace
        lap = laplace(np.asarray(img.convert("L")).astype(np.float64))
        fg = self.fg_mask(img)
        fgv, bgv = float(lap[fg].var()), float(lap[~fg].var())
        cap, blur = self.clip_scores(img, caption)
        return dict(bg_var=bgv, fg_var=fgv, ratio=fgv / max(bgv, 1e-6), log_bg=float(np.log(bgv + 1e-6)),
                    log_fg=float(np.log(fgv + 1e-6)), clip_caption=cap, clip_blur=blur, fg_mask=fg)


# ---------------------------------------------------------------- data
def dof_photos(n_per_class=None):
    from datasets import load_dataset
    ds = load_dataset("svnfs/depth-of-field", split="train")
    names = ds.features["label"].names
    labels = np.array(ds["label"])
    keep = np.arange(len(ds))
    if n_per_class:
        keep = np.concatenate([np.flatnonzero(labels == c)[:n_per_class] for c in np.unique(labels)])
    imgs = []
    for i in keep:
        im = ds[int(i)]["image"].convert("RGB")
        w, h = im.size
        r = 512 / min(w, h)
        im = im.resize((max(512, round(w * r)), max(512, round(h * r))), Image.BICUBIC)
        w, h = im.size
        imgs.append(im.crop(((w - 512) // 2, (h - 512) // 2, (w - 512) // 2 + 512, (h - 512) // 2 + 512)))
    return imgs, labels[keep], names


def captions():
    from datasets import load_dataset
    ds = load_dataset("yerevann/coco-karpathy", split="validation")
    caps = [ds[i]["sentences"][0].strip().rstrip(".") for i in range(33)]
    return caps[:25], caps[25:33]


# ---------------------------------------------------------------- stages
def collect(cfg, sd):
    path = cfg.out / "acts.pt"
    if path.exists():
        return
    from scipy.ndimage import laplace
    imgs, labels, names = dof_photos(30 if cfg.smoke else None)
    log(f"{len(imgs)} photos; label names {names}; counts {np.bincount(labels).tolist()}")
    # check which label is shallow depth of field: its photos should be less sharp overall
    sharp = np.array([np.log(laplace(np.asarray(im.convert('L')).astype(float)).var() + 1e-6) for im in imgs])
    by = {int(c): float(sharp[labels == c].mean()) for c in np.unique(labels)}
    shallow = min(by, key=by.get)
    log(f"mean log Laplacian variance by label {by}: shallow (blurrier) = label {shallow}")
    H = sd.acts(imgs)
    torch.save({"H": H, "labels": torch.from_numpy(labels), "shallow": shallow, "names": names, "sharp_by_label": by},
               path)
    log(f"activations {tuple(H.shape)} saved")


def train(cfg):
    path = cfg.out / "directions.pt"
    if path.exists():
        return
    sys.path.insert(0, str(NARFM))
    from na_rfm.rfm_training import RFMTrainConfig, get_direction, train_binary_rfm
    from sklearn.metrics import roc_auc_score
    a = torch.load(cfg.out / "acts.pt")
    H, y, shallow = a["H"], a["labels"], a["shallow"]
    t0 = time.time()
    res = None
    for dev in ("cuda", "cpu"):
        try:
            conf = RFMTrainConfig(bandwidth=BANDWIDTH, reg=1e-3, n_iters=5, top_k=5, device=dev, verbose=True)
            res = train_binary_rfm(H.float(), y, target_value=shallow, config=conf)
            break
        except torch.cuda.OutOfMemoryError:
            log("RFM out of GPU memory; retrying on CPU")
            torch.cuda.empty_cache()
    rfm = get_direction(res, topk=1).float().cpu()
    log(f"RFM: held-out AUC {res.best_auc:.3f}, best iteration {res.best_iter} [{time.time() - t0:.0f} s]")
    Hf = H.float()
    is_sh = (y == shallow)
    md = Hf[is_sh].mean(0) - Hf[~is_sh].mean(0)
    # same 80/20 split as train_binary_rfm (seed 42) for the mean-difference AUC
    perm = np.random.default_rng(42).permutation(len(y))
    val = perm[:int(len(y) * 0.2)]
    trn = perm[int(len(y) * 0.2):]
    md_tr = Hf[trn][is_sh[trn]].mean(0) - Hf[trn][~is_sh[trn]].mean(0)
    md_auc = roc_auc_score(is_sh[val].numpy(), (Hf[val] @ md_tr).numpy())
    cos = float(torch.nn.functional.cosine_similarity(rfm, md, dim=0))
    log(f"mean difference: held-out AUC {md_auc:.3f}; cosine with RFM direction {cos:.3f}")
    torch.save({"rfm": rfm, "meandiff": md, "rfm_auc": float(res.best_auc), "meandiff_auc": float(md_auc),
                "cos": cos, "eigenvalues": res.eigenvalues.cpu()}, path)


def run_set(cfg, sd, meter, name, prompts, pidx, seeds, mode, s, direction, rec):
    """Generate and meter every (prompt, seed) of one condition; resumable via the records file."""
    todo = [(pi, x) for pi in pidx for x in seeds if (name, pi, x) not in rec]
    img_dir = cfg.out / "images" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for j in range(0, len(todo), 4):
        chunk = todo[j:j + 4]
        imgs = sd.generate([prompts[pi] for pi, _ in chunk], [x for _, x in chunk], mode, s, direction)
        with (cfg.out / "records.jsonl").open("a") as fh:
            for (pi, x), im in zip(chunk, imgs):
                im.save(img_dir / f"{pi:02d}_{x}.jpg", quality=95)
                m = meter(im, prompts[pi])
                fg = m.pop("fg_mask")
                np.save(img_dir / f"{pi:02d}_{x}_fg.npy", np.packbits(fg))
                r = dict(cond=name, mode=mode, s=s, prompt=pi, seed=x, **m)
                rec[(name, pi, x)] = r
                fh.write(json.dumps(r) + "\n")
    if todo:
        log(f"{name}: {len(todo)} images, {(time.time() - t0) / len(todo):.1f} s per image")


def load_records(cfg):
    rec = {}
    p = cfg.out / "records.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            rec[(r["cond"], r["prompt"], r["seed"])] = r
    return rec


def pixel_change(cfg, a, b, pidx, seeds):
    """Mean |a - b| / 255 inside the unedited image's foreground and background (where the edit landed)."""
    fgc, bgc = [], []
    for pi in pidx:
        for x in seeds:
            A = np.asarray(Image.open(cfg.out / "images" / a / f"{pi:02d}_{x}.jpg")).astype(float)
            B = np.asarray(Image.open(cfg.out / "images" / b / f"{pi:02d}_{x}.jpg")).astype(float)
            fg = np.unpackbits(np.load(cfg.out / "images" / a / f"{pi:02d}_{x}_fg.npy"))[:512 * 512].reshape(512, 512)
            dlt = np.abs(A - B).mean(2) / 255
            fgc.append(dlt[fg.astype(bool)].mean())
            bgc.append(dlt[~fg.astype(bool)].mean())
    return np.array(fgc), np.array(bgc)


def hboot(mat, n=2000, seed=0):
    """Mean and 95% interval of a (prompts x seeds) array: resample prompts, then seeds within prompt."""
    mat = np.asarray(mat, float)
    rng = np.random.default_rng(seed)
    P, S = mat.shape
    pi = rng.integers(0, P, (n, P))
    si = rng.integers(0, S, (n, P, S))
    bs = mat[pi[..., None], si].mean((1, 2))
    return float(mat.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def f3(t, nd=3):
    return f"{t[0]:+.{nd}f} ({t[1]:+.{nd}f}, {t[2]:+.{nd}f})"


def reproduce(cfg, sd, meter, test, pidx, seeds, dirs, rec):
    run_set(cfg, sd, meter, "none", test, pidx, seeds, "none", 0.0, None, rec)
    run_set(cfg, sd, meter, "rfm", test, pidx, seeds, "rfm", S_RFM, dirs["rfm"], rec)


def compare(cfg, sd, meter, test, dev, pidx, seeds, dpidx, dseeds, dirs, rec):
    run_set(cfg, sd, meter, "dev_none", dev, dpidx, dseeds, "none", 0.0, None, rec)
    run_set(cfg, sd, meter, "dev_rfm", dev, dpidx, dseeds, "rfm", S_RFM, dirs["rfm"], rec)
    effect = lambda name: np.mean([rec[(name, pi, x)]["log_bg"] - rec[("dev_none", pi, x)]["log_bg"]
                                   for pi in dpidx for x in dseeds])
    target = effect("dev_rfm")
    log(f"dev target: change in log background sharpness {target:+.3f} (RFM, s = {S_RFM})")
    picked = {}
    for mode, grid in SWEEP.items():
        grid = grid[:2] if cfg.smoke else grid
        got = {}
        for s in grid:
            name = f"dev_{mode}_s{s:g}"
            run_set(cfg, sd, meter, name, dev, dpidx, dseeds, mode, s, dirs.get(mode), rec)
            got[s] = effect(name)
            log(f"dev {mode} s = {s:g}: {got[s]:+.3f}")
        picked[mode] = min(got, key=lambda s: abs(got[s] - target))
        log(f"picked {mode}: s = {picked[mode]:g} (dev {got[picked[mode]]:+.3f} vs target {target:+.3f})")
    (cfg.out / "picked.json").write_text(json.dumps({"target": target, **{k: v for k, v in picked.items()}}, indent=1))
    for mode, s in picked.items():
        run_set(cfg, sd, meter, mode, test, pidx, seeds, mode, s, dirs.get(mode), rec)


def report(cfg, test, pidx, seeds, dirs, rec):
    conds = [c for c in ("rfm", "meandiff", "prompt") if all((c, pi, x) in rec for pi in pidx for x in seeds)]
    a = torch.load(cfg.out / "acts.pt")
    L = ["# SD 1.5 depth-of-field steering: the paper's setup reproduced, then compared", ""]
    if cfg.smoke:
        L += ["**SMOKE TEST: too few images for any number here to mean anything.**", ""]
    L += [f"Direction data: {len(a['labels'])} photos, label names {a['names']}, shallow = label {a['shallow']} "
          f"(mean log Laplacian variance by label {json.dumps({k: round(v, 2) for k, v in a['sharp_by_label'].items()})}).",
          f"RFM held-out AUC {dirs['rfm_auc']:.3f}; mean-difference held-out AUC {dirs['meandiff_auc']:.3f}; cosine "
          f"between the two directions {dirs['cos']:+.3f}.",
          f"Steering window: DDIM steps {sd_window_text(cfg)} of {STEPS}.",
          f"Test: {len(pidx)} captions x seeds {seeds}.", ""]
    none = lambda k: np.array([[rec[("none", pi, x)][k] for x in seeds] for pi in pidx])
    get = lambda c, k: np.array([[rec[(c, pi, x)][k] for x in seeds] for pi in pidx])
    L += ["## 1. Paper's numbers vs ours", "",
          "| | background sharpness (mean Laplacian variance) | change | FG / BG ratio (mean of images) |",
          "|---|---|---|---|",
          "| paper, unsteered -> steered | 1085.03 -> 785.89 | -27.6% | 2.247 -> 2.585 |"]
    for c in conds:
        b0, b1 = none("bg_var").mean(), get(c, "bg_var").mean()
        L.append(f"| ours, {c} | {b0:.1f} -> {b1:.1f} | {100 * (b1 / b0 - 1):+.1f}% | "
                 f"{none('ratio').mean():.3f} -> {get(c, 'ratio').mean():.3f} |")
    picked = json.loads((cfg.out / "picked.json").read_text()) if (cfg.out / "picked.json").exists() else {}
    L += ["", "## 2. Paired changes vs. unedited (same caption and seed)", "",
          "Intervals: bootstrap over captions, then seeds within caption. Effect: background blur. Selectivity: the "
          "foreground and the caption should not change. Pixel change is mean |edited - unedited| / 255 inside the "
          "unedited image's foreground and background.", "",
          "| condition | s | log BG sharpness | log FG sharpness | log FG/BG ratio | CLIP blur score | CLIP caption "
          "score | pixel change, FG | pixel change, BG |", "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        s = S_RFM if c == "rfm" else picked.get(c, float("nan"))
        dl = {k: hboot(get(c, k) - none(k)) for k in ("log_bg", "log_fg", "clip_blur", "clip_caption")}
        rat = hboot(np.log(get(c, "ratio")) - np.log(none("ratio")))
        fgc, bgc = pixel_change(cfg, "none", c, pidx, seeds)
        L.append(f"| {c} | {s:g} | {f3(dl['log_bg'])} | {f3(dl['log_fg'])} | {f3(rat)} | {f3(dl['clip_blur'])} | "
                 f"{f3(dl['clip_caption'])} | {fgc.mean():.3f} | {bgc.mean():.3f} |")
    if picked:
        L += ["", f"Dev target (RFM's change in log background sharpness on dev captions): {picked['target']:+.3f}; "
              f"picked s: {', '.join(f'{k} {v:g}' for k, v in picked.items() if k != 'target')}."]
    # visual check: 6 captions, first seed, all conditions side by side (<= 24 images)
    show = pidx[:6]
    cols = ["none"] + conds
    W = 160
    sheet = Image.new("RGB", (W * len(cols), W * len(show)), "white")
    for r, pi in enumerate(show):
        for c, name in enumerate(cols):
            im = Image.open(cfg.out / "images" / name / f"{pi:02d}_{seeds[0]}.jpg").resize((W, W), Image.LANCZOS)
            sheet.paste(im, (c * W, r * W))
    sheet.save(cfg.out / "examples.png")
    L += ["", f"`examples.png`: columns {', '.join(cols)}; rows test captions {show[0]}-{show[-1]}, seed {seeds[0]}.",
          "", "Captions: " + "; ".join(f"{pi}: {test[pi]}" for pi in show), ""]
    (cfg.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def sd_window_text(cfg):
    w = json.loads((cfg.out / "window.json").read_text())
    return f"{w[0]}-{w[-1]}" if w else "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=["collect", "train", "reproduce", "compare", "report"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="60 photos, 2 test captions x 1 seed, 1 dev caption")
    cfg = ap.parse_args()
    cfg.out = cfg.out or ROOT / "outputs" / ("q2_sd_dof_smoke" if cfg.smoke else "q2_sd_dof")
    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    test, dev = captions()
    pidx, seeds = (list(range(2)), SEEDS[:1]) if cfg.smoke else (list(range(25)), SEEDS)
    dpidx, dseeds = (list(range(1)), DEV_SEEDS[:1]) if cfg.smoke else (list(range(8)), DEV_SEEDS)
    sd = meter = None
    for st in cfg.stages:
        log(f"stage {st}{' (smoke)' if cfg.smoke else ''}")
        if st == "train":            # no SD in memory: the RFM needs the GPU's memory
            train(cfg)
            continue
        if sd is None and st != "report":
            sd = SD("cuda")
            (cfg.out / "window.json").write_text(json.dumps(sd.window))
            sig = [round(float(sd.sig_all[int(t)]), 3) for t in sd.pipe.scheduler.timesteps]
            log(f"sigma per DDIM step: first {sig[:3]} ... last {sig[-3:]}; window = steps {sd.window[0]}-"
                f"{sd.window[-1]} ({len(sd.window)} of {STEPS})")
        if st == "collect":
            collect(cfg, sd)
            continue
        dirs = torch.load(cfg.out / "directions.pt")
        rec = load_records(cfg)
        if st in ("reproduce", "compare"):
            meter = meter or DofMeter("cuda")
        if st == "reproduce":
            reproduce(cfg, sd, meter, test, pidx, seeds, dirs, rec)
        elif st == "compare":
            compare(cfg, sd, meter, test, dev, pidx, seeds, dpidx, dseeds, dirs, rec)
        report(cfg, test, pidx, seeds, dirs, load_records(cfg))
    log("done")


if __name__ == "__main__":
    main()
