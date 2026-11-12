#!/usr/bin/env python3
"""Q2 round 5: can a small output audit predict a steering direction's side effects on new generations? (CIFAR-10)

Steering: the NA-RFM paper's RFM-only class steering (one learned direction per class at block 9, w = 1, s = 2, every
step), on the paper's improved-DDPM CIFAR-10 model, 100 DDIM steps. Classes: airplane, cat, deer, ship.

Definitions (fixed before any test image exists; see README.md):
  native_c   unsteered (unguided) model images that the paper's ConvNeXt evaluator assigns to class c: the model's own
             class-c images. Steered images are compared with these, so the change needed to make class c is not
             counted as a side effect, and the model's general differences from real photos cancel.
  side effect of steering toward c on meter m:
             X_c(m) = (mean of m over steered images assigned to c - mean over native_c) / SD of m over real class-c
             reference images. Set-level meters (diversity): ratio steered / native - 1.
  random control: the same hook with a random direction (the hook rescales any direction to strength x |h|, so the
             size is identical). Its shift against unguided images on the same seeds is the generic-perturbation part.

Predictions (written to predictions.json, hashed, before the test images are generated):
  REF        reference images only: steering toward c makes images like the class references, so no side effects.
  OUT_n      paired outputs only: n dev steered images vs. the model's native_c images from a dev unguided pool.
  BOTH_n     outputs + references: n dev steered images vs. the real class-c reference images.
  TRANSFER   the other three classes' OUT_256 estimates, averaged (does a side effect carry over between classes?).
Truth: the same X_c(m) on new seeds (steered seeds 3000-3511, native pool seeds 20000-25119).

Stages: meters (positive controls on real images), gen_dev, predict, gen_test, evaluate.
  python quarter2/cifar_audit/cifar_audit.py meters gen_dev predict      # pod 1
  python quarter2/cifar_audit/cifar_audit.py gen_test evaluate           # pod 2, after predictions are frozen
  --smoke: 2 classes and a few images, to separate folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "cifar10_replication"))
import replicate as R  # noqa: E402  (paths, CLASSES, cifar_train, env_paths)

AUDIT_CLASSES = [0, 3, 4, 8]                       # airplane, cat, deer, ship
UNGUIDED = "--gaussian-strength 0 --rfm-strength 0 --rfm-sigma-lo 1e8 --rfm-sigma-hi 1e9"
RFM_ONLY = "--gaussian-strength 0 --rfm-sigma-lo 0 --rfm-sigma-hi 1e4"
TAU = 0.25          # per-image meters: |effect| in SDs of real class-c images
TAU_SET = 0.10      # set-level meters: |ratio - 1|
N_BOOT = 1000
SEEDS = {           # (start, count); dev steered and dev unguided pairs are the replication's seeds 0-255
    "dev_pairs": (0, 256), "dev_native": (10000, 2048),
    "test_pairs": (3000, 512), "test_native": (20000, 5120)}
SMOKE_SEEDS = {"dev_pairs": (0, 16), "dev_native": (10000, 256), "test_pairs": (3000, 16), "test_native": (20000, 256)}
OUT_BUDGETS = (32, 256)
SMOKE_BUDGETS = (8, 16)

IMAGE_METERS = ["brightness", "contrast", "saturation", "colorfulness", "blueness", "greenness", "redness",
                "border_clutter", "figure_ground", "sharpness"]
SET_METERS = ["feature_diversity", "background_diversity", "pixel_diversity"]
METER_HELP = {
    "brightness": "mean luma", "contrast": "SD of luma within the image", "saturation": "mean HSV saturation",
    "colorfulness": "Hasler-Suesstrunk colourfulness", "blueness": "mean B - (R+G)/2",
    "greenness": "mean G - (R+B)/2", "redness": "mean R - (G+B)/2",
    "border_clutter": "SD of luma in the 4-pixel border ring (background busyness)",
    "figure_ground": "RGB distance between the centre 16x16 mean and the border-ring mean",
    "sharpness": "mean absolute Laplacian of luma",
    "feature_diversity": "mean pairwise cosine distance of ConvNeXt features",
    "background_diversity": "mean pairwise distance of border-ring mean colours",
    "pixel_diversity": "mean pairwise L2 distance of 8x8 downsampled images"}


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


# ---------------------------------------------------------------- meters (images: N x 32 x 32 x 3 in [0, 1])
def luma(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def ring(x, w=4):
    m = np.ones(x.shape[1:3], bool)
    m[w:-w, w:-w] = False
    return m


def image_meters(x):
    x = x.astype(np.float64)
    y = luma(x)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    rg, yb = r - g, 0.5 * (r + g) - b
    m = ring(x)
    border = x[:, m]                                   # N x P x 3
    centre = x[:, 8:24, 8:24].reshape(len(x), -1, 3)
    lap = (-4 * y[:, 1:-1, 1:-1] + y[:, :-2, 1:-1] + y[:, 2:, 1:-1] + y[:, 1:-1, :-2] + y[:, 1:-1, 2:])
    return {
        "brightness": y.mean((1, 2)),
        "contrast": y.std((1, 2)),
        "saturation": ((mx - mn) / (mx + 1e-6)).mean((1, 2)),
        "colorfulness": np.sqrt(rg.std((1, 2)) ** 2 + yb.std((1, 2)) ** 2)
                        + 0.3 * np.sqrt(rg.mean((1, 2)) ** 2 + yb.mean((1, 2)) ** 2),
        "blueness": (b - 0.5 * (r + g)).mean((1, 2)),
        "greenness": (g - 0.5 * (r + b)).mean((1, 2)),
        "redness": (r - 0.5 * (g + b)).mean((1, 2)),
        "border_clutter": y[:, m].std(1),
        "figure_ground": np.linalg.norm(centre.mean(1) - border.mean(1), axis=1),
        "sharpness": np.abs(lap).mean((1, 2)),
    }


def mean_pairwise(z, metric, rng, cap=256):
    if len(z) < 2:
        return float("nan")
    if len(z) > cap:
        z = z[rng.choice(len(z), cap, replace=False)]
    if metric == "cosine":
        u = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        d = 1 - u @ u.T
    else:
        sq = (z ** 2).sum(1)
        d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * z @ z.T, 0))
    n = len(z)
    return float(d[np.triu_indices(n, 1)].mean())


def set_meters(x, feats, rng):
    m = ring(x)
    bg = x[:, m].mean(1)
    small = x.reshape(len(x), 8, 4, 8, 4, 3).mean((2, 4)).reshape(len(x), -1)
    return {"feature_diversity": mean_pairwise(feats, "cosine", rng),
            "background_diversity": mean_pairwise(bg, "l2", rng),
            "pixel_diversity": mean_pairwise(small, "l2", rng)}


# ---------------------------------------------------------------- evaluator: predictions and features
class Evaluator:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        R.env_paths()
        from repro.run_cifar10_na_rfm import CIFAR_HF_EVAL_MODEL, cifar_label_maps
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(CIFAR_HF_EVAL_MODEL)
        self.model = AutoModelForImageClassification.from_pretrained(CIFAR_HF_EVAL_MODEL).cuda().eval()
        _, self.logit_to_cifar = cifar_label_maps(self.model)

    def __call__(self, x01, bs=128):
        from PIL import Image
        torch = self.torch
        preds, feats = [], []
        u8 = (np.clip(x01, 0, 1) * 255).round().astype(np.uint8)
        for i in range(0, len(u8), bs):
            inp = self.proc([Image.fromarray(a) for a in u8[i:i + bs]], return_tensors="pt")
            with torch.no_grad():
                out = self.model.convnext(pixel_values=inp["pixel_values"].cuda())
                f = out.pooler_output
                logits = self.model.classifier(f)
            preds += [self.logit_to_cifar[int(j)] for j in logits.argmax(-1).cpu().tolist()]
            feats.append(f.float().cpu().numpy())
        return np.array(preds), np.concatenate(feats)


# ---------------------------------------------------------------- generation
def to01(samples):
    return ((samples.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy().astype(np.float32)


class Sampler:
    def __init__(self):
        import os
        import torch
        R.env_paths()
        os.chdir(R.REPO)
        from na_rfm.ddim_sampling import build_ddim_schedule
        from na_rfm.model_loaders import load_openai_unet
        from repro.run_cifar10_na_rfm import parse_args, sample_class
        self.torch, self.parse_args, self.sample_class = torch, parse_args, sample_class
        self.unet = load_openai_unet(str(R.CKPT), image_size=32, device="cuda")
        self.sched = build_ddim_schedule(1000, 100, 0.0001, 0.02, "cuda")

    def draw(self, k, flags, start, n, concepts=None):
        extra = f" --concepts-root {concepts}" if concepts else ""
        args = self.parse_args(f"--classes {k} --num-samples {n} --seed-start {start} --batch-size 64 "
                               f"--inference-steps 100 {flags}{extra}".split())
        with self.torch.no_grad():
            s, _ = self.sample_class(unet=self.unet, schedule=self.sched, class_idx=k, args=args)
        return s


def random_concepts(cfg):
    """One random direction per audited class, saved where sample_class looks for class_<k>.pt."""
    import torch
    d = cfg.samples / "random_concepts"
    d.mkdir(parents=True, exist_ok=True)
    for k in cfg.classes:
        f = d / f"class_{k}.pt"
        if f.exists():
            continue
        real = torch.load(R.CONCEPTS / f"class_{k}.pt", weights_only=False)["direction"].float()
        g = torch.Generator().manual_seed(1000 + k)
        v = torch.randn(real.shape, generator=g)
        v = v * real.norm() / v.norm()        # the hook rescales anyway; same norm for clarity
        torch.save({"direction": v, "note": f"random control for class {k}, seed {1000 + k}"}, f)
        print(f"   random direction {k}: cosine with class direction "
              f"{float(torch.nn.functional.cosine_similarity(v.flatten(), real.flatten(), dim=0)):+.3f}", flush=True)
    return d


def cached(path, fn):
    import torch
    if path.exists():
        return torch.load(path, weights_only=False)
    t0 = time.time()
    s = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(s, path)
    print(f"   {path.relative_to(cfg_global.samples)}: {tuple(s.shape)} [{time.time() - t0:.0f} s]", flush=True)
    return s


def generate(cfg, split):
    """Steered (RFM-only), unguided and random-direction images on the paired seeds; a large unguided native pool."""
    import torch
    sp = cfg.seeds[f"{split}_pairs"]
    nat = cfg.seeds[f"{split}_native"]
    sampler = None

    def S():
        nonlocal sampler
        if sampler is None:
            sampler = Sampler()
        return sampler

    rdir = random_concepts(cfg)
    out = cfg.samples / split
    if split == "dev" and not cfg.smoke:   # reuse the replication's seeds 0-255 (same code, same seeds)
        for row, name in (("rfm_only", "steered"), ("unguided", None)):
            for k in (cfg.classes if name else [cfg.classes[0]]):
                src = R.WORK / "samples" / row / R.CLASSES[k] / "samples.pt"
                dst = out / (f"steered_{k}.pt" if name else "unguided.pt")
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    s = torch.load(src, weights_only=False)
                    assert len(s) == sp[1], (src, len(s))
                    torch.save(s, dst)
    for k in cfg.classes:
        cached(out / f"steered_{k}.pt", lambda: S().draw(k, RFM_ONLY, *sp))
        cached(out / f"random_{k}.pt", lambda: S().draw(k, RFM_ONLY, *sp, concepts=rdir))
    cached(out / "unguided.pt", lambda: S().draw(cfg.classes[0], UNGUIDED, *sp))
    # native pool, in chunks so an interrupted run resumes
    chunk = 1024
    for c0 in range(nat[0], nat[0] + nat[1], chunk):
        n = min(chunk, nat[0] + nat[1] - c0)
        cached(out / "native_pool" / f"seeds_{c0}.pt", lambda: S().draw(cfg.classes[0], UNGUIDED, c0, n))


def load_split(cfg, split):
    import torch
    d = cfg.samples / split
    imgs = {"unguided": to01(torch.load(d / "unguided.pt", weights_only=False)),
            "native_pool": to01(torch.cat([torch.load(f, weights_only=False)
                                           for f in sorted((d / "native_pool").glob("seeds_*.pt"),
                                                           key=lambda p: int(p.stem.split("_")[1]))]))}
    for k in cfg.classes:
        imgs[f"steered_{k}"] = to01(torch.load(d / f"steered_{k}.pt", weights_only=False))
        imgs[f"random_{k}"] = to01(torch.load(d / f"random_{k}.pt", weights_only=False))
    return imgs


def describe(cfg, split, ev):
    """Meters, evaluator predictions and features for every image set of a split (cached as .npz)."""
    f = cfg.out / f"{split}_measures.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z["data"].item()
    imgs = load_split(cfg, split)
    data = {}
    for name, x in imgs.items():
        pred, feat = ev(x)
        data[name] = {"x": x, "pred": pred, "feat": feat, "m": image_meters(x)}
        print(f"   {split}/{name}: {len(x)} images; assigned to own class "
              f"{np.mean(pred == int(name.split('_')[1])) * 100 if name.startswith('steered') else float('nan'):.1f}%",
              flush=True)
    np.savez_compressed(f, data=np.array(data, dtype=object))
    return data


def real_refs():
    pixels, labels = R.cifar_train()
    first = {k: np.flatnonzero(labels == k)[:1000] for k in range(10)}   # the RFM training images
    return {k: pixels[first[k]].astype(np.float32) / 255 for k in range(10)}


# ---------------------------------------------------------------- effects
def effect(a, b, sd, rng, paired=False):
    """(mean a - mean b) / sd with a bootstrap 95% interval."""
    est = (a.mean() - b.mean()) / sd
    boots = []
    for _ in range(N_BOOT):
        if paired:
            i = rng.integers(0, len(a), len(a))
            boots.append((a[i].mean() - b[i].mean()) / sd)
        else:
            boots.append((a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()) / sd)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def set_effect(xa, fa, xb, fb, rng, n_boot=200):
    """ratio of set meters a / b - 1, with a bootstrap interval."""
    ma, mb = set_meters(xa, fa, rng), set_meters(xb, fb, rng)
    out = {}
    boots = {k: [] for k in SET_METERS}
    for _ in range(n_boot):
        # resample, then drop repeats: a repeated image adds zero distances and would bias diversity down
        ia = np.unique(rng.integers(0, len(xa), len(xa)))
        ib = np.unique(rng.integers(0, len(xb), len(xb)))
        sa, sb = set_meters(xa[ia], fa[ia], rng), set_meters(xb[ib], fb[ib], rng)
        for k in SET_METERS:
            boots[k].append(sa[k] / sb[k] - 1)
    for k in SET_METERS:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[k] = (float(ma[k] / mb[k] - 1), float(lo), float(hi))
    return out


def flagged(e, tau):
    est, lo, hi = e
    return abs(est) > tau and (lo > 0 or hi < 0)


def side_effects(steered, native, ref_sd, rng):
    """X_c(m) for all meters: steered images assigned to c vs. native_c."""
    assert len(steered["x"]) >= 2 and len(native["x"]) >= 2, (len(steered["x"]), len(native["x"]))
    res = {}
    for m in IMAGE_METERS:
        res[m] = effect(steered["m"][m], native["m"][m], ref_sd[m], rng)
    res.update(set_effect(steered["x"], steered["feat"], native["x"], native["feat"], rng,
                          n_boot=100 if len(steered["x"]) > 300 else 200))
    return res


def subset(d, idx):
    return {"x": d["x"][idx], "pred": d["pred"][idx], "feat": d["feat"][idx],
            "m": {k: v[idx] for k, v in d["m"].items()}}


def tau_of(m):
    return TAU_SET if m in SET_METERS else TAU


# ---------------------------------------------------------------- stages
def stage_meters(cfg):
    """Positive controls on real CIFAR-10 images: each meter should respond to the edit it is meant to measure."""
    from scipy.ndimage import gaussian_filter
    pixels, labels = R.cifar_train()
    x = pixels[np.random.default_rng(0).choice(len(pixels), 1000, replace=False)].astype(np.float32) / 255
    base = image_meters(x)
    sd = {k: v.std() for k, v in base.items()}
    y = luma(x)[..., None]
    m = ring(x)
    flat_bg = x.copy()
    flat_bg[:, m] = x[:, m].mean(1, keepdims=True)
    edits = {
        "brighter (+0.1)": np.clip(x + 0.1, 0, 1),
        "desaturate (halfway to grey)": 0.5 * x + 0.5 * y,
        "bluer (+0.08 blue)": np.clip(x + np.array([0, 0, 0.08], np.float32), 0, 1),
        "greener (+0.08 green)": np.clip(x + np.array([0, 0.08, 0], np.float32), 0, 1),
        "lower contrast (x0.6 around the mean)": x.mean((1, 2, 3), keepdims=True) + 0.6 * (x - x.mean((1, 2, 3), keepdims=True)),
        "blur (Gaussian, 1 px)": np.stack([gaussian_filter(a, (1, 1, 0)) for a in x]),
        "flat background (border ring set to its mean)": flat_bg,
    }
    L = ["# Meter checks on 1,000 real CIFAR-10 training images", "",
         "Change in each meter, in SDs of the meter over the 1,000 originals. A meter is useful if it responds clearly "
         "to the edit it is named for and its responses to unrelated edits are understood.", "",
         "| edit | " + " | ".join(IMAGE_METERS) + " |", "|---|" + "---|" * len(IMAGE_METERS)]
    res = {}
    for name, xe in edits.items():
        me = image_meters(xe)
        row = {k: float((me[k] - base[k]).mean() / sd[k]) for k in IMAGE_METERS}
        res[name] = row
        L.append(f"| {name} | " + " | ".join(f"{row[k]:+.2f}" for k in IMAGE_METERS) + " |")
    # set meters: 500 mixed-class images vs. 500 images of one class (less variety, so all three should fall)
    rng = np.random.default_rng(0)
    ev = Evaluator()
    refs = real_refs()
    res["set_checks"] = {}
    L += ["", "Set meters: 500 real images of all classes vs. 500 of one class (less variety; all three should fall). "
          "Ratio one-class / mixed:", "", "| one class | " + " | ".join(SET_METERS) + " |", "|---|" + "---|" * len(SET_METERS)]
    mixed = x[:500]
    mm = set_meters(mixed, ev(mixed)[1], rng)
    for k in cfg.classes:
        one = refs[k][:500]
        mo = set_meters(one, ev(one)[1], rng)
        r = {m: mo[m] / mm[m] for m in SET_METERS}
        res["set_checks"][R.CLASSES[k]] = r
        L.append(f"| {R.CLASSES[k]} | " + " | ".join(f"{r[m]:.2f}" for m in SET_METERS) + " |")
    L += ["", "Meters: " + "; ".join(f"**{k}**: {v}" for k, v in METER_HELP.items())]
    (cfg.out / "meter_checks.md").write_text("\n".join(L) + "\n")
    (cfg.out / "meter_checks.json").write_text(json.dumps(res, indent=2))
    print("\n".join(L[:14]), flush=True)
    log("meters OK")


def stage_gen(cfg, split):
    log(f"generate {split}")
    generate(cfg, split)
    log(f"generate {split} OK")


def stage_predict(cfg):
    pf = cfg.out / "predictions.json"
    if pf.exists():
        log(f"predictions already frozen: {pf} (sha256 {hashlib.sha256(pf.read_bytes()).hexdigest()})")
        return
    ev = Evaluator()
    dev = describe(cfg, "dev", ev)
    refs = real_refs()
    rng = np.random.default_rng(1)
    ref_m = {k: image_meters(refs[k]) for k in cfg.classes}
    ref_sd = {k: {m: float(ref_m[k][m].std()) for m in IMAGE_METERS} for k in cfg.classes}
    ref_feat = {k: ev(refs[k])[1] for k in cfg.classes}
    pool = dev["native_pool"]
    pred = {"tau": TAU, "tau_set": TAU_SET, "classes": [R.CLASSES[k] for k in cfg.classes], "seeds": cfg.seeds,
            "meters": IMAGE_METERS + SET_METERS, "by_class": {}}
    n_pairs = len(dev["unguided"]["x"])
    for k in cfg.classes:
        st = dev[f"steered_{k}"]
        native = subset(pool, np.flatnonzero(pool["pred"] == k))
        real_k = {"x": refs[k], "feat": ref_feat[k], "m": ref_m[k], "pred": np.full(len(refs[k]), k)}
        c = {"native_n": int(len(native["x"])), "steered_accuracy": float(np.mean(st["pred"] == k))}
        for n in cfg.budgets:
            first = np.arange(n)
            st_n = subset(st, first[st["pred"][:n] == k])
            c[f"OUT_{n}"] = side_effects(st_n, native, ref_sd[k], rng)
            c[f"BOTH_{n}"] = side_effects(st_n, real_k, ref_sd[k], rng)
            c[f"n_{n}_used"] = int(len(st_n["x"]))
        # generic perturbation on dev: random direction vs. unguided, same seeds (paired), in the same units
        rnd, ung = dev[f"random_{k}"], dev["unguided"]
        c["RANDOM_dev"] = {m: effect(rnd["m"][m], ung["m"][m], ref_sd[k][m], rng, paired=True) for m in IMAGE_METERS}
        c["RANDOM_dev"].update(set_effect(rnd["x"], rnd["feat"], ung["x"], ung["feat"], rng))
        c["random_assigned_to_c"] = float(np.mean(rnd["pred"] == k))
        pred["by_class"][R.CLASSES[k]] = c
    big = f"OUT_{cfg.budgets[-1]}"
    for k in cfg.classes:
        others = [R.CLASSES[j] for j in cfg.classes if j != k]
        pred["by_class"][R.CLASSES[k]]["TRANSFER"] = {
            m: (float(np.mean([pred["by_class"][o][big][m][0] for o in others])), np.nan, np.nan)
            for m in IMAGE_METERS + SET_METERS}
    # flags: predicted side effects (TRANSFER: |mean| > tau, no interval)
    for name, c in pred["by_class"].items():
        c["flags"] = {}
        for src in [f"OUT_{n}" for n in cfg.budgets] + [f"BOTH_{n}" for n in cfg.budgets]:
            c["flags"][src] = [m for m in IMAGE_METERS + SET_METERS if flagged(c[src][m], tau_of(m))]
        c["flags"]["TRANSFER"] = [m for m in IMAGE_METERS + SET_METERS if abs(c["TRANSFER"][m][0]) > tau_of(m)]
        c["flags"]["REF"] = []
    text = json.dumps(pred, indent=1, default=float)
    pf.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (cfg.out / "predictions.sha256").write_text(digest + "\n")
    L = [f"# Frozen predictions ({time.strftime('%F %T')}), sha256 {digest}", "",
         f"Side effect = steered images assigned to the class vs. the model's own images of the class (native), in SDs "
         f"of real class images; flagged if |effect| > {TAU} (set meters: |ratio - 1| > {TAU_SET}) and the 95% interval "
         "excludes 0. RANDOM_dev: random direction vs. unguided on the same seeds (generic perturbation).", ""]
    srcs = [f"OUT_{n}" for n in cfg.budgets] + [f"BOTH_{n}" for n in cfg.budgets] + ["TRANSFER", "RANDOM_dev"]
    for name, c in pred["by_class"].items():
        L += [f"## {name}: steered accuracy {c['steered_accuracy']:.1%}, native images in the dev pool "
              f"{c['native_n']}, random direction assigned to {name} {c['random_assigned_to_c']:.1%}", "",
              "| meter | " + " | ".join(srcs) + " |", "|---|" + "---|" * len(srcs)]
        for m in IMAGE_METERS + SET_METERS:
            cells = []
            for src in srcs:
                if src not in c:
                    cells.append("–")
                    continue
                e = c[src][m]
                mark = "**" if src in c["flags"] and m in c["flags"][src] else ""
                ci = "" if np.isnan(e[1]) else f" ({e[1]:+.2f}, {e[2]:+.2f})"
                cells.append(f"{mark}{e[0]:+.2f}{mark}{ci}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    (cfg.out / "predictions.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    log(f"predictions frozen, sha256 {digest}")


def stage_evaluate(cfg):
    pf = cfg.out / "predictions.json"
    text = pf.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert digest == (cfg.out / "predictions.sha256").read_text().strip(), "predictions changed after freezing"
    pred = json.loads(text)
    ev = Evaluator()
    test = describe(cfg, "test", ev)
    refs = real_refs()
    rng = np.random.default_rng(2)
    pool = test["native_pool"]
    truth, rows = {}, []
    sources = list(next(iter(pred["by_class"].values()))["flags"].keys())
    for k in cfg.classes:
        name = R.CLASSES[k]
        ref_sd = {m: float(image_meters(refs[k])[m].std()) for m in IMAGE_METERS}
        st = test[f"steered_{k}"]
        native = subset(pool, np.flatnonzero(pool["pred"] == k))
        st_c = subset(st, np.flatnonzero(st["pred"] == k))
        t = side_effects(st_c, native, ref_sd, rng)
        rnd, ung = test[f"random_{k}"], test["unguided"]
        gen = {m: effect(rnd["m"][m], ung["m"][m], ref_sd[m], rng, paired=True) for m in IMAGE_METERS}
        gen.update(set_effect(rnd["x"], rnd["feat"], ung["x"], ung["feat"], rng))
        confirmed = [m for m in IMAGE_METERS + SET_METERS if flagged(t[m], tau_of(m))]
        generic = [m for m in confirmed if np.sign(gen[m][0]) == np.sign(t[m][0]) and abs(gen[m][0]) >= 0.5 * abs(t[m][0])]
        truth[name] = {"effects": t, "random": gen, "confirmed": confirmed, "generic": generic,
                       "native_n": int(len(native["x"])), "steered_n": int(len(st_c["x"])),
                       "steered_accuracy": float(np.mean(st["pred"] == k))}
        for src in sources:
            fl = set(pred["by_class"][name]["flags"][src])
            conf = set(confirmed)
            rows.append({"class": name, "source": src, "flagged": len(fl), "confirmed": len(conf),
                         "hits": len(fl & conf), "false_alarms": len(fl - conf), "missed": len(conf - fl),
                         "hits_specific": len(fl & (conf - set(generic)))})
    # rank agreement of predicted and true effect sizes, pooled over classes and meters (per source)
    agree = {}
    for src in [f"OUT_{n}" for n in cfg.budgets] + [f"BOTH_{n}" for n in cfg.budgets] + ["TRANSFER"]:
        p, t = [], []
        for name in truth:
            for m in IMAGE_METERS:
                p.append(pred["by_class"][name][src][m][0])
                t.append(truth[name]["effects"][m][0])
        from scipy.stats import spearmanr
        agree[src] = {"spearman": float(spearmanr(p, t)[0]),
                      "sign_agreement": float(np.mean(np.sign(p) == np.sign(t))),
                      "median_abs_error": float(np.median(np.abs(np.array(p) - np.array(t))))}
    (cfg.out / "truth.json").write_text(json.dumps({"truth": truth, "rows": rows, "agreement": agree,
                                                    "predictions_sha256": digest}, indent=1, default=float))
    report(cfg, pred, truth, rows, agree, digest)
    blind_sheets(cfg, test)
    log("evaluate OK")


def report(cfg, pred, truth, rows, agree, digest):
    L = ["# Q2 round 5: predicting steering side effects on new seeds (CIFAR-10, RFM-only)", "",
         f"Predictions frozen before the test images were generated (sha256 {digest}).", "",
         "## Confirmed side effects on the test seeds", "",
         f"Steered images assigned to the class vs. the model's own images of the class; confirmed if |effect| > {TAU} "
         f"SD of real class images (set meters: |ratio - 1| > {TAU_SET}) and the 95% interval excludes 0. Generic: the "
         "random direction moves the meter the same way by at least half as much.", "",
         "| class | steered accuracy | steered / native images | confirmed | of which generic |", "|---|---|---|---|---|"]
    for name, t in truth.items():
        L.append(f"| {name} | {t['steered_accuracy']:.1%} | {t['steered_n']} / {t['native_n']} | "
                 f"{', '.join(t['confirmed']) or 'none'} | {', '.join(t['generic']) or 'none'} |")
    L += ["", "## How well each prediction source did (summed over classes)", "",
          "| source | flagged | hits | false alarms | missed | hits that are not generic |", "|---|---|---|---|---|---|"]
    srcs = []
    for r in rows:
        if r["source"] not in srcs:
            srcs.append(r["source"])
    for s in srcs:
        rs = [r for r in rows if r["source"] == s]
        L.append(f"| {s} | {sum(r['flagged'] for r in rs)} | {sum(r['hits'] for r in rs)} | "
                 f"{sum(r['false_alarms'] for r in rs)} | {sum(r['missed'] for r in rs)} | "
                 f"{sum(r['hits_specific'] for r in rs)} |")
    L += ["", "Effect-size agreement over the 10 per-image meters x classes (Spearman, sign agreement, median |error| "
          "in SDs):", ""]
    for s, a in agree.items():
        L.append(f"- {s}: {a['spearman']:+.2f}, {a['sign_agreement']:.0%}, {a['median_abs_error']:.2f}")
    L += ["", "## Effects per class (test truth, with the random-direction shift)", ""]
    small, large = cfg.budgets
    cols = [f"OUT_{small}", f"OUT_{large}", f"BOTH_{large}", "TRANSFER"]
    for name, t in truth.items():
        L += [f"### {name}", "", "| meter | true effect (95% interval) | " + " | ".join(cols) + " | random |",
              "|---|---|" + "---|" * len(cols) + "---|"]
        p = pred["by_class"][name]
        for m in IMAGE_METERS + SET_METERS:
            e = t["effects"][m]
            mark = "**" if m in t["confirmed"] else ""
            L.append(f"| {m} | {mark}{e[0]:+.2f}{mark} ({e[1]:+.2f}, {e[2]:+.2f}) | "
                     + " | ".join(f"{p[c][m][0]:+.2f}" for c in cols) + f" | {t['random'][m][0]:+.2f} |")
        L.append("")
    (cfg.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def blind_sheets(cfg, test):
    """Per class: 12 steered and 12 native test images, shuffled; the key is saved separately (24 images a sheet)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    key = {}
    pool = test["native_pool"]
    for k in cfg.classes:
        st = test[f"steered_{k}"]
        ok = np.flatnonzero(st["pred"] == k)
        si = rng.choice(ok, min(12, len(ok)), replace=False)
        ni = rng.choice(np.flatnonzero(pool["pred"] == k), min(12, int((pool["pred"] == k).sum())), replace=False)
        items = [("steered", st["x"][i]) for i in si] + [("native", pool["x"][i]) for i in ni]
        order = rng.permutation(len(items))
        fig, axes = plt.subplots(4, 6, figsize=(9, 6.4))
        for ax in axes.flat:
            ax.axis("off")
        for pos, (ax, j) in enumerate(zip(axes.flat, order)):
            ax.imshow(items[j][1], interpolation="nearest")
            ax.set_title(str(pos + 1), fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{R.CLASSES[k]}: 24 images, two sources mixed", fontsize=10)
        fig.tight_layout()
        fig.savefig(cfg.out / f"blind_{R.CLASSES[k]}.png", dpi=120)
        plt.close(fig)
        key[R.CLASSES[k]] = {str(pos + 1): items[j][0] for pos, j in enumerate(order)}
    (cfg.out / "blind_key.json").write_text(json.dumps(key, indent=1))


# ---------------------------------------------------------------- main
cfg_global = None


def main():
    global cfg_global
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["meters", "gen_dev", "predict", "gen_test", "evaluate"])
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    cfg.classes = AUDIT_CLASSES[:2] if cfg.smoke else AUDIT_CLASSES
    cfg.seeds = SMOKE_SEEDS if cfg.smoke else SEEDS
    cfg.budgets = SMOKE_BUDGETS if cfg.smoke else OUT_BUDGETS
    cfg.out = ROOT / "outputs" / ("q2_cifar_audit_smoke" if cfg.smoke else "q2_cifar_audit")
    cfg.samples = R.WORK / ("samples_audit_smoke" if cfg.smoke else "samples_audit")
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.samples.mkdir(parents=True, exist_ok=True)
    cfg_global = cfg
    for s in cfg.stages:
        if s == "gen_test" and not (cfg.out / "predictions.sha256").exists():
            sys.exit("refusing to generate test images before predictions are frozen (run predict first)")
        t0 = time.time()
        {"meters": stage_meters, "gen_dev": lambda c: stage_gen(c, "dev"), "predict": stage_predict,
         "gen_test": lambda c: stage_gen(c, "test"), "evaluate": stage_evaluate}[s](cfg)
        log(f"stage {s} done [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
