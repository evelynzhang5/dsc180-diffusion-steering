#!/usr/bin/env python3
"""Replicate Table 2 of Wang, Belkin and Wang (2026) on CIFAR-10 with the paper's public code.

Stages (each skips work already on disk, so a rerun resumes):
  setup    clone na-rfm and TFG at fixed commits, apply the two code fixes, download the checkpoint and CIFAR-10
  offline  Gaussian denoisers (PCA, 1,000 directions) and RFM directions for all ten classes
  generate five settings, 256 images per class, seeds 0-255, 100 DDIM steps
  measure  accuracy (ConvNeXt-Tiny evaluator) and per-class FID against each class's 6,000 CIFAR-10 images
  time     seconds per image at batch size 64 (and per batch of 16, Table 2's setting)
  report   table next to the paper's Table 2, per-class tables, sample grids, PCA and AUC figures
  sweep    (encouraged) noise alignment only over a range of lambda: accuracy against FID
  big      (encouraged) NA-RFM and NA-only (lambda 3) at 2,048 images per class; FID also at 256, 512, 1,024

Usage, from the repo root, on a GPU:
  python cifar10_replication/replicate.py all            # setup offline generate measure time report
  python cifar10_replication/replicate.py all --smoke    # 2 classes, 8 images each, to a separate folder
  python cifar10_replication/replicate.py sweep big
Results (table, figures, JSON) go to cifar10_replication/results/; samples to downloads/narfm/ (not committed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "downloads" / "narfm"               # code, checkpoint, data and samples (not committed)
REPO = WORK / "na-rfm"
TFG = REPO / "baselines" / "tfg"
CKPT = REPO / "data" / "checkpoints" / "openai_cifar10.pt"
FILTERS = REPO / "repro" / "assets" / "cifar10_filters"
CONCEPTS = REPO / "repro" / "assets" / "cifar10_concepts"
BY_CLASS = WORK / "cifar10" / "train_by_class"

NARFM_COMMIT = "8e9619bc6735b03d73056bfdc13dd3659b96e127"   # na-rfm, Oct 27, 2026 (as in the starter notebook)
TFG_COMMIT = "f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77"     # Training-Free-Guidance, Oct 29, 2025
CKPT_MD5 = "2754c3e3c778ef6e0ab98e8aacaefaf9"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SIGMA_RFM = 0.21     # sigma of the RFM training images: the model's step t = 60

# The five settings. kind "narfm": run_cifar10_na_rfm.py with these flags; kind "na": run_cifar10_na_only.py --mode diff.
# Unguided: both windows are placed above every sigma the sampler visits (max ~157), so it is plain DDIM with one
# network call per step. It gives the same images as --gaussian-strength 0 --rfm-strength 0, at half the cost.
ROWS = {
    "unguided": ("narfm", "--gaussian-strength 0 --rfm-strength 0 --rfm-sigma-lo 1e8 --rfm-sigma-hi 1e9"),
    "na_l3": ("na", 3.0),
    "na_l8": ("na", 8.0),
    "rfm_only": ("narfm", "--gaussian-strength 0 --rfm-sigma-lo 0 --rfm-sigma-hi 1e4"),
    "na_rfm": ("narfm", ""),   # the scripts' defaults are the paper's: lambda 3 for sigma >= 3.33; w 1, s 2 on [0.01, 11.69]
}
LABELS = {"unguided": "Unguided (not in Table 2)", "na_l3": "Noise alignment only (λ = 3)",
          "na_l8": "Noise alignment only (λ = 8)", "rfm_only": "RFM only (all steps)", "na_rfm": "NA-RFM"}
PAPER = {  # Table 2: accuracy %, FID (2,048 images per class), seconds for 16 images on an A100
    "na_l3": (62.0, 99.0, 5.8), "na_l8": (80.0, 120.0, 5.8), "rfm_only": (94.8, 40.3, 6.8), "na_rfm": (96.6, 41.4, 6.2)}
PAPER_256 = {"rfm_only": 94.3, "na_rfm": 97.9}   # accuracy at 256 images per class (paper's Figures 8 and 7)
SWEEP = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


def run(cmd, cwd=None):
    print("$", cmd, flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, shell=True, cwd=cwd or REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [l for l in proc.stdout.splitlines() if "it/s]" not in l and "s/it]" not in l]
    print("\n".join(lines[-15:]), flush=True)
    if proc.returncode != 0:
        print("\n".join(lines[-80:]))
        raise RuntimeError(f"exit code {proc.returncode}: {cmd}")
    print(f"[{time.time() - t0:.0f} s]", flush=True)


def fix(path, old, new):
    text = path.read_text()
    if new not in text:
        assert old in text, f"fix target not found in {path}"
        path.write_text(text.replace(old, new))


def env_paths():
    os.environ["PYTHONPATH"] = f"{REPO}:{TFG}"
    for p in (str(TFG), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------- setup
def setup(cfg):
    WORK.mkdir(parents=True, exist_ok=True)
    if not REPO.exists():
        run(f"git clone -q https://github.com/isotrivial/na-rfm.git {REPO}", cwd=WORK)
    run(f"git checkout -q {NARFM_COMMIT}")
    if not TFG.exists():
        run(f"git clone -q https://github.com/YWolfeee/Training-Free-Guidance.git {TFG}")
    run(f"git checkout -q {TFG_COMMIT}", cwd=TFG)
    # Fix 1: the FID script loads CIFAR-10 under its old Hugging Face name.
    fix(REPO / "repro" / "evaluate_saved_samples.py", 'load_dataset("cifar10")', 'load_dataset("uoft-cs/cifar10")')
    # Fix 2: TFG's sqrtm imports a private SciPy module that fails in SciPy 1.16 unless another loads first.
    imp = "from scipy.linalg._matfuncs_sqrtm_triu import within_block_loop"
    fix(TFG / "evaluations" / "utils" / "torch_sqrtm.py", imp, "import scipy.linalg._matfuncs_sqrtm\n" + imp)

    if not CKPT.exists():
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        run(f"{sys.executable} -m gdown 1eInndCjx389Tl45kfmntVo23f_-82bHl -O {CKPT}")
    md5 = hashlib.md5(CKPT.read_bytes()).hexdigest()
    assert md5 == CKPT_MD5, f"not the paper's checkpoint (md5 {md5})"

    if not (BY_CLASS / "label_9").exists():
        from PIL import Image
        pixels, labels = cifar_train()
        for i, (img, y) in enumerate(zip(pixels, labels)):
            d = BY_CLASS / f"label_{y}"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(d / f"{i:05d}.png")
    counts = {CLASSES[k]: len(list((BY_CLASS / f"label_{k}").glob("*.png"))) for k in range(10)}
    assert all(v == 5000 for v in counts.values()), counts
    log("setup OK")


def cifar_train():
    cache = WORK / "cifar10" / "train.npz"
    if cache.exists():
        z = np.load(cache)
        return z["pixels"], z["labels"]
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", split="train")
    assert ds.features["label"].names == CLASSES
    labels = np.array(ds["label"])
    pixels = np.stack([np.asarray(img) for img in ds["img"]])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pixels=pixels, labels=labels)
    return pixels, labels


# ---------------------------------------------------------------- offline parts
def offline(cfg):
    env_paths()
    import torch
    build = (f"{sys.executable} pca/build_filters.py --image-root {BY_CLASS} --output-dir {FILTERS} "
             f"--image-size 32 --components 1000 --backend cpu")
    if not (FILTERS / "full_dataset.pt").exists():
        run(f"{build} --full-dataset --num-images 0")                 # D_all: all 50,000 images
    missing = [k for k in cfg.classes if not (FILTERS / f"class_{k}.pt").exists()]
    if missing:
        run(f"{build} --classes {' '.join(map(str, missing))} --num-images 5000")   # D_k: each class's 5,000 images
    for k in cfg.classes:
        f = torch.load(FILTERS / f"class_{k}.pt", weights_only=False)
        assert f["num_images"] == 5000 and f["eigenvectors"].shape[1] == 1000, (k, f["num_images"])

    missing = [k for k in cfg.classes if not (CONCEPTS / f"class_{k}.pt").exists()]
    if missing:
        os.chdir(REPO)
        from torch.utils.data import DataLoader, TensorDataset
        from na_rfm.activation_collection import collect_activations
        from na_rfm.model_loaders import load_openai_unet
        from na_rfm.rfm_training import RFMTrainConfig, get_direction, train_binary_rfm
        pixels, labels = cifar_train()
        first = np.concatenate([np.flatnonzero(labels == k)[:1000] for k in range(10)])   # first 1,000 of each class
        images = torch.from_numpy(pixels[first]).permute(0, 3, 1, 2).float() / 127.5 - 1
        loader = DataLoader(TensorDataset(images, torch.from_numpy(labels[first])), batch_size=250)
        unet = load_openai_unet(str(CKPT), image_size=32, device="cuda")
        torch.manual_seed(0)
        H, Hy = collect_activations(unet, loader, block_name="input_blocks_9",
                                    alpha_prod_t=1 / (1 + SIGMA_RFM**2), timestep=60, device="cuda")
        del unet
        torch.cuda.empty_cache()
        print("block-9 features:", tuple(H.shape), flush=True)
        CONCEPTS.mkdir(parents=True, exist_ok=True)
        conf = RFMTrainConfig(bandwidth=100.0, reg=1e-3, n_iters=5, top_k=3, device="cuda")
        for k in missing:
            t0 = time.time()
            res = train_binary_rfm(H, Hy, target_value=k, config=conf)
            v = get_direction(res, topk=3).cpu()
            torch.save({"direction": v, "eigenvectors": res.eigenvectors.cpu(), "eigenvalues": res.eigenvalues.cpu(),
                        "sign": int(res.sign), "auc": float(res.best_auc), "block_name": "input_blocks_9",
                        "timestep": 60, "sigma": SIGMA_RFM}, CONCEPTS / f"class_{k}.pt")
            print(f"RFM {CLASSES[k]} vs rest: held-out AUC {res.best_auc:.4f} ({time.time() - t0:.0f} s)", flush=True)
            torch.cuda.empty_cache()
    aucs = {CLASSES[k]: float(torch.load(CONCEPTS / f"class_{k}.pt", weights_only=False)["auc"]) for k in cfg.classes}
    (cfg.out / "rfm_auc.json").write_text(json.dumps(aucs, indent=2))
    log(f"offline OK; AUCs {aucs}")


# ---------------------------------------------------------------- generate
def row_dir(cfg, row):
    return cfg.samples / row


def generate_row(cfg, row, kind, spec):
    import torch
    out = row_dir(cfg, row)
    for k in cfg.classes:
        cdir = out / CLASSES[k]
        if (cdir / "samples.pt").exists() and (cdir / "metrics.json").exists():
            continue
        t0 = time.time()
        if kind == "narfm":
            run(f"{sys.executable} repro/run_cifar10_na_rfm.py --classes {k} --num-samples {cfg.n} --seed-start 0 "
                f"--batch-size 64 --inference-steps 100 --eval-mode hf_convnext --output-dir {out} {spec}")
        else:   # noise alignment only; its script saves one file per class, so copy it into <row>/<class>/samples.pt
            raw = cfg.samples / "_raw" / row
            run(f"{sys.executable} repro/run_cifar10_na_only.py --class-idx {k} --mode diff --strength {spec} "
                f"--num-seeds {cfg.n} --inference-steps 100 --num-components 1000 --start-step 0 --end-step 50 "
                f"--batch-size 64 --eval-mode hf_convnext --filters-dir {FILTERS} --output-dir {raw}")
            f = next(raw.glob(f"{CLASSES[k]}_mode=diff_str={spec}_seeds={cfg.n}_start=0_end=50.pt"))
            d = torch.load(f, weights_only=False)
            preds = torch.as_tensor(d["predictions"]).long()
            cdir.mkdir(parents=True, exist_ok=True)
            torch.save(d["samples"], cdir / "samples.pt")
            (cdir / "metrics.json").write_text(json.dumps({
                "class_idx": k, "class_name": CLASSES[k], "lambda": spec, "eval_mode": d["eval_mode"],
                "correct": int((preds == k).sum()), "total": int(len(preds)), "accuracy": float(d["accuracy"]),
                "predictions": preds.tolist()}, indent=2))
        m = json.loads((cdir / "metrics.json").read_text())
        print(f"   {row} {CLASSES[k]}: accuracy {m['accuracy']:.1f}% [{time.time() - t0:.0f} s]", flush=True)


def generate(cfg):
    env_paths()
    for row, (kind, spec) in ROWS.items():
        log(f"generate {row}")
        generate_row(cfg, row, kind, spec)


# ---------------------------------------------------------------- measure
def measure_row(cfg, row):
    q = row_dir(cfg, row) / "quality.json"
    targets = ",".join(map(str, cfg.classes))
    if q.exists() and json.loads(q.read_text()).get("_targets") == targets:
        return
    run(f"{sys.executable} repro/evaluate_saved_samples.py --dataset cifar10 --samples-root {row_dir(cfg, row)} "
        f"--targets {targets} --output {q} --batch-size 64 --num-ref {cfg.num_ref}")
    d = json.loads(q.read_text())
    d["_targets"] = targets
    q.write_text(json.dumps(d, indent=2))
    print(f"   {row}: accuracy {d.get('average_accuracy', float('nan')):.1f}%, FID {d['average_fid']:.1f}", flush=True)


def measure(cfg):
    env_paths()
    for row in ROWS:
        log(f"measure {row}")
        measure_row(cfg, row)


# ---------------------------------------------------------------- time
def time_rows(cfg):
    env_paths()
    os.chdir(REPO)
    import torch
    from na_rfm.ddim_sampling import build_ddim_schedule
    from na_rfm.model_loaders import load_openai_unet
    from repro import run_cifar10_na_only as na_only
    from repro.run_cifar10_na_rfm import parse_args, sample_class
    car = 1 if 1 in cfg.classes else cfg.classes[0]
    unet = load_openai_unet(str(CKPT), image_size=32, device="cuda")

    def seconds(kind, spec, batch, repeats):
        if kind == "narfm":
            args = parse_args(f"--classes {car} --num-samples {batch} --batch-size {batch} {spec}".split())
            sched = build_ddim_schedule(1000, args.inference_steps, 0.0001, 0.02, "cuda")
            draw = lambda: sample_class(unet=unet, schedule=sched, class_idx=car, args=args)
        else:
            sched = na_only.build_ddim_schedule(1000, 100, 0.0001, 0.02, "cuda")
            g = na_only.WienerGuidance(str(FILTERS / f"class_{car}.pt"), str(FILTERS / "full_dataset.pt"), mode="diff",
                                       strength=spec, start_step=0, end_step=50, num_components=1000, device="cuda")
            draw = lambda: na_only.sample_with_wiener_guidance(
                unet, sched, g, batch, na_only.StackedRandomGenerator("cuda", list(range(batch))))
        ts = []
        for _ in range(repeats + 1):   # the first batch warms up
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                draw()
            torch.cuda.synchronize()
            ts.append(time.time() - t0)
        return min(ts[1:])

    res = {"gpu": torch.cuda.get_device_name(0)}
    reps = 1 if cfg.smoke else 2
    for row, (kind, spec) in ROWS.items():
        b64 = seconds(kind, spec, 64, reps)
        b16 = seconds(kind, spec, 16, reps)
        res[row] = {"s_per_image_b64": b64 / 64, "s_per_batch16": b16}
        print(f"   {row}: {b64 / 64:.3f} s per image at batch 64; {b16:.1f} s per batch of 16", flush=True)
    (cfg.out / "timing.json").write_text(json.dumps(res, indent=2))
    log("time OK")


# ---------------------------------------------------------------- report
def grid_figure(cfg, row, path, n=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    fig, axes = plt.subplots(len(cfg.classes), 1, figsize=(n * 0.55, len(cfg.classes) * 0.62))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, cfg.classes):
        s = torch.load(row_dir(cfg, row) / CLASSES[k] / "samples.pt", weights_only=False)[:n].clamp(-1, 1)
        strip = ((s + 1) / 2).permute(2, 0, 3, 1).reshape(32, -1, 3).numpy()   # 32 x (n*32) x 3
        ax.imshow(strip, interpolation="nearest")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(CLASSES[k], rotation=0, ha="right", va="center", fontsize=8)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(f"{LABELS[row]}: the first 16 images (seeds 0-15) of each class", fontsize=9)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pca_figure(cfg, path, pick=(1, 3)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    pick = [k for k in pick if k in cfg.classes] or cfg.classes[:1]
    rows = [(CLASSES[k], f"class_{k}.pt") for k in pick] + [("all images", "full_dataset.pt")]
    fig, axes = plt.subplots(len(rows), 9, figsize=(13, 1.55 * len(rows)))
    for r, (name, file) in enumerate(rows):
        f = torch.load(FILTERS / file, weights_only=False)
        mean = f["mean_patch"].numpy().reshape(3, 32, 32).transpose(1, 2, 0)
        axes[r, 0].imshow(np.clip(mean / 2 + 0.5, 0, 1))
        axes[r, 0].set_title(f"mean, {name}", fontsize=8)
        ev = f.get("eigenvalues")
        tot = float(ev.sum()) if ev is not None else None
        for j in range(8):
            u = f["eigenvectors"][:, j].numpy().reshape(3, 32, 32).transpose(1, 2, 0)
            axes[r, j + 1].imshow(np.clip(0.5 + 0.5 * u / np.abs(u).max(), 0, 1))
            share = f" ({100 * float(ev[j]) / tot:.1f}%)" if tot else ""
            axes[r, j + 1].set_title(f"direction {j + 1}{share}", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Gaussian denoisers: mean image and first PCA directions (grey = 0; % = share of the top-1,000 "
                 "variance)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fmt(x, nd=1, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}{'%' if pct else ''}"


def report(cfg):
    fig_dir = cfg.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    q = {r: json.loads((row_dir(cfg, r) / "quality.json").read_text()) for r in ROWS
         if (row_dir(cfg, r) / "quality.json").exists()}
    tim = json.loads((cfg.out / "timing.json").read_text()) if (cfg.out / "timing.json").exists() else {}
    aucs = json.loads((cfg.out / "rfm_auc.json").read_text())
    for r in q:   # save per-row summaries next to the report
        (cfg.out / f"quality_{r}.json").write_text(json.dumps(q[r], indent=2))

    L = [f"# CIFAR-10 replication of Table 2 ({cfg.n} images per class, seeds 0-{cfg.n - 1}, 100 DDIM steps)", ""]
    if cfg.smoke:
        L += ["**SMOKE TEST: too few images for any number here to mean anything.**", ""]
    L += [f"Classes: {', '.join(CLASSES[k] for k in cfg.classes)}. GPU: {tim.get('gpu', '?')}.",
          f"FID: each class against its {'6,000' if cfg.num_ref < 0 else cfg.num_ref} CIFAR-10 images (train + "
          "test), averaged over classes. The paper's FID used 2,048 images per class, so its FIDs are lower than "
          "ours for that reason alone; compare our rows with each other.", "",
          "| Setting | Accuracy, ours | Accuracy, paper | FID, ours | FID, paper | s / image, batch 64 (ours) "
          "| s / 16 images (ours) | s / 16 images, paper (A100) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ROWS:
        p = PAPER.get(r, (None, None, None))
        acc = q[r].get("average_accuracy") if r in q else None
        fid = q[r]["average_fid"] if r in q else None
        t = tim.get(r, {})
        pacc = fmt(p[0], 1, True) + (f" ({PAPER_256[r]:.1f}% at 256/class)" if r in PAPER_256 else "")
        L.append(f"| {LABELS[r]} | {fmt(acc, 1, True)} | {pacc} | {fmt(fid)} | {fmt(p[1])} | "
                 f"{fmt(t.get('s_per_image_b64'), 3)} | {fmt(t.get('s_per_batch16'))} | {fmt(p[2])} |")
    L += ["", "## Per class", "", "Accuracy (%) / FID", "",
          "| Class | RFM AUC | " + " | ".join(LABELS[r] for r in ROWS) + " |", "|---|---|" + "---|" * len(ROWS)]
    for i, k in enumerate(cfg.classes):
        cells = []
        for r in ROWS:
            c = q[r]["classes"][i] if r in q else None
            cells.append(f"{fmt(c.get('accuracy'), 0)} / {fmt(c['fid'], 0)}" if c else "–")
        L.append(f"| {CLASSES[k]} | {aucs.get(CLASSES[k], float('nan')):.3f} | " + " | ".join(cells) + " |")
    L += ["", "RFM AUC: held-out (20%) AUC of each class's RFM classifier (class against the other nine) on block-9 "
          "features of 1,000 training images per class at σ = 0.21.", ""]

    for r in ROWS:
        if all((row_dir(cfg, r) / CLASSES[k] / "samples.pt").exists() for k in cfg.classes):
            grid_figure(cfg, r, fig_dir / f"grid_{r}.png")
            L.append(f"![{LABELS[r]}](figures/grid_{r}.png)")
    pca_figure(cfg, fig_dir / "pca_mean_directions.png")
    L += ["", "![PCA](figures/pca_mean_directions.png)", ""]
    sw = cfg.out / "sweep.json"
    if sw.exists():
        L += sweep_section(cfg, json.loads(sw.read_text()))
    bj = cfg.out / "big.json"
    if bj.exists():
        L += big_section(cfg, json.loads(bj.read_text()))
    (cfg.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:20]))
    log(f"report written to {cfg.out / 'report.md'}")


# ---------------------------------------------------------------- sweep (encouraged)
def sweep(cfg):
    env_paths()
    res = {}
    for lam in SWEEP:
        row = {3.0: "na_l3", 8.0: "na_l8"}.get(lam, f"na_l{lam:g}")
        LABELS.setdefault(row, f"Noise alignment only (λ = {lam:g})")
        log(f"sweep lambda {lam:g}")
        generate_row(cfg, row, "na", lam)
        measure_row(cfg, row)
        d = json.loads((row_dir(cfg, row) / "quality.json").read_text())
        res[f"{lam:g}"] = {"accuracy": d.get("average_accuracy"), "fid": d["average_fid"]}
    for r in ("unguided", "rfm_only", "na_rfm"):
        p = row_dir(cfg, r) / "quality.json"
        if p.exists():
            d = json.loads(p.read_text())
            res[r] = {"accuracy": d.get("average_accuracy"), "fid": d["average_fid"]}
    (cfg.out / "sweep.json").write_text(json.dumps(res, indent=2))
    report(cfg)


def sweep_section(cfg, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lams = [k for k in res if k[0].isdigit()]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    xs, ys = [res[k]["fid"] for k in lams], [res[k]["accuracy"] for k in lams]
    ax.plot(xs, ys, "o-", color="#2b6cb0", label="noise alignment only")
    for k, x, y in zip(lams, xs, ys):
        ax.annotate(f"λ={k}", (x, y), textcoords="offset points", xytext=(5, -10), fontsize=7)
    for r, mk in (("unguided", "s"), ("rfm_only", "^"), ("na_rfm", "D")):
        if r in res:
            ax.plot(res[r]["fid"], res[r]["accuracy"], mk, label=LABELS[r], color="#333")
    ax.set_xlabel(f"mean FID over classes ({cfg.n} images per class; lower is better)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.out / "figures" / "sweep_acc_fid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    L = ["## Noise alignment only: accuracy against FID over λ", "", "| λ | Accuracy | FID |", "|---|---|---|"]
    L += [f"| {k} | {fmt(res[k]['accuracy'], 1, True)} | {fmt(res[k]['fid'])} |" for k in lams]
    return L + ["", "![sweep](figures/sweep_acc_fid.png)", ""]


# ---------------------------------------------------------------- 2,048 images per class (encouraged)
BIG_ROWS = ["na_rfm", "na_l3"]


def big(cfg):
    """NA-RFM and noise alignment only (lambda 3) at the paper's 2,048 images per class, seeds 0-2047. FID is also
    measured on the first 256, 512 and 1,024 images of each class (the same images, nested), which shows how FID
    depends on the number of images."""
    env_paths()
    import torch
    sizes = [8, 16] if cfg.smoke else [256, 512, 1024, 2048]
    sub = argparse.Namespace(**vars(cfg))
    sub.samples = WORK / ("samples_big_smoke" if cfg.smoke else "samples_big")
    sub.n = sizes[-1]
    res = {}
    for row in BIG_ROWS:
        full = f"{row}_n{sizes[-1]}"
        log(f"big: {row} at {sizes[-1]} images per class")
        generate_row(sub, full, *ROWS[row])
        for m in sizes[:-1]:
            for k in cfg.classes:
                src, dst = sub.samples / full / CLASSES[k], sub.samples / f"{row}_n{m}" / CLASSES[k]
                if (dst / "metrics.json").exists():
                    continue
                dst.mkdir(parents=True, exist_ok=True)
                torch.save(torch.load(src / "samples.pt", weights_only=False)[:m].clone(), dst / "samples.pt")
                meta = json.loads((src / "metrics.json").read_text())
                preds = meta["predictions"][:m]
                meta.pop("target_confidence", None)   # an average over all images; not valid for the subset
                meta.update(num_samples=m, predictions=preds, total=m, correct=sum(int(p) == k for p in preds))
                meta["accuracy"] = 100.0 * meta["correct"] / m
                (dst / "metrics.json").write_text(json.dumps(meta, indent=2))
        res[row] = {}
        for m in sizes:
            log(f"measure {row}, first {m} images per class")
            measure_row(sub, f"{row}_n{m}")
            d = json.loads((sub.samples / f"{row}_n{m}" / "quality.json").read_text())
            res[row][str(m)] = {"accuracy": d.get("average_accuracy"), "fid": d["average_fid"],
                                "class_fid": {c["name"]: c["fid"] for c in d["classes"]},
                                "class_accuracy": {c["name"]: c.get("accuracy") for c in d["classes"]}}
        (cfg.out / "big.json").write_text(json.dumps(res, indent=2))
    report(cfg)


def big_section(cfg, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    colors = {"na_rfm": "#2b6cb0", "na_l3": "#c05621"}
    for row, r in res.items():
        ns = sorted(r, key=int)
        ax.plot([int(n) for n in ns], [r[n]["fid"] for n in ns], "o-", color=colors.get(row), label=f"{LABELS[row]}, ours")
        if not cfg.smoke:
            ax.plot([2048], [PAPER[row][1]], "D", mfc="none", color=colors.get(row), label=f"{LABELS[row]}, paper")
    ns = sorted(next(iter(res.values())), key=int)
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(n) for n in ns], ns)
    ax.set_xlabel("images per class (the first n of the same samples)")
    ax.set_ylabel("mean FID over classes")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cfg.out / "figures" / "fid_vs_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    L = [f"## At up to {ns[-1]} images per class (seeds 0-{int(ns[-1]) - 1})", "",
         "The same images throughout: each row uses the first n images of each class. The paper's FIDs are at 2,048.",
         "", "| Setting | Images per class | Accuracy | FID, ours | FID, paper |", "|---|---|---|---|---|"]
    for row, r in res.items():
        for n in sorted(r, key=int):
            paper = fmt(PAPER[row][1]) if n == "2048" else "–"
            L.append(f"| {LABELS[row]} | {n} | {fmt(r[n]['accuracy'], 1, True)} | {fmt(r[n]['fid'])} | {paper} |")
    for row, r in res.items():
        top = ns[-1]
        L += ["", f"{LABELS[row]}, per class at {top} images (accuracy / FID): " + ", ".join(
            f"{c} {fmt(r[top]['class_accuracy'][c], 0, True)} / {fmt(r[top]['class_fid'][c], 0)}"
            for c in r[top]["class_fid"]) + "."]
    return L + ["", "![FID against images per class](figures/fid_vs_images.png)", ""]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+",
                    choices=["setup", "offline", "generate", "measure", "time", "report", "sweep", "big", "all"])
    ap.add_argument("--smoke", action="store_true", help="2 classes, 8 images each, 500 FID references")
    ap.add_argument("--out", type=Path, default=None)
    cfg = ap.parse_args()
    cfg.classes = [1, 3] if cfg.smoke else list(range(10))
    cfg.n = 8 if cfg.smoke else 256
    cfg.num_ref = 500 if cfg.smoke else -1
    cfg.out = cfg.out or (ROOT / "outputs" / "cifar10_replication_smoke" if cfg.smoke else HERE / "results")
    cfg.out = cfg.out.resolve()
    cfg.samples = WORK / ("samples_smoke" if cfg.smoke else "samples")
    cfg.out.mkdir(parents=True, exist_ok=True)
    stages = ["setup", "offline", "generate", "measure", "time", "report"] if "all" in cfg.stages else cfg.stages
    funcs = {"setup": setup, "offline": offline, "generate": generate, "measure": measure, "time": time_rows,
             "report": report, "sweep": sweep, "big": big}
    for s in stages:
        log(f"stage {s}{' (smoke)' if cfg.smoke else ''}")
        funcs[s](cfg)
    log("done")


if __name__ == "__main__":
    main()
