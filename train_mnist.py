"""MNIST CFG network for the checkpoint: train the lab's U-Net from scratch, then measure CFG.

    python train_mnist.py train                     # train on the 60,000 MNIST training digits
    python train_mnist.py eval                      # label accuracy and FID for w = 0, 1, 2, 4
    python train_mnist.py eval --checkpoint generator.pt --out outputs/mnist_course   # the course's model

Pieces reused unchanged from mnist_guidance_lab.py (written by guidance_mnist.ipynb): the U-Net, the CFG sampler
u + w (c - u), and the frozen digit classifier, which only scores samples and never guides them.

Training follows the lab's convention: x0 is noise at t = 0, x1 is data at t = 1,
x_t = sin(pi t / 2) x1 + cos(pi t / 2) x0, and the network predicts the clean image x1. Each time an example is
drawn, its label is replaced by 10 ("no label") with probability 0.1, so one network gives both guesses.

FID here is computed on the classifier's 128 features, not Inception features, so it is not comparable with
CIFAR-10 FIDs or published MNIST FIDs. Compare runs only at the same number of samples.
"""
from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

import mnist_guidance_lab as mg

NO_LABEL = 10
# The course's frozen classifier (and its generator, for comparison), as listed in guidance_mnist.ipynb.
COURSE_FILES = {
    "evaluator.pt": (
        "https://drive.google.com/uc?export=download&id=1RkocEim9ejow8n76WI0LGpNF18K5FxlW",
        "05e955d1c14856991785c408f9a818b5d887cdc2c5142d63eac134367f50b7fe",
    ),
    "generator.pt": (
        "https://drive.google.com/uc?export=download&id=1ET8_47MTKGwFr2hbG7eEyQwfLhqw3ql3",
        "ec413c1df3fdaf1321059a9fbde2265fa10c6499b6c7e0b9031b218e74f99bb6",
    ),
}


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_mnist(root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """All MNIST images as one tensor in [-1, 1], shape (n, 1, 28, 28), and their labels."""
    from torchvision.datasets import MNIST

    ds = MNIST(root, train=train, download=True)
    x = ds.data.float().div(127.5).sub(1).unsqueeze(1)
    return x, ds.targets.clone()


def load_evaluator(folder: str, device: str) -> mg.Evaluator:
    mg.fetch({"evaluator.pt": COURSE_FILES["evaluator.pt"]}, folder)
    ck = torch.load(Path(folder) / "evaluator.pt", map_location="cpu", weights_only=True)
    ev = mg.Evaluator().to(device)
    ev.load_state_dict(ck["state"])
    return ev.eval().requires_grad_(False)


def load_generator(path: str, device: str) -> mg.UNet:
    path = Path(path)
    if path.name == "generator.pt" and not path.exists():
        mg.fetch({"generator.pt": COURSE_FILES["generator.pt"]}, path.parent)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = mg.UNet(ch=ck["ch"]).to(device)
    net.load_state_dict(ck["state"])
    return net.eval().requires_grad_(False)


# ------------------------------------------------------------------ training
def train(args) -> None:
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x_all, y_all = load_mnist(args.data, train=True)
    if args.limit:
        x_all, y_all = x_all[: args.limit], y_all[: args.limit]
    x_all, y_all = x_all.to(device), y_all.to(device)
    n = len(x_all)

    net = mg.UNet(ch=args.ch).to(device)
    ema = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(n / args.batch)
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(  # short warmup, then cosine decay to 10% of the peak rate
        opt, lambda s: min(1.0, (s + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / total))))
    print(f"device {device}, {n} training digits, {sum(p.numel() for p in net.parameters()):,} parameters, "
          f"{args.epochs} epochs = {total} steps")

    log, step, start = [], 0, time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            x1, y = x_all[idx], y_all[idx].clone()
            drop = torch.rand(len(y), device=device) < args.label_dropout
            y[drop] = NO_LABEL                                   # the 10% "no label" examples
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), device=device)
            a, s = mg.alpha_sigma(t[:, None, None, None])
            xt = a * x1 + s * x0
            loss = F.mse_loss(net(xt, t, y), x1)                 # predict the clean image, as the lab's net does

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                decay = min(args.ema, (1 + step) / (10 + step))
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(p, 1 - decay)
            running += loss.item()
            log.append({"step": step, "epoch": epoch, "loss": loss.item()})
            step += 1
        print(f"epoch {epoch + 1}/{args.epochs}  loss {running / steps_per_epoch:.4f}  "
              f"{(time.time() - start) / 60:.1f} min")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ck = {"ch": args.ch, "state": {k: v.cpu() for k, v in ema.state_dict().items()},
                  "epoch": epoch + 1, "step": step, "args": vars(args)}
            torch.save(ck, out / "mnist_cfg.pt")                  # same format as the course's generator.pt

    with open(out / "loss.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "epoch", "loss"])
        w.writeheader()
        w.writerows(log)
    losses = np.array([r["loss"] for r in log])
    k = max(1, len(losses) // 200)
    smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(6, 3.5))
    plt.plot(losses, alpha=0.25, lw=0.5, label="per step")
    plt.plot(np.arange(len(smooth)) + k - 1, smooth, label=f"mean of {k} steps")
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("MSE on the clean image")
    plt.title("MNIST CFG network: training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss.png", dpi=150)
    print(f"saved {out / 'mnist_cfg.pt'}, loss.csv and loss.png in {(time.time() - start) / 60:.1f} min")


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def features(evaluator, x, device, bs=500) -> tuple[np.ndarray, np.ndarray]:
    """The classifier's 128 features and its predicted digits."""
    feats, preds = [], []
    for i in range(0, len(x), bs):
        logits, f = evaluator(x[i : i + bs].to(device))
        feats.append(f.cpu().double())                       # MPS has no float64, so convert on the CPU
        preds.append(logits.argmax(1).cpu())
    return torch.cat(feats).numpy(), torch.cat(preds).numpy()


def fid(f1: np.ndarray, f2: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets.

    tr sqrt(C1 C2) is computed as the sum of square roots of the eigenvalues of sqrt(C1) C2 sqrt(C1), a symmetric
    matrix with the same eigenvalues. This stays accurate when some ReLU features are always zero, which makes the
    covariances singular."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    c1, c2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    vals, vecs = linalg.eigh(c1)
    s1 = (vecs * np.sqrt(vals.clip(min=0))) @ vecs.T
    tr_covmean = np.sqrt(linalg.eigvalsh(s1 @ c2 @ s1).clip(min=0)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * tr_covmean)


def save_grid(rows: dict, path: Path, per_row: int = 10) -> None:
    """One row per w; each row shows digits 0-9 (sample i requests digit i mod 10, same noise across rows)."""
    fig, axes = plt.subplots(len(rows), per_row, figsize=(1.0 * per_row, 1.1 * len(rows)), squeeze=False)
    for r, (name, images) in enumerate(rows.items()):
        for j, ax in enumerate(axes[r]):
            ax.imshow(images[j, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(args) -> None:
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = load_generator(args.checkpoint, device)
    evaluator = load_evaluator(args.weights, device)

    # Reference: classifier features of the 10,000 MNIST test digits.
    x_test, y_test = load_mnist(args.data, train=False)
    f_test, p_test = features(evaluator, x_test, device)
    print(f"classifier accuracy on real test digits: {(p_test == y_test.numpy()).mean():.4f}")
    # Floor for this metric: FID of N real training digits against the test set.
    x_train, _ = load_mnist(args.data, train=True)
    g = torch.Generator().manual_seed(args.seed)
    real = x_train[torch.randperm(len(x_train), generator=g)[: args.n]]
    floor = fid(features(evaluator, real, device)[0], f_test)
    print(f"FID of {args.n} real training digits vs. test digits (the best a model can expect): {floor:.3f}")

    labels = torch.arange(args.n) % 10                         # balanced requests: image i asks for digit i mod 10
    rows, results = {}, []
    for w in args.w:
        start = time.time()
        images, nfe = mg.sample(net, args.n, labels, device, steps=args.steps, w=w, seed=args.seed)
        f_gen, pred = features(evaluator, images, device)
        acc = float((pred == labels.numpy()).mean())
        per_class = [fid(f_gen[labels.numpy() == c], f_test[y_test.numpy() == c]) for c in range(10)]
        row = {"w": w, "n": args.n, "steps": args.steps, "nfe_per_image": nfe,
               "label_accuracy": round(acc, 4), "fid": round(fid(f_gen, f_test), 3),
               "fid_class": round(float(np.mean(per_class)), 3),
               "pixel_spread": mg.diagnostics(evaluator, images, labels, device)["pixel_spread"]}
        results.append(row)
        rows[f"w={w:g}"] = images
        print(row, f"{time.time() - start:.0f}s")

    save_grid(rows, out / "cfg_grid.png")
    with open(out / "cfg_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0]))
        wr.writeheader()
        wr.writerows(results)
    with open(out / "cfg_metrics.json", "w") as f:
        json.dump({"checkpoint": str(args.checkpoint), "seed": args.seed, "fid_real_floor": round(floor, 3),
                   "results": results}, f, indent=2)

    ws = [r["w"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, key, name in zip(axes, ["label_accuracy", "fid", "fid_class"],
                             ["label accuracy", "FID (classifier features)", "FID_class (mean of 10)"]):
        shown = [r for r in results if key != "fid_class" or r["w"] > 0]   # w = 0 requests no digit
        ax.plot([r["w"] for r in shown], [r[key] for r in shown], "o-")
        ax.set_xlabel("guidance weight w")
        ax.set_title(name + (" (w > 0)" if key == "fid_class" else ""))
        ax.set_xticks(ws)
    axes[1].axhline(floor, ls="--", c="gray", lw=1, label="real digits")
    axes[1].legend()
    fig.suptitle(f"MNIST CFG, {args.n} samples per w, {args.steps} steps")
    fig.tight_layout()
    fig.savefig(out / "cfg_metrics.png", dpi=150)
    print(f"saved cfg_grid.png, cfg_metrics.csv/.json/.png in {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    common.add_argument("--data", default="downloads/mnist", help="where torchvision stores MNIST")
    common.add_argument("--out", default="outputs/mnist_cfg")
    common.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--ch", type=int, default=64, help="U-Net width (the course's generator.pt records its own)")
    t.add_argument("--label-dropout", type=float, default=0.1)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--save-every", type=int, default=5, help="epochs between checkpoint saves")
    t.add_argument("--limit", type=int, default=0, help="use only the first N digits (smoke tests)")

    e = sub.add_parser("eval", parents=[common])
    e.add_argument("--checkpoint", default="outputs/mnist_cfg/mnist_cfg.pt")
    e.add_argument("--weights", default=".", help="folder holding evaluator.pt (downloaded if missing)")
    e.add_argument("--w", type=float, nargs="+", default=[0, 1, 2, 4])
    e.add_argument("--n", type=int, default=5000, help="samples per w, balanced over the 10 digits")
    e.add_argument("--steps", type=int, default=16)

    args = p.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
