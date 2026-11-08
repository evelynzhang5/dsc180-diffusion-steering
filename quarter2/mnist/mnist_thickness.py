"""MNIST diagnostic: when stroke thickness is readable from the generator's activations, does pushing along the
thickness direction change the final digit's thickness, and does it keep the digit's identity?

    python quarter2/mnist/mnist_thickness.py --out outputs/q2_mnist_thickness [--steps 16] [--n 500]

Model: the frozen class-conditional MNIST U-Net (generator.pt) sampled with deterministic DDIM, w = 1 (the label
as trained; accuracy is already near 100%, so thickness is the target and identity the thing to protect).
Course convention: t = 0 is noise, t = 1 is data, x_t = sin(pi t/2) x1 + cos(pi t/2) noise.

1. Measures (on ink = (x + 1) / 2 > 0.5): thickness = ink area / skeleton length (pixels), slant = -mu11/mu02
   (shear of the second moments), position = ink centroid; identity = evaluator's probability of the requested
   digit and whether it is the top class.
2. Directions, per layer (d3 = last encoder block, the NA-RFM choice; u2 = a decoder block at 14x14) and noise
   level (steps 1/4, 1/2, 3/4 of the way): real MNIST training digits are split in two halves *before* noising.
   On half A, within each digit, mean activation of the thickest third minus the thinnest third; the ten digits'
   differences are averaged and normalized. Within-digit, so the direction is not "naturally thick digit (0)
   vs. thin digit (1)".
3. Readability on half B: R^2 of a ridge probe predicting within-digit thickness from activations, and the
   correlation of the projection on the direction with it. Centering, scale and the ridge penalty (5-fold CV) all
   come from half A, so half B is an untouched test set.
4. Intervention: same starting noise for every run. At one step, the block's output h gets + a * |h| * d
   (a = strength, |h| per image). Directions: +d, -d, and 4 random unit directions. Immediate effect = change in
   that step's clean-image guess; final effect = change in the finished digit; both vs. the unedited run, in
   units of the within-digit SD of thickness in real digits.
Each edit is on for one step, or for every remaining step ("rest").
Writes OUT/results.csv, OUT/report.md, OUT/figure_one.png, OUT/figure_rest.png and OUT/examples.png. Uses only dev seed 0; test seeds are
kept for later comparisons.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mnist_guidance_lab as mg  # noqa: E402

LAYERS = ["d3", "u2"]
STRENGTHS = [0.25, 0.5, 1.0]
LENGTHS = ["one", "rest"]  # edit on for one step, or for every remaining step
N_RANDOM = 4


# ------------------------------------------------------------------ measurements
def measures(x):
    """x: (n,1,28,28) in [-1,1] -> dict of numpy arrays: thickness, slant, cx, cy."""
    from skimage.morphology import skeletonize
    ink = ((x[:, 0].cpu().numpy() + 1) / 2) > 0.5
    th, sl, cx, cy = [], [], [], []
    yy, xx = np.mgrid[0:28, 0:28]
    for im in ink:
        area = im.sum()
        if area < 3:
            th.append(np.nan); sl.append(np.nan); cx.append(np.nan); cy.append(np.nan)
            continue
        th.append(area / max(skeletonize(im).sum(), 1))
        mx, my = (xx * im).sum() / area, (yy * im).sum() / area
        mu11 = ((xx - mx) * (yy - my) * im).sum() / area
        mu02 = ((yy - my) ** 2 * im).sum() / area
        sl.append(-mu11 / mu02)
        cx.append(mx); cy.append(my)
    return {k: np.array(v) for k, v in dict(thickness=th, slant=sl, cx=cx, cy=cy).items()}


@torch.no_grad()
def identity(evaluator, x, y, device):
    logits, _ = evaluator(x.to(device))
    p = logits.softmax(1)
    return (p.argmax(1) == y.to(device)).float().cpu().numpy(), p.gather(1, y.to(device)[:, None])[:, 0].cpu().numpy()


# ------------------------------------------------------------------ sampler with an optional edit
class Edit:
    """Forward hook on one block: records its output, and adds a * |h| * d when active."""
    def __init__(self, module):
        self.d, self.a, self.active, self.last = None, 0.0, False, None
        self.absolute = False  # True: add a * d (fixed size) instead of a * |h| * d
        module.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.active and self.d is not None:
            norm = 1.0 if self.absolute else out.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
            out = out + self.a * norm * self.d
        self.last = out
        return out


@torch.no_grad()
def run(net, edits, x_init, y, device, steps, edit_step=None, layer=None, d=None, a=0.0, length=1):
    """Deterministic DDIM, w = 1; the edit is on for `length` steps from edit_step. Returns (final images,
    clean-image guess at edit_step)."""
    x = x_init.to(device)
    y = y.to(device)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    guess = None
    for i, (t, s) in enumerate(zip(ts[:-1], ts[1:])):
        on = edit_step is not None and edit_step <= i < edit_step + length
        if layer:
            edits[layer].active, edits[layer].d, edits[layer].a = on, d, a
        x1 = net(x, t.expand(len(x)).contiguous(), y).clamp(-1, 1)
        if layer:
            edits[layer].active = False
        if i == edit_step:
            guess = x1.clone()
        if i == steps - 1:
            x = x1
            break
        at, st = mg.alpha_sigma(t)
        a_s, s_s = mg.alpha_sigma(s)
        x = a_s * x1 + s_s * (x - at * x1) / st
    return x.clamp(-1, 1).cpu(), (guess.cpu() if guess is not None else None)


# ------------------------------------------------------------------ directions and readability
@torch.no_grad()
def activations(net, edits, layer, x1, y, t, device, seed, bs=500):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(x1), bs):
        xb = x1[i:i + bs]
        noise = torch.randn(xb.shape, generator=g)
        a, s = mg.alpha_sigma(torch.tensor(t))
        net((a * xb + s * noise).to(device), torch.full((len(xb),), t, device=device), y[i:i + bs].to(device))
        out.append(edits[layer].last.flatten(1).cpu())
    return torch.cat(out)


def within(v, y):
    """Subtract each digit's mean (numpy 1-D or torch 2-D)."""
    v = v.clone() if torch.is_tensor(v) else v.copy()
    for c in range(10):
        m = y == c
        v[m] = v[m] - v[m].mean(0)
    return v


def ridge_r2(XA, tA, XB, tB, lam):
    """Kernel-form ridge (n << dim): fit on A, R^2 on B."""
    K = XA @ XA.T
    alpha = torch.linalg.solve(K + lam * torch.eye(len(K), dtype=K.dtype), tA)
    pred = XB @ (XA.T @ alpha)
    return float(1 - ((tB - pred) ** 2).sum() / ((tB - tB.mean()) ** 2).sum())


def probe(hA, thA, yA, hB, thB, yB, d=None, lams=(0.1, 1.0, 10.0, 100.0, 1000.0), folds=5):
    """Held-out readability of within-digit thickness. Everything is derived from half A: per-digit means used to
    center both halves, the feature scale, and the ridge penalty (5-fold CV inside A). Half B is touched only for
    the final R^2 and the correlation of its projection on d. Returns (R^2 on B, corr on B, chosen penalty)."""
    hA, hB = hA.double(), hB.double()
    thA, thB = torch.as_tensor(thA, dtype=torch.float64), torch.as_tensor(thB, dtype=torch.float64)
    yA, yB = torch.as_tensor(yA), torch.as_tensor(yB)
    XA, XB, tA, tB = hA.clone(), hB.clone(), thA.clone(), thB.clone()
    for c in range(10):
        ma, mb = yA == c, yB == c
        mu_h, mu_t = hA[ma].mean(0), thA[ma].mean()
        XA[ma] -= mu_h
        XB[mb] -= mu_h
        tA[ma] -= mu_t
        tB[mb] -= mu_t
    scale = float(XA.norm(dim=1).mean())
    XA, XB = XA / scale, XB / scale
    idx = torch.randperm(len(XA), generator=torch.Generator().manual_seed(0))
    parts = idx.chunk(folds)
    def cv(lam):
        out = []
        for f in range(folds):
            te = parts[f]
            tr = torch.cat([parts[g] for g in range(folds) if g != f])
            out.append(ridge_r2(XA[tr], tA[tr], XA[te], tA[te], lam))
        return float(np.mean(out))
    lam = max(lams, key=cv)
    r2 = ridge_r2(XA, tA, XB, tB, lam)
    corr = float(np.corrcoef((XB @ d.double()).numpy(), tB.numpy())[0, 1]) if d is not None else float("nan")
    return r2, corr, lam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--n", type=int, default=500, help="generated digits per run (50 per class)")
    ap.add_argument("--n-real", type=int, default=3000, help="real digits per half for directions and probes")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    net, evaluator = mg.load_models(ROOT, dev)
    edits = {l: Edit(getattr(net, l)) for l in LAYERS}
    edit_steps = [a.steps // 4, a.steps // 2, 3 * a.steps // 4]

    # real digits, split into halves before any noising
    from torchvision.datasets import MNIST
    ds = MNIST(ROOT / "downloads" / "mnist", train=True, download=False)
    perm = torch.randperm(len(ds.data), generator=torch.Generator().manual_seed(0))
    xr = ds.data.float().div(127.5).sub(1).unsqueeze(1)[perm][:2 * a.n_real]
    yr = ds.targets[perm][:2 * a.n_real]
    th = measures(xr)["thickness"]
    keep = ~np.isnan(th)
    xr, yr, th = xr[keep], yr[keep], th[keep]
    half = len(xr) // 2
    A, B = slice(0, half), slice(half, None)
    th_w = within(th, yr.numpy())
    sd_real = float(th_w.std())
    print(f"real digits: thickness {th.mean():.2f} px, within-digit SD {sd_real:.3f}", flush=True)

    # sanity check: the custom sampler reproduces the lab's sampler
    y_gen = torch.arange(10).repeat_interleave(a.n // 10)
    x_init = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(0))
    base, _ = run(net, edits, x_init, y_gen, dev, a.steps)
    ref, _ = mg._sample_chunk(net, x_init, y_gen, dev, a.steps, 1.0, 0.0, "ddim", 1)
    print(f"sampler check: max |diff| = {float((base - ref.cpu()).abs().max()):.2e}", flush=True)
    m0 = measures(base)
    ok0, p0 = identity(evaluator, base, y_gen, dev)
    print(f"unedited: accuracy {ok0.mean():.1%}, thickness {np.nanmean(m0['thickness']):.2f}", flush=True)

    rows = []
    examples = {}
    for layer in LAYERS:
        for k in edit_steps:
            t = float(k / a.steps)
            # direction from half A, readability on half B
            hA = activations(net, edits, layer, xr[A], yr[A], t, dev, seed=1)
            hB = activations(net, edits, layer, xr[B], yr[B], t, dev, seed=2)
            yA, yB = yr[A].numpy(), yr[B].numpy()
            diffs = []
            for c in range(10):
                tc = th[A][yA == c]
                lo, hi = np.quantile(tc, [1 / 3, 2 / 3])
                hc = hA[torch.from_numpy(yA == c)]
                diffs.append(hc[torch.from_numpy(tc >= hi)].mean(0) - hc[torch.from_numpy(tc <= lo)].mean(0))
            d = torch.stack(diffs).mean(0)
            d = d / d.norm()
            r2, corr, _ = probe(hA, th[A], yA, hB, th[B], yB, d)
            print(f"{layer} step {k} (t={t:.2f}): probe R^2 {r2:.2f}, direction corr {corr:.2f}", flush=True)

            shape = edits[layer].last.shape[1:]
            g = torch.Generator().manual_seed(100 + k)
            dirs = {"+d": d, "-d": -d}
            for j in range(N_RANDOM):
                r = torch.randn(d.shape, generator=g)
                dirs[f"rand{j}"] = r / r.norm()
            _, guess0 = run(net, edits, x_init, y_gen, dev, a.steps, edit_step=k)
            g0 = measures(guess0)
            for (name, vec), s, n_on in [(dv, s, n_on) for dv in dirs.items() for s in STRENGTHS for n_on in LENGTHS]:
                    vec = vec.view(shape).to(dev)
                    length = 1 if n_on == "one" else a.steps
                    fin, guess = run(net, edits, x_init, y_gen, dev, a.steps, edit_step=k, layer=layer, d=vec, a=s,
                                     length=length)
                    mf, mg_ = measures(fin), measures(guess)
                    ok, p = identity(evaluator, fin, y_gen, dev)
                    rows.append(dict(
                        layer=layer, step=k, t=round(t, 3), direction=name, strength=s, length=n_on, probe_r2=round(r2, 3),
                        dir_corr=round(corr, 3),
                        immediate=round(float(np.nanmean(mg_["thickness"] - g0["thickness"])) / sd_real, 3),
                        final=round(float(np.nanmean(mf["thickness"] - m0["thickness"])) / sd_real, 3),
                        acc_drop=round(float(ok0.mean() - ok.mean()), 4),
                        prob_drop=round(float(p0.mean() - p.mean()), 4),
                        slant_change=round(float(np.nanmean(np.abs(mf["slant"] - m0["slant"]))), 4),
                        shift_px=round(float(np.nanmean(np.hypot(mf["cx"] - m0["cx"], mf["cy"] - m0["cy"]))), 3),
                        pixel_change=round(float((fin - base).abs().mean()), 4)))
                    if name in ("+d", "-d", "rand0") and s == 0.5 and n_on == "one":
                        examples[(layer, k, name)] = fin[::a.n // 10][:10]
            print(f"   done {len(dirs) * len(STRENGTHS) * len(LENGTHS)} edited runs", flush=True)

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows, out, a, sd_real, ok0.mean(), float(np.nanmean(m0["thickness"])))
    figure(rows, out, edit_steps, a.steps)
    example_grid(examples, base[::a.n // 10][:10], out, edit_steps)


def summary(rows, layer, k, s, n_on="one"):
    sel = [r for r in rows if r["layer"] == layer and r["step"] == k and r["strength"] == s and r["length"] == n_on]
    pick = lambda n: next(r for r in sel if r["direction"] == n)
    rand = [r for r in sel if r["direction"].startswith("rand")]
    mean = lambda key: float(np.mean([r[key] for r in rand]))
    spread = lambda key: float(np.max([abs(r[key]) for r in rand]))
    return pick("+d"), pick("-d"), mean, spread


def report(rows, out, a, sd_real, acc0, th0):
    L = [f"# MNIST thickness steering diagnostic\n",
         f"{a.n} generated digits per run (dev seed 0), DDIM {a.steps} steps, w = 1. Unedited: accuracy {acc0:.1%}, "
         f"thickness {th0:.2f} px. Thickness changes are in units of the within-digit SD of real digits "
         f"({sd_real:.3f} px). Random = largest |change| over {N_RANDOM} random directions of the same size.\n"]
    for n_on, s in [(n, s) for n in LENGTHS for s in STRENGTHS]:
        L += [f"## Edit on for {'one step' if n_on == 'one' else 'every remaining step'}, strength {s} "
              f"(+ {s} x |h| along the direction)\n",
              "| layer | step (t) | probe R^2 | dir. corr | immediate +d / -d / random | final +d / -d / random | "
              "kept (final/immediate, +d) | accuracy drop +d / -d / random | slant change +d |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for layer in LAYERS:
            for k in sorted({r["step"] for r in rows}):
                p, m, mean, spread = summary(rows, layer, k, s, n_on)
                keep = p["final"] / p["immediate"] if abs(p["immediate"]) > 0.05 else float("nan")
                L.append(f"| {layer} | {k} ({p['t']:.2f}) | {p['probe_r2']:.2f} | {p['dir_corr']:.2f} | "
                         f"{p['immediate']:+.2f} / {m['immediate']:+.2f} / {spread('immediate'):.2f} | "
                         f"{p['final']:+.2f} / {m['final']:+.2f} / {spread('final'):.2f} | {keep:.2f} | "
                         f"{p['acc_drop']:+.1%} / {m['acc_drop']:+.1%} / {max(r['acc_drop'] for r in rows if r['layer'] == layer and r['step'] == k and r['strength'] == s and r['length'] == n_on and r['direction'].startswith('rand')):+.1%} | "
                         f"{p['slant_change']:.3f} |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


def figure(rows, out, edit_steps, steps):
    for n_on in LENGTHS:
        figure_one(rows, out, edit_steps, steps, n_on)


def figure_one(rows, out, edit_steps, steps, n_on):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = 0.5
    fig, ax = plt.subplots(len(LAYERS), 4, figsize=(14, 3.2 * len(LAYERS)), squeeze=False)
    ts = [k / steps for k in edit_steps]
    for i, layer in enumerate(LAYERS):
        S = [summary(rows, layer, k, s, n_on) for k in edit_steps]
        ax[i, 0].plot(ts, [p["probe_r2"] for p, *_ in S], "o-", label="ridge probe R^2")
        ax[i, 0].plot(ts, [p["dir_corr"] for p, *_ in S], "s--", label="projection on d (corr)")
        ax[i, 0].set_ylim(0, 1)
        for j, key in enumerate(["immediate", "final", "acc_drop"], 1):
            ax[i, j].plot(ts, [p[key] for p, *_ in S], "o-", color="C3", label="+d")
            ax[i, j].plot(ts, [m[key] for _, m, *_ in S], "o-", color="C0", label="-d")
            ax[i, j].fill_between(ts, [-sp(key) for *_, sp in S], [sp(key) for *_, sp in S], color="gray",
                                  alpha=0.3, label="random (max |.|)")
            ax[i, j].axhline(0, color="k", lw=0.5)
        titles = ["readable? (held-out)", "immediate thickness change (SD)", "final thickness change (SD)",
                  "identity damage (accuracy drop)"]
        for j in range(4):
            ax[i, j].set_title(f"{layer}: {titles[j]}", fontsize=9)
            ax[i, j].set_xlabel("edit time t (0 = noise, 1 = data)")
        ax[i, 0].legend(fontsize=7)
        ax[i, 1].legend(fontsize=7)
    fig.suptitle(f"Thickness direction vs. random, strength {s} x |h|, edit on for "
                 f"{'one step' if n_on == 'one' else 'every remaining step'}")
    fig.tight_layout()
    fig.savefig(out / f"figure_{n_on}.png", dpi=110)


def example_grid(examples, base, out, edit_steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("none",)] + list(examples)
    fig, ax = plt.subplots(len(keys), 10, figsize=(10, 1.05 * len(keys)))
    for r, key in enumerate(keys):
        ims = base if key == ("none",) else examples[key]
        for c in range(10):
            ax[r, c].imshow(ims[c, 0], cmap="gray", vmin=-1, vmax=1)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 0].set_ylabel("unedited" if key == ("none",) else f"{key[0]} s{key[1]} {key[2]}", rotation=0,
                            ha="right", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=90)


if __name__ == "__main__":
    main()
