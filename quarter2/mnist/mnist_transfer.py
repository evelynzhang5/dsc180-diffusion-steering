"""MNIST direction-transfer matrix: does a thickness direction learned at one noise level control thickness when
applied at another, and is the late-stage weakness shared by every direction?

    python quarter2/mnist/mnist_transfer.py --out outputs/q2_mnist_transfer [--n 1000]

For each layer (u2, d3) a within-digit thick-minus-thin direction is learned at t_learn in {0.25, 0.5, 0.75}
(from half A of the real digits, as in mnist_release.py). Each direction, both signs, is applied at every
t_apply in the same set, on for t in [t_apply, t_apply + 0.125) at strength 0.5 x |h| (16 DDIM steps, w = 1; the
release test showed 16 and 64 steps agree at matched duration), then switched off. Controls: 8 random unit
directions per (layer, t_apply).

Per image (saved in per_image.csv) and per run:
- on: thickness change of the edited clean-image guess at t_apply (the denoiser's immediate response);
- release: thickness change of the unedited guess on the edited state at t_apply + 0.125;
- end: thickness change of the final digit; all vs. the same image without the edit, in units of the within-digit
  SD of real digits;
- out: RMS change of the edited guess (pixels in [-1, 1]), so on / out separates "how much the output moves"
  from "how much of that movement is thickness";
- clip: change in the share of guess pixels at the +-1 clamp;
- correct: the evaluator still reads the requested digit.
Two seeds of starting noise: 0 (dev, used so far) and 1 (test, untouched until now); the report leads with the
test seed. 95% intervals bootstrap over images (paired: the same images with and without the edit).
Also reported: held-out probe R^2 per t (penalty and centering from half A only) and the cosine similarity of the
directions learned at different t.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_release import follow, learn_direction, GRID  # noqa: E402
from mnist_thickness import ROOT, Edit, activations, identity, measures, mg, probe, within  # noqa: E402

LAYERS = ["u2", "d3"]
TIMES = [0.25, 0.5, 0.75]
WINDOW = 0.125
STRENGTH = 0.5
N_RANDOM = 8
STEPS = 16
SEEDS = {"test": 1, "dev": 0}


def boot(v, n=1000, seed=0):
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-real", type=int, default=3000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    out, dev = Path(a.out), a.device
    out.mkdir(parents=True, exist_ok=True)
    net, evaluator = mg.load_models(ROOT, dev)
    edits = {l: Edit(getattr(net, l)) for l in LAYERS}

    from torchvision.datasets import MNIST
    ds = MNIST(ROOT / "downloads" / "mnist", train=True, download=False)
    perm = torch.randperm(len(ds.data), generator=torch.Generator().manual_seed(0))
    xr = ds.data.float().div(127.5).sub(1).unsqueeze(1)[perm][:2 * a.n_real]
    yr = ds.targets[perm][:2 * a.n_real]
    th = measures(xr)["thickness"]
    keep = ~np.isnan(th)
    xr, yr, th = xr[keep], yr[keep], th[keep]
    half = a.n_real  # half A = the first n_real draws, exactly as in mnist_release.py
    nA = int(keep[:half].sum())
    A, B = slice(0, nA), slice(nA, None)
    sd = float(within(th[A], yr[A].numpy()).std())

    # directions, probe R^2 and cross-time cosine
    dirs, shapes, probes, cos = {}, {}, {}, {}
    for layer in LAYERS:
        for t in TIMES:
            d, shape = learn_direction(net, edits, layer, t, xr[A], yr[A], th[A], dev)
            dirs[(layer, t)], shapes[layer] = d, shape
            hA = activations(net, edits, layer, xr[A], yr[A], t, dev, seed=1)
            hB = activations(net, edits, layer, xr[B], yr[B], t, dev, seed=2)
            probes[(layer, t)] = probe(hA, th[A], yr[A].numpy(), hB, th[B], yr[B].numpy(), d)
            print(f"{layer} t={t}: probe R^2 {probes[(layer, t)][0]:.2f} (penalty {probes[(layer, t)][2]:g}), "
                  f"corr {probes[(layer, t)][1]:.2f}", flush=True)
        for t1 in TIMES:
            for t2 in TIMES:
                cos[(layer, t1, t2)] = float(dirs[(layer, t1)] @ dirs[(layer, t2)])

    y_gen = torch.arange(10).repeat_interleave(a.n // 10)
    rows = []
    for split, seed in SEEDS.items():
        x_init = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(seed))
        base = follow(net, edits, x_init, y_gen, dev, STEPS)
        bth = {k: measures(base[k][1])["thickness"] for k in range(GRID + 1)}
        ok0, _ = identity(evaluator, base[GRID][0], y_gen, dev)
        clip0 = {k: (base[k][1].abs() >= 0.999).float().mean((1, 2, 3)).numpy() for k in range(GRID + 1)}
        for layer in LAYERS:
            for t_apply in TIMES:
                k_on, k_rel = int(t_apply * GRID), int((t_apply + WINDOW) * GRID)
                g = torch.Generator().manual_seed(int(1000 * t_apply) + (0 if layer == "u2" else 7))
                runs = [(f"{s}learn{t_learn}", t_learn, sg * dirs[(layer, t_learn)])
                        for t_learn in TIMES for s, sg in (("+", 1), ("-", -1))]
                for j in range(N_RANDOM):
                    r = torch.randn(dirs[(layer, TIMES[0])].shape, generator=g)
                    runs.append((f"rand{j}", None, r / r.norm()))
                for name, t_learn, vec in runs:
                    tr = follow(net, edits, x_init, y_gen, dev, STEPS, layer, vec.view(shapes[layer]).to(dev),
                                STRENGTH, t_apply)
                    on = (measures(tr[k_on][1])["thickness"] - bth[k_on]) / sd
                    rel = (measures(tr[k_rel][1])["thickness"] - bth[k_rel]) / sd
                    end = (measures(tr[GRID][0])["thickness"] - bth[GRID]) / sd
                    outc = (tr[k_on][1] - base[k_on][1]).pow(2).mean((1, 2, 3)).sqrt().numpy()
                    clip = (tr[k_on][1].abs() >= 0.999).float().mean((1, 2, 3)).numpy() - clip0[k_on]
                    ok, _ = identity(evaluator, tr[GRID][0], y_gen, dev)
                    for i in range(len(y_gen)):
                        rows.append(dict(split=split, layer=layer, t_apply=t_apply, direction=name,
                                         t_learn=t_learn if t_learn is not None else "",
                                         image=i, digit=int(y_gen[i]), on=round(float(on[i]), 4),
                                         release=round(float(rel[i]), 4), end=round(float(end[i]), 4),
                                         out=round(float(outc[i]), 5), clip=round(float(clip[i]), 5),
                                         correct=int(ok[i]), correct0=int(ok0[i])))
                print(f"{split} {layer} apply {t_apply}: {len(runs)} runs", flush=True)

    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report(rows, probes, cos, out, sd, a.n)
    figure(rows, out)


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def arr(rs, key):
    return np.array([float(r[key]) for r in rs])


def report(rows, probes, cos, out, sd, n):
    L = ["# MNIST direction-transfer matrix\n",
         f"{n} generated digits per run, 16 DDIM steps, w = 1, edit on for t in [t_apply, t_apply + {WINDOW}) at "
         f"{STRENGTH} x |h|, then off. Thickness changes in units of the within-digit SD of real digits "
         f"({sd:.3f} px). Mean (95% interval over images). Test seed unless marked.\n",
         "## Readability and direction similarity\n",
         "| layer | t | held-out probe R^2 | corr of projection on d | cosine with d learned at 0.25 / 0.5 / 0.75 |",
         "| --- | --- | --- | --- | --- |"]
    for layer in LAYERS:
        for t in TIMES:
            r2, c, _ = probes[(layer, t)]
            L.append(f"| {layer} | {t} | {r2:.2f} | {c:.2f} | "
                     + " / ".join(f"{cos[(layer, t, t2)]:.2f}" for t2 in TIMES) + " |")
    L.append("")
    for layer in LAYERS:
        L += [f"## {layer}: final thickness change, +d (rows: learned at; columns: applied at)\n",
              "| learned at | " + " | ".join(f"applied {t}" for t in TIMES) + " |",
              "| --- |" + " --- |" * len(TIMES)]
        for tl in TIMES:
            cells = []
            for ta in TIMES:
                m, lo, hi = boot(arr(sel(rows, split="test", layer=layer, t_apply=ta, direction=f"+learn{tl}"), "end"))
                cells.append(f"{m:+.2f} ({lo:+.2f}, {hi:+.2f})")
            L.append(f"| {tl} | " + " | ".join(cells) + " |")
        cells_s, cells_a = [], []
        for ta in TIMES:
            rs = [r for r in sel(rows, split="test", layer=layer, t_apply=ta) if r["direction"].startswith("rand")]
            per_dir = [arr([r for r in rs if r["direction"] == f"rand{j}"], "end") for j in range(N_RANDOM)]
            cells_s.append(f"{np.mean([np.nanmean(v) for v in per_dir]):+.3f} "
                           f"(range {min(np.nanmean(v) for v in per_dir):+.3f} to {max(np.nanmean(v) for v in per_dir):+.3f})")
            cells_a.append(f"{np.mean([np.nanmean(np.abs(v)) for v in per_dir]):.3f}")
        L += ["| random, signed mean over 8 | " + " | ".join(cells_s) + " |",
              "| random, mean absolute per image | " + " | ".join(cells_a) + " |", ""]
        L += [f"### {layer}: where the change comes from (+d, test seed)\n",
              "| learned at | applied at | guess change while on | RMS output change | thickness per unit output | "
              "at release | at end | per-image slope end~release (R^2) | +d / -d end | same sign as requested | "
              "abs. change +d | accuracy drop | clip change |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for tl in TIMES:
            for ta in TIMES:
                rs = sel(rows, split="test", layer=layer, t_apply=ta, direction=f"+learn{tl}")
                rm = sel(rows, split="test", layer=layer, t_apply=ta, direction=f"-learn{tl}")
                on, outc, rel, end = arr(rs, "on"), arr(rs, "out"), arr(rs, "release"), arr(rs, "end")
                m = ~np.isnan(rel) & ~np.isnan(end)
                slope = float(np.polyfit(rel[m], end[m], 1)[0])
                r2 = float(np.corrcoef(rel[m], end[m])[0, 1] ** 2)
                acc = float(np.mean(arr(rs, "correct0")) - np.mean(arr(rs, "correct")))
                L.append(f"| {tl} | {ta} | {np.nanmean(on):+.2f} | {outc.mean():.3f} | "
                         f"{np.nanmean(on) / outc.mean():.1f} | {np.nanmean(rel):+.2f} | {np.nanmean(end):+.2f} | "
                         f"{slope:.2f} ({r2:.2f}) | {np.nanmean(end):+.2f} / {np.nanmean(arr(rm, 'end')):+.2f} | "
                         f"{np.nanmean(end > 0):.0%} | {np.nanmean(np.abs(end)):.2f} | {acc:+.1%} | "
                         f"{arr(rs, 'clip').mean():+.4f} |")
        L.append("")
        L += [f"### {layer}: per-image distribution and by digit (+d, learned = applied, test seed)\n",
              "| t | median | 10th-90th percentile | " + " | ".join(f"digit {c}" for c in range(10)) + " |",
              "| --- | --- | --- |" + " --- |" * 10]
        for t in TIMES:
            rs = sel(rows, split="test", layer=layer, t_apply=t, direction=f"+learn{t}")
            e = arr(rs, "end")
            dig = [np.nanmean(arr([r for r in rs if r["digit"] == c], "end")) for c in range(10)]
            L.append(f"| {t} | {np.nanmedian(e):+.2f} | {np.nanquantile(e, 0.1):+.2f} to {np.nanquantile(e, 0.9):+.2f} | "
                     + " | ".join(f"{v:+.2f}" for v in dig) + " |")
        L.append("")
    L += ["## Dev vs. test seed (final change, +d, learned = applied)\n", "| layer | t | dev | test |",
          "| --- | --- | --- | --- |"]
    for layer in LAYERS:
        for t in TIMES:
            v = {s: boot(arr(sel(rows, split=s, layer=layer, t_apply=t, direction=f"+learn{t}"), "end")) for s in SEEDS}
            L.append(f"| {layer} | {t} | {v['dev'][0]:+.2f} ({v['dev'][1]:+.2f}, {v['dev'][2]:+.2f}) | "
                     f"{v['test'][0]:+.2f} ({v['test'][1]:+.2f}, {v['test'][2]:+.2f}) |")
    L.append("")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


def figure(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 3.8), squeeze=False)
    for j, layer in enumerate(LAYERS):
        a = ax[0, j]
        for i, tl in enumerate(TIMES):
            ys = [boot(arr(sel(rows, split="test", layer=layer, t_apply=ta, direction=f"+learn{tl}"), "end"))
                  for ta in TIMES]
            a.errorbar(np.array(TIMES) + 0.01 * (i - 1), [y[0] for y in ys],
                       yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]], marker="o", capsize=3,
                       label=f"learned at t = {tl}")
        rand = [np.mean([np.nanmean(np.abs(arr(sel(rows, split="test", layer=layer, t_apply=ta, direction=f"rand{k}"),
                                                "end"))) for k in range(N_RANDOM)]) for ta in TIMES]
        a.plot(TIMES, rand, "k:", label="random, mean |change|")
        a.axhline(0, color="k", lw=0.5)
        a.set_xlabel("applied at t (0 = noise, 1 = data)")
        a.set_ylabel("final thickness change (SD)")
        a.set_title(f"{layer}: transfer across noise levels (test seed)", fontsize=9)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "transfer.png", dpi=110)


if __name__ == "__main__":
    main()
