"""MNIST release-and-follow test: does a thickness edit survive after it is switched off, and is it the same at
16 and 64 sampling steps when the edit covers the same stretch of time?

    python quarter2/mnist/mnist_release.py --out outputs/q2_mnist_release [--n 1000]

Each edit is on for a fixed window of the time variable, t in [t0, t0 + 0.125) (2 steps of 16, or 8 steps of 64),
with the same per-step amplitude (strength x |h| along the direction), then switched off; sampling continues
normally to the end. Directions: the within-digit thick-minus-thin direction learned at t0 (as in
mnist_thickness.py), its negative, and random unit directions. Strengths: 0.25 and 0.5, both signs.

Both trajectories (edited and unedited, same starting noise) are followed at the 17 shared time points
t = 0, 1/16, ..., 1. At each point:
- state gap = |x_edit - x_base| / alpha_t (RMS per pixel), the state difference in data units; it measures how
  much of the edit is in the trajectory without asking the denoiser;
- thickness gap = mean thickness of the denoiser's clean-image guess, edited minus unedited (in units of the
  within-digit SD of real digits); after release the guess is an unedited network call on the edited state.
Retention after release = gap at the end / gap at release (the first time point at or after t0 + 0.125). Below 1
means the remaining (unedited) sampling shrinks the difference; about 1 means it carries it unchanged.
Writes OUT/release.csv (one row per run and time point), OUT/summary.csv, OUT/report.md and OUT/curves.png.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_thickness import ROOT, Edit, activations, identity, measures, mg, within  # noqa: E402

LAYERS = ["d3", "u2"]
STARTS = [0.25, 0.5, 0.75]
WINDOW = 0.125
STRENGTHS = [0.25, 0.5]
N_RANDOM = 2
GRID = 16  # shared time points t = k / 16


def learn_direction(net, edits, layer, t, xr, yr, th, dev):
    """Within-digit thick-minus-thin mean activation difference at time t, averaged over digits, unit norm."""
    h = activations(net, edits, layer, xr, yr, t, dev, seed=1)
    y = yr.numpy()
    diffs = []
    for c in range(10):
        tc = th[y == c]
        lo, hi = np.quantile(tc, [1 / 3, 2 / 3])
        hc = h[torch.from_numpy(y == c)]
        diffs.append(hc[torch.from_numpy(tc >= hi)].mean(0) - hc[torch.from_numpy(tc <= lo)].mean(0))
    d = torch.stack(diffs).mean(0)
    return d / d.norm(), edits[layer].last.shape[1:]


@torch.no_grad()
def follow(net, edits, x_init, y, dev, steps, layer=None, d=None, a=0.0, t0=None):
    """DDIM w = 1; edit on while t0 <= t < t0 + WINDOW. Returns {k: (state x_t, guess x1_hat)} at t = k / GRID."""
    x, y = x_init.to(dev), y.to(dev)
    ts = torch.linspace(0, 1, steps + 1)
    every = steps // GRID
    out = {}
    for i in range(steps):
        t, s = ts[i], ts[i + 1]
        on = layer is not None and t0 - 1e-9 <= float(t) < t0 + WINDOW - 1e-9
        if layer:
            edits[layer].active, edits[layer].d, edits[layer].a = on, d, a
        x1 = net(x, t.to(dev).expand(len(x)).contiguous(), y).clamp(-1, 1)
        if layer:
            edits[layer].active = False
        if i % every == 0:
            out[i // every] = (x.cpu(), x1.cpu())
        if i == steps - 1:
            x = x1
            break
        at, st = mg.alpha_sigma(t.to(dev))
        a_s, s_s = mg.alpha_sigma(s.to(dev))
        x = a_s * x1 + s_s * (x - at * x1) / st
    out[GRID] = (x.clamp(-1, 1).cpu(), x.clamp(-1, 1).cpu())
    return out


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
    xr = ds.data.float().div(127.5).sub(1).unsqueeze(1)[perm][:a.n_real]  # half A of mnist_thickness.py
    yr = ds.targets[perm][:a.n_real]
    th = measures(xr)["thickness"]
    keep = ~np.isnan(th)
    xr, yr, th = xr[keep], yr[keep], th[keep]
    sd = float(within(th, yr.numpy()).std())

    y_gen = torch.arange(10).repeat_interleave(a.n // 10)
    x_init = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(0))
    base = {st: follow(net, edits, x_init, y_gen, dev, st) for st in (16, 64)}
    base_th = {st: {k: measures(g)["thickness"] for k, (_, g) in b.items()} for st, b in base.items()}
    ok0 = {st: identity(evaluator, base[st][GRID][0], y_gen, dev)[0].mean() for st in base}
    print(f"unedited accuracy: 16 steps {ok0[16]:.1%}, 64 steps {ok0[64]:.1%}", flush=True)

    rows, summ = [], []
    for layer in LAYERS:
        for t0 in STARTS:
            d, shape = learn_direction(net, edits, layer, t0, xr, yr, th, dev)
            g = torch.Generator().manual_seed(int(1000 * t0))
            dirs = {"+d": d, "-d": -d}
            for j in range(N_RANDOM):
                r = torch.randn(d.shape, generator=g)
                dirs[f"rand{j}"] = r / r.norm()
            rel = int(round((t0 + WINDOW) * GRID))
            for name, vec in dirs.items():
                for s in STRENGTHS:
                    for st in (16, 64):
                        tr = follow(net, edits, x_init, y_gen, dev, st, layer, vec.view(shape).to(dev), s, t0)
                        state, thick = {}, {}
                        for k in range(GRID + 1):
                            alpha = float(mg.alpha_sigma(torch.tensor(k / GRID))[0])
                            state[k] = float((tr[k][0] - base[st][k][0]).pow(2).mean().sqrt()) / max(alpha, 1e-6) \
                                if k > 0 else 0.0
                            thick[k] = float(np.nanmean(measures(tr[k][1])["thickness"] - base_th[st][k])) / sd
                            rows.append(dict(layer=layer, t0=t0, direction=name, strength=s, steps=st, t=k / GRID,
                                             state_gap=round(state[k], 5), thick_gap=round(thick[k], 4)))
                        ok = identity(evaluator, tr[GRID][0], y_gen, dev)[0].mean()
                        summ.append(dict(layer=layer, t0=t0, direction=name, strength=s, steps=st,
                                         state_release=round(state[rel], 5), state_end=round(state[GRID], 5),
                                         state_kept=round(state[GRID] / state[rel], 3) if state[rel] else "",
                                         thick_release=round(thick[rel], 3), thick_end=round(thick[GRID], 3),
                                         thick_kept=round(thick[GRID] / thick[rel], 3) if abs(thick[rel]) > 0.03 else "",
                                         thick_per_state=round(thick[rel] / state[rel], 2) if state[rel] else "",
                                         acc_drop=round(float(ok0[st] - ok), 4)))
            print(f"{layer} t0={t0}: done", flush=True)

    for name, rs in (("release.csv", rows), ("summary.csv", summ)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    report(summ, out, a.n, sd)
    curves(rows, out)


def report(summ, out, n, sd):
    L = ["# MNIST release-and-follow test\n",
         f"{n} digits, dev seed 0. Edit on for t in [t0, t0 + {WINDOW}) at the same per-step amplitude, then off. "
         f"Thickness in units of the within-digit SD of real digits ({sd:.3f} px); state gap = RMS of the state "
         "difference in data units. 'kept' = value at the end / value at release. Random = mean of "
         f"{N_RANDOM} random directions.\n"]
    for s in STRENGTHS:
        L += [f"## Strength {s}\n",
              "| layer | t0 | steps | thickness at release: +d / -d / random | thickness at end: +d / -d / random | "
              "thickness kept +d / -d | state kept +d / random | thickness per unit state +d / random | "
              "accuracy drop +d / -d |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for layer in LAYERS:
            for t0 in STARTS:
                for st in (16, 64):
                    sel = {r["direction"]: r for r in summ if r["layer"] == layer and r["t0"] == t0
                           and r["strength"] == s and r["steps"] == st}
                    rand = [r for k, r in sel.items() if k.startswith("rand")]
                    rm = lambda key: np.mean([float(r[key]) for r in rand if r[key] != ""]) if rand else float("nan")
                    p, m = sel["+d"], sel["-d"]
                    f = lambda v: f"{v:.2f}" if v != "" else "–"
                    L.append(f"| {layer} | {t0} | {st} | {p['thick_release']:+.2f} / {m['thick_release']:+.2f} / "
                             f"{rm('thick_release'):+.2f} | {p['thick_end']:+.2f} / {m['thick_end']:+.2f} / "
                             f"{rm('thick_end'):+.2f} | {f(p['thick_kept'])} / {f(m['thick_kept'])} | "
                             f"{f(p['state_kept'])} / {rm('state_kept'):.2f} | {f(p['thick_per_state'])} / "
                             f"{rm('thick_per_state'):.2f} | {p['acc_drop']:+.1%} / {m['acc_drop']:+.1%} |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


def curves(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = 0.5
    fig, ax = plt.subplots(2, len(STARTS), figsize=(4.2 * len(STARTS), 6.4), squeeze=False)
    for j, t0 in enumerate(STARTS):
        for i, key in enumerate(["thick_gap", "state_gap"]):
            a = ax[i, j]
            for name, col in (("+d", "C3"), ("-d", "C0"), ("rand0", "gray")):
                for st, ls in ((16, "-"), (64, "--")):
                    rs = [r for r in rows if r["layer"] == "u2" and r["t0"] == t0 and r["direction"] == name
                          and r["strength"] == s and r["steps"] == st]
                    a.plot([r["t"] for r in rs], [r[key] for r in rs], ls, color=col, label=f"{name}, {st} steps")
            a.axvspan(t0, t0 + WINDOW, color="orange", alpha=0.2)
            a.axhline(0, color="k", lw=0.5)
            a.set_xlabel("t (0 = noise, 1 = data)")
            a.set_title(f"u2, edit on t in [{t0}, {t0 + WINDOW}): "
                        + ("thickness gap of the guess (SD)" if key == "thick_gap" else "state gap (data units)"),
                        fontsize=8)
    ax[0, 0].legend(fontsize=6)
    fig.suptitle(f"Release-and-follow, strength {s}: edited minus unedited trajectory (shaded = edit on)")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)


if __name__ == "__main__":
    main()
