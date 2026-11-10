"""MNIST mechanism and calibration test for thickness steering.

    python quarter2/mnist/mnist_calibrate.py --out outputs/q2_mnist_calibrate [--n 1000] [--target 0.2]

Part 1, fixed-size edits (dev seed 0). One direction per layer (learned at t = 0.5) is applied at
t_apply in {0.25, 0.5, 0.75} for t in [t_apply, t_apply + 0.125) (2 of 16 DDIM steps), either as the usual
relative edit a x |h_t| x d or as a fixed absolute edit eps x d, with eps the same at every t (multiples of the
average |h| over the three times), both signs. At the first edited call, against the unedited trajectory:
- raw: RMS change of the denoiser's unclipped clean-image guess (the network's own response);
- clipped: the same after the [-1, 1] clamp the sampler uses (clipped / raw shows how much the clamp absorbs);
- gain = raw / eps: output response per unit activation change along d;
- on: thickness change of that guess; release: RMS state change when the edit switches off, in data units;
- end: thickness change of the final digit (units: within-digit SD of real digits).
If fixed eps removes the late decline, changing |h_t| explains it; if raw gain still falls, the network's
sensitivity along d falls; if gain holds but end falls, the sampler / conversion to thickness accounts for it.

Part 2, calibration (u2). For each window ([t, t + 0.125) at t = 0.25, 0.5, 0.75, and every step, t in [0, 1)) and
each sign, the relative strength a is tuned on dev seed 0 so the mean final thickness change hits +-target. The
strengths are frozen and evaluated on fresh seed 2 (seeds 1 = earlier test, now used for design; 3 = untouched).
Reported: achieved change and its error, per-image spread, correct->incorrect and incorrect->correct identity
transitions, evaluator confidence drop, slant change, centroid shift, and the activation perturbation used.
Writes OUT/fixed.csv, OUT/calibration.csv (per image), OUT/report.md and OUT/fixed.png.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_release import learn_direction  # noqa: E402
from mnist_thickness import ROOT, Edit, identity, measures, mg, within  # noqa: E402

LAYERS = ["u2", "d3"]
TIMES = [0.25, 0.5, 0.75]
WINDOW = 0.125
STEPS = 16
T_LEARN = 0.5
EPS_MULT = [0.125, 0.25, 0.5]  # fixed eps = multiple x average |h| over TIMES
REL = 0.5  # the relative strength used in the transfer matrix
WINDOWS = {"0.25": (0.25, 0.375), "0.5": (0.5, 0.625), "0.75": (0.75, 0.875), "all": (0.0, 1.0)}
SEED_DEV, SEED_FRESH = 0, 2


@torch.no_grad()
def sample(net, edits, x_init, y, dev, layer=None, d=None, a=0.0, on=(None, None), absolute=False):
    """DDIM (16 steps, w = 1). Edit on while on[0] <= t < on[1]. Returns final images and per step k:
    state x, raw guess, |h| of the layer (as seen by the next block, i.e. after any edit)."""
    x, y = x_init.to(dev), y.to(dev)
    ts = torch.linspace(0, 1, STEPS + 1)
    states, raws, norms = {}, {}, {}
    for i in range(STEPS):
        t, s = ts[i], ts[i + 1]
        active = layer is not None and on[0] - 1e-9 <= float(t) < on[1] - 1e-9
        for l, e in edits.items():
            e.active = active and l == layer
            e.d, e.a, e.absolute = (d, a, absolute) if l == layer else (None, 0.0, False)
        raw = net(x, t.to(dev).expand(len(x)).contiguous(), y)
        for e in edits.values():
            e.active = False
        states[i], raws[i] = x.cpu(), raw.cpu()
        norms[i] = {l: e.last.flatten(1).norm(dim=1).cpu() for l, e in edits.items()}
        x1 = raw.clamp(-1, 1)
        if i == STEPS - 1:
            x = x1
            break
        at, st = mg.alpha_sigma(t.to(dev))
        a_s, s_s = mg.alpha_sigma(s.to(dev))
        x = a_s * x1 + s_s * (x - at * x1) / st
    return x.clamp(-1, 1).cpu(), states, raws, norms


def rms(v):
    return v.flatten(1).pow(2).mean(1).sqrt().numpy()


def boot(v, n=1000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def ci(v, fmt="+.2f"):
    m, lo, hi = boot(v)
    return f"{m:{fmt}} ({lo:{fmt}}, {hi:{fmt}})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-real", type=int, default=3000)
    ap.add_argument("--target", type=float, default=0.2)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    out, dev = Path(a.out), a.device
    out.mkdir(parents=True, exist_ok=True)
    net, evaluator = mg.load_models(ROOT, dev)
    edits = {l: Edit(getattr(net, l)) for l in LAYERS}

    from torchvision.datasets import MNIST
    ds = MNIST(ROOT / "downloads" / "mnist", train=True, download=False)
    perm = torch.randperm(len(ds.data), generator=torch.Generator().manual_seed(0))
    xr = ds.data.float().div(127.5).sub(1).unsqueeze(1)[perm][:a.n_real]  # half A, as before
    yr = ds.targets[perm][:a.n_real]
    m = measures(xr)
    keep = ~np.isnan(m["thickness"])
    xr, yr = xr[keep], yr[keep]
    sd = float(within(m["thickness"][keep], yr.numpy()).std())
    sd_slant = float(within(m["slant"][keep], yr.numpy()).std())

    dirs = {}
    for layer in LAYERS:
        d, shape = learn_direction(net, edits, layer, T_LEARN, xr, yr, m["thickness"][keep], dev)
        dirs[layer] = d.view(shape).to(dev)

    y_gen = torch.arange(10).repeat_interleave(a.n // 10)
    x_dev = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(SEED_DEV))
    x_fresh = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(SEED_FRESH))

    # ---------------------------------------------------------------- part 1: fixed-size edits
    b_fin, b_st, b_raw, b_norm = sample(net, edits, x_dev, y_gen, dev)
    b_end = measures(b_fin)["thickness"]
    hbar = {l: {t: float(b_norm[int(t * STEPS)][l].mean()) for t in TIMES} for l in LAYERS}
    fixed = []
    for layer in LAYERS:
        H = float(np.mean(list(hbar[layer].values())))
        for t in TIMES:
            k_on, k_off = int(t * STEPS), int((t + WINDOW) * STEPS)
            alpha_off = float(mg.alpha_sigma(torch.tensor(k_off / STEPS))[0])
            b_on = measures(b_raw[k_on].clamp(-1, 1))["thickness"]
            runs = [("relative", REL, False, REL * hbar[layer][t])]
            runs += [(f"fixed x{c}", c * H, True, c * H) for c in EPS_MULT]
            for name, amp, absolute, eps in runs:
                for sign in (1, -1):
                    fin, st, raw, _ = sample(net, edits, x_dev, y_gen, dev, layer, sign * dirs[layer], amp,
                                             (t, t + WINDOW), absolute)
                    r = rms(raw[k_on] - b_raw[k_on])
                    c = rms(raw[k_on].clamp(-1, 1) - b_raw[k_on].clamp(-1, 1))
                    on = (measures(raw[k_on].clamp(-1, 1))["thickness"] - b_on) / sd
                    rel = rms(st[k_off] - b_st[k_off]) / alpha_off
                    end = (measures(fin)["thickness"] - b_end) / sd
                    for i in range(len(y_gen)):
                        fixed.append(dict(layer=layer, t_apply=t, edit=name, sign=sign, eps=round(eps, 3),
                                          image=i, raw=round(float(r[i]), 5), clipped=round(float(c[i]), 5),
                                          on=round(float(on[i]), 4), release=round(float(rel[i]), 5),
                                          end=round(float(end[i]), 4)))
            print(f"part 1 {layer} t={t} done", flush=True)
    write(out / "fixed.csv", fixed)

    # ---------------------------------------------------------------- part 2: calibration (u2)
    layer = "u2"
    ok_dev0, _ = identity(evaluator, b_fin, y_gen, dev)
    f_fin, _, _, f_norm = sample(net, edits, x_fresh, y_gen, dev)
    f_m = measures(f_fin)
    ok0, p0 = identity(evaluator, f_fin, y_gen, dev)
    print(f"unedited accuracy: dev {ok_dev0.mean():.1%}, fresh {ok0.mean():.1%}", flush=True)

    def mean_change(amp, sign, win):
        return mean_change_vec(amp, sign * dirs[layer], win)

    calib, cal_log = [], []

    def mean_change_vec(amp, vec, win):
        fin = sample(net, edits, x_dev, y_gen, dev, layer, vec, amp, win)[0]
        return float(np.nanmean(measures(fin)["thickness"] - b_end)) / sd

    def evaluate(wname, sign, amp, vec, win):
        fin = sample(net, edits, x_fresh, y_gen, dev, layer, vec, amp, win)[0]
        mm = measures(fin)
        ok, p = identity(evaluator, fin, y_gen, dev)
        ks = [k for k in range(STEPS) if win[0] - 1e-9 <= k / STEPS < win[1] - 1e-9]
        # perturbation size: a x |h| at each edited call, using the unedited trajectory's |h| at that step
        per_call = np.stack([amp * f_norm[k][layer].numpy() for k in ks])
        for i in range(len(y_gen)):
            calib.append(dict(window=wname, sign=sign, strength=round(amp, 4), image=i, digit=int(y_gen[i]),
                              change=round(float((mm["thickness"][i] - f_m["thickness"][i]) / sd), 4),
                              slant=round(float((mm["slant"][i] - f_m["slant"][i]) / sd_slant), 4),
                              shift=round(float(np.hypot(mm["cx"][i] - f_m["cx"][i], mm["cy"][i] - f_m["cy"][i])), 4),
                              correct0=int(ok0[i]), correct=int(ok[i]), p_drop=round(float(p0[i] - p[i]), 4),
                              eps_call=round(float(per_call[:, i].mean()), 3),
                              eps_total=round(float(per_call[:, i].sum()), 3)))

    for wname, win in WINDOWS.items():
        for sign in (1, -1):
            goal = sign * a.target
            amp, lo, hi = (0.1 if wname == "all" else 0.5), 0.0, None
            hist = []
            for _ in range(8):  # secant on the observed slope, kept inside a bracket
                f = mean_change(amp, sign, win)
                hist.append((amp, f))
                if abs(f - goal) < 0.005:
                    break
                if abs(f) < abs(goal):
                    lo = amp
                else:
                    hi = amp
                new = amp * goal / f if f * goal > 0 else amp * 2
                if hi is not None:
                    new = min(max(new, lo + 0.1 * (hi - lo)), hi - 0.1 * (hi - lo))
                amp = min(new, 4.0)
            amp, f_dev = min(hist, key=lambda h: abs(h[1] - goal))
            cal_log.append(dict(window=wname, sign=sign, strength=amp, dev_change=f_dev, tries=len(hist)))
            print(f"calibrated {wname} sign {sign:+d}: a = {amp:.3f}, dev change {f_dev:+.3f} ({len(hist)} tries)",
                  flush=True)

            evaluate(wname, sign, amp, sign * dirs[layer], win)
    # control: a random unit direction at the 0.5 window and the same strength as the transfer matrix
    r = torch.randn(dirs[layer].shape, generator=torch.Generator().manual_seed(5))
    r = (r / r.norm()).to(dev)
    cal_log.append(dict(window="random, 0.5", sign=1, strength=REL, dev_change=mean_change_vec(REL, r, WINDOWS["0.5"]),
                        tries=1))
    evaluate("random, 0.5", 1, REL, r, WINDOWS["0.5"])
    write(out / "calibration.csv", calib)
    report(fixed, calib, cal_log, hbar, a, sd, sd_slant, ok0.mean(), out)
    figure(fixed, out)


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def col(rows, key, **kw):
    return np.array([float(r[key]) for r in rows if all(r[k] == v for k, v in kw.items())])


def report(fixed, calib, cal_log, hbar, a, sd, sd_slant, acc0, out):
    L = ["# MNIST mechanism and calibration test\n",
         f"{a.n} generated digits, 16 DDIM steps, w = 1. Thickness in within-digit SD of real digits ({sd:.3f} px); "
         f"slant in within-digit SD of real slant ({sd_slant:.3f}). Mean (95% interval over images).\n",
         f"## Part 1: fixed-size edits (dev seed {SEED_DEV}; direction learned at t = {T_LEARN})\n",
         "Average activation norm |h| of the unedited trajectory at each t:\n",
         "| layer | " + " | ".join(f"t = {t}" for t in TIMES) + " |", "| --- |" + " --- |" * len(TIMES)]
    for layer in LAYERS:
        L.append(f"| {layer} | " + " | ".join(f"{hbar[layer][t]:.1f}" for t in TIMES) + " |")
    L.append("")
    for layer in LAYERS:
        L += [f"### {layer}\n",
              "| edit | t | eps (activation change) | raw output change | clipped / raw | gain = raw / eps | "
              "guess thickness while on | state change at release | final change +eps | final change -eps | "
              "final per unit eps |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        names = sorted({r["edit"] for r in fixed if r["layer"] == layer}, key=lambda s: (s != "relative", s))
        for name in names:
            for t in TIMES:
                kw = dict(layer=layer, t_apply=t, edit=name)
                eps = col(fixed, "eps", sign=1, **kw)[0]
                raw, clp = col(fixed, "raw", sign=1, **kw), col(fixed, "clipped", sign=1, **kw)
                end_p, end_m = col(fixed, "end", sign=1, **kw), col(fixed, "end", sign=-1, **kw)
                L.append(f"| {name} | {t} | {eps:.2f} | {raw.mean():.4f} | {clp.mean() / raw.mean():.2f} | "
                         f"{raw.mean() / eps:.5f} | {np.nanmean(col(fixed, 'on', sign=1, **kw)):+.2f} | "
                         f"{col(fixed, 'release', sign=1, **kw).mean():.4f} | {ci(end_p)} | {ci(end_m)} | "
                         f"{np.nanmean(end_p) / eps:+.4f} |")
        L.append("")

    L += [f"## Part 2: calibrated to a target change of +-{a.target} SD (u2)\n",
          f"Strength tuned on dev seed {SEED_DEV}, frozen, evaluated on fresh seed {SEED_FRESH} (unedited accuracy "
          f"{acc0:.1%}). Windows of 2 steps, or every step ('all'). eps per call = a x |h| along the unedited "
          "trajectory; total = summed over edited calls.\n",
          "| window | sign | strength a (dev tries) | dev change | fresh change | error vs target | requested direction | "
          "10th-90th pct | correct->wrong | wrong->correct | confidence drop | abs. slant change | centroid shift (px) | "
          "eps per call | eps total |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cal_log:
        kw = dict(window=c["window"], sign=c["sign"])
        ch = col(calib, "change", **kw)
        c0, c1 = col(calib, "correct0", **kw), col(calib, "correct", **kw)
        goal = c["sign"] * a.target
        L.append(f"| {c['window']} | {c['sign']:+d} | {c['strength']:.3f} ({c['tries']}) | {c['dev_change']:+.3f} | "
                 f"{ci(ch)} | {'–' if c['window'].startswith('random') else f'{np.nanmean(ch) - goal:+.3f}'} | {np.nanmean(np.sign(ch) == c['sign']):.0%} | "
                 f"{np.nanquantile(ch, 0.1):+.2f} to {np.nanquantile(ch, 0.9):+.2f} | "
                 f"{int(((c0 == 1) & (c1 == 0)).sum())} | {int(((c0 == 0) & (c1 == 1)).sum())} | "
                 f"{ci(col(calib, 'p_drop', **kw), '+.3f')} | {ci(np.abs(col(calib, 'slant', **kw)), '.3f')} | "
                 f"{ci(col(calib, 'shift', **kw), '.3f')} | {col(calib, 'eps_call', **kw).mean():.1f} | "
                 f"{col(calib, 'eps_total', **kw).mean():.1f} |")
    L.append("")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


def figure(fixed, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for layer, ls in (("u2", "-"), ("d3", "--")):
        names = sorted({r["edit"] for r in fixed if r["layer"] == layer}, key=lambda s: (s != "relative", s))
        for j, name in enumerate(names):
            kw = dict(layer=layer, edit=name, sign=1)
            gain = [col(fixed, "raw", t_apply=t, **kw).mean() / col(fixed, "eps", t_apply=t, **kw)[0] for t in TIMES]
            end = [np.nanmean(col(fixed, "end", t_apply=t, **kw)) for t in TIMES]
            per = [e / col(fixed, "eps", t_apply=t, **kw)[0] for e, t in zip(end, TIMES)]
            for a_, v in zip(ax, (gain, end, per)):
                a_.plot(TIMES, v, ls, marker="o", color=f"C{j}", label=f"{layer} {name}")
    for a_, title in zip(ax, ("raw output change per unit eps (gain)", "final thickness change (SD)",
                              "final change per unit eps")):
        a_.set_title(title, fontsize=9)
        a_.set_xlabel("applied at t (0 = noise, 1 = data)")
        a_.axhline(0, color="k", lw=0.5)
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fixed.png", dpi=110)


if __name__ == "__main__":
    main()
