"""MNIST side-effect audit: does evidence from interventions predict a steering direction's side effects on unseen
digit classes better than the reference examples it was learned from?

    python quarter2/mnist/mnist_audit.py --out outputs/q2_mnist_audit [--per-class 100] [--truth-per-class 200]

A toy version of "predict unintended steering effects on unseen prompts", with digit classes as prompts and a
fixed bank of measured attributes in place of a VLM.

Directions. A thickness direction (u2, t = 0.25, within-digit thick-minus-thin mean difference) is learned from a
reference set of real digits (60 thick + 60 thin per class). A hidden bias is planted on one nuisance attribute:
a share q of the thick digits is drawn from above that attribute's within-digit median and the same share of thin
digits from below (q = 1 strong, 0.75 moderate). Plus one natural set (no selection). The auditor is not told which
attribute was biased.

Attributes (units: within-digit SD over the real pool): slant, width, height, horizontal / vertical centroid (cx, cy),
stroke length (skeleton pixels) and darkness (mean ink intensity). Thickness itself is the target. Every effect is
expressed per 1 SD of achieved thickness change, so audits and truth are comparable at any strength.

Contexts. Classes are split into audit classes and held-out classes (0-4 / 5-9, and the reverse). The edit
(window [0.25, 0.375), strength tuned on dev seed 0 on the audit classes to +0.3 SD thickness) is audited on the audit
classes and judged on the held-out classes.

Audits (each flags an attribute when it predicts a side effect of at least tau per SD of thickness):
- REF: thick-minus-thin difference of the attribute in the reference set, divided by the thickness difference;
- OUT-1: one paired run (seed 10) on the audit classes, flag on the point estimate alone;
- OUT-rep: two paired runs (seeds 10 and 11) on the audit classes; flag only if both runs' 99.3% bootstrap intervals
  exclude 0 with the same sign (Bonferroni over 7 attributes) and the average clears tau;
- BOTH: REF and OUT-rep agree.
Budgets: 20 or `per-class` generated digits per audit class per run.
Truth: paired run (seed 12, `truth-per-class` per held-out class); an attribute is a true side effect when its
95% interval excludes 0 and |effect| >= tau. Seed 3 stays untouched.
Scores over (set, split, attribute): precision, recall, false alarms, and the rank correlation between predicted
and true effects. Writes OUT/items.csv, OUT/report.md. Resumable via OUT/items.jsonl.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_calibrate import sample  # noqa: E402
from mnist_thickness import ROOT, Edit, activations, mg, within  # noqa: E402

LAYER = "u2"
T_LEARN = 0.25
WIN = (0.25, 0.375)
POOL = 20000
ATTRS = ["slant", "width", "height", "cx", "cy", "length", "darkness"]
QS = [1.0, 0.75]
SPLITS = {"A": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), "B": ([5, 6, 7, 8, 9], [0, 1, 2, 3, 4])}
TAUS = [0.1, 0.15, 0.25]
SEED_DEV, SEED_AUD1, SEED_AUD2, SEED_TRUTH = 0, 10, 11, 12


def attrs(x):
    """x: (n, 1, 28, 28) in [-1, 1] -> dict of arrays: thickness and ATTRS (nan if almost no ink)."""
    from skimage.morphology import skeletonize
    v = (x[:, 0].cpu().numpy() + 1) / 2
    keys = ["thickness"] + ATTRS
    out = {k: [] for k in keys}
    for iv in v:
        im = iv > 0.5
        area = im.sum()
        if area < 3:
            for k in keys:
                out[k].append(np.nan)
            continue
        sk = skeletonize(im).sum()
        ys, xs = np.nonzero(im)
        mx, my = xs.mean(), ys.mean()
        mu11, mu02 = ((xs - mx) * (ys - my)).mean(), ((ys - my) ** 2).mean()
        vals = dict(thickness=area / max(sk, 1), slant=-mu11 / mu02 if mu02 > 0 else np.nan, width=xs.max() - xs.min() + 1,
                    height=ys.max() - ys.min() + 1, cx=mx, cy=my, length=sk, darkness=iv[im].mean())
        for k in keys:
            out[k].append(float(vals[k]))
    return {k: np.array(v_) for k, v_ in out.items()}


def choose(cell, k, rng, name):
    if len(cell) < k:
        print(f"warning: {name}: cell has {len(cell)} < {k} digits, sampling with replacement", flush=True)
    return list(rng.choice(cell, k, replace=len(cell) < k)) if k else []


def ratio_ci(da, dt, level, n=1000, seed=0):
    """Per-unit effect mean(da) / mean(dt), with a paired bootstrap interval over images."""
    m = ~np.isnan(da) & ~np.isnan(dt)
    da, dt = da[m], dt[m]
    est = da.mean() / dt.mean()
    idx = np.random.default_rng(seed).integers(0, len(da), (n, len(da)))
    bs = da[idx].mean(1) / dt[idx].mean(1)
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(est), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--truth-per-class", type=int, default=200)
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--attrs", default=",".join(ATTRS), help="nuisances to plant (a subset for smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out, dev = Path(a.out), a.device
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {dev}", flush=True)
    net, _ = mg.load_models(ROOT, dev)
    edits = {LAYER: Edit(getattr(net, LAYER))}

    from torchvision.datasets import MNIST
    ds = MNIST(ROOT / "downloads" / "mnist", train=True, download=False)
    perm = torch.randperm(len(ds.data), generator=torch.Generator().manual_seed(0))
    xr = ds.data.float().div(127.5).sub(1).unsqueeze(1)[perm][:POOL]
    yr = ds.targets[perm][:POOL]
    m = attrs(xr)
    keep = np.all([np.isfinite(m[k]) for k in m], axis=0)
    xr, yr = xr[keep], yr[keep]
    y = yr.numpy()
    m = {k: v[keep] for k, v in m.items()}
    sd = {k: float(within(v, y).std()) for k, v in m.items()}
    print("within-digit SD: " + ", ".join(f"{k} {v:.3f}" for k, v in sd.items()), flush=True)

    # reference sets
    rng = np.random.default_rng(0)
    sets = {"natural": None}
    for att in a.attrs.split(","):
        for q in QS:
            sets[f"{att}@{q}"] = (att, q)
    groups = {}
    for name, spec in sets.items():
        thick, thin = [], []
        for c in range(10):
            idx = np.where(y == c)[0]
            tc = m["thickness"][idx]
            lo_t, hi_t = np.quantile(tc, [1 / 3, 2 / 3])
            T, N = tc >= hi_t, tc <= lo_t
            if spec is None:
                thick += list(rng.choice(idx[T], a.per, replace=False))
                thin += list(rng.choice(idx[N], a.per, replace=False))
                continue
            att, q = spec
            ac = m[att][idx] + rng.uniform(0, 1e-6, len(idx))  # break ties in discrete attributes
            H, L = ac >= np.median(ac), ac < np.median(ac)
            k = int(round(q * a.per))
            thick += choose(idx[T & H], k, rng, name) + choose(idx[T & L], a.per - k, rng, name)
            thin += choose(idx[N & L], k, rng, name) + choose(idx[N & H], a.per - k, rng, name)
        groups[name] = (np.array(thick), np.array(thin))

    union = np.unique(np.concatenate([np.concatenate(g) for g in groups.values()]))
    H_ = activations(net, edits, LAYER, xr[union], yr[union], T_LEARN, dev, seed=1)
    shape = edits[LAYER].last.shape[1:]
    pos = {j: i for i, j in enumerate(union)}

    def gen(classes, per_class, seed):
        yg = torch.tensor(classes).repeat_interleave(per_class)
        x0 = torch.randn(len(yg), 1, 28, 28, generator=torch.Generator().manual_seed(seed))
        return x0, yg

    # unedited runs, cached per (classes, seed)
    base = {}

    def run(classes, per_class, seed, vec=None, amp=0.0):
        x0, yg = gen(classes, per_class, seed)
        key = (tuple(classes), per_class, seed)
        if key not in base:
            base[key] = attrs(sample(net, edits, x0, yg, dev)[0])
        fin = sample(net, edits, x0, yg, dev, LAYER, vec, amp, WIN)[0] if vec is not None else None
        ed = attrs(fin)
        return {k: (ed[k] - base[key][k]) / sd[k] for k in ed}, yg.numpy()

    items_path = out / "items.jsonl"
    done = {json.loads(l)["key"] for l in open(items_path)} if items_path.exists() else set()

    for name, (tk, tn) in groups.items():
        X = H_[[pos[i] for i in np.concatenate([tk, tn])]]
        nk = len(tk)
        d = torch.stack([X[:nk][torch.from_numpy(y[tk] == c)].mean(0) - X[nk:][torch.from_numpy(y[tn] == c)].mean(0)
                         for c in range(10)]).mean(0)
        vec = (d / d.norm()).view(shape).to(dev)
        for split, (aud, held) in SPLITS.items():
            key = f"{name}|{split}"
            if key in done:
                continue
            # REF: reference difference per unit of thickness difference (audit-class reference digits only)
            ref = {}
            ta, na = tk[np.isin(y[tk], aud)], tn[np.isin(y[tn], aud)]
            dth = (m["thickness"][ta].mean() - m["thickness"][na].mean()) / sd["thickness"]
            for att in ATTRS:
                ref[att] = float((m[att][ta].mean() - m[att][na].mean()) / sd[att] / dth)
            # strength on dev seed, audit classes
            amp, lo, hi, hist = 0.4, 0.0, None, []
            for _ in range(8):
                ch, _ = run(aud, a.per_class, SEED_DEV, vec, amp)
                f = float(np.nanmean(ch["thickness"]))
                hist.append((amp, f))
                if abs(f - a.target) < 0.01:
                    break
                lo, hi = (amp, hi) if f < a.target else (lo, amp)
                new = amp * a.target / f if f > 0 else amp * 2
                if hi is not None:
                    new = min(max(new, lo + 0.1 * (hi - lo)), hi - 0.1 * (hi - lo))
                amp = min(new, 6.0)
            amp, f_dev = min(hist, key=lambda h: abs(h[1] - a.target))
            aud_runs = {s: run(aud, a.per_class, s, vec, amp) for s in (SEED_AUD1, SEED_AUD2)}
            truth, _ = run(held, a.truth_per_class, SEED_TRUTH, vec, amp)
            rec = dict(key=key, set=name, split=split, strength=round(amp, 4), dev_change=round(f_dev, 4),
                       truth_thickness=round(float(np.nanmean(truth["thickness"])), 4),
                       planted=sets[name][0] if sets[name] else "", q=sets[name][1] if sets[name] else "")
            for att in ATTRS:
                rec[f"ref_{att}"] = round(ref[att], 4)
                rec[f"truth_{att}"] = [round(v, 4) for v in ratio_ci(truth[att], truth["thickness"], 0.95)]
                for budget in (20, a.per_class):
                    for s, (ch, yg) in aud_runs.items():
                        sel = np.concatenate([np.where(yg == c)[0][:budget] for c in aud])
                        rec[f"out{s}_{budget}_{att}"] = [round(v, 4) for v in
                                                          ratio_ci(ch[att][sel], ch["thickness"][sel], 1 - 0.05 / 7)]
            with open(items_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{key}: a = {amp:.2f}, held-out thickness {rec['truth_thickness']:+.2f}; "
                  + ", ".join(f"{t} {rec['truth_' + t][0]:+.2f}" for t in ATTRS), flush=True)
    report(out, a)


def flags(rec, budget, tau, per_class):
    """Predicted (flag, value) per attribute for each audit."""
    res = {}
    for att in ATTRS:
        r = rec[f"ref_{att}"]
        o1 = rec[f"out{SEED_AUD1}_{budget}_{att}"]
        o2 = rec[f"out{SEED_AUD2}_{budget}_{att}"]
        rep_sig = (o1[1] > 0 and o2[1] > 0) or (o1[2] < 0 and o2[2] < 0)
        rep_val = (o1[0] + o2[0]) / 2
        rep = rep_sig and abs(rep_val) >= tau
        res[att] = {"REF": (abs(r) >= tau, r), "OUT-1": (abs(o1[0]) >= tau, o1[0]),
                    "OUT-rep": (rep, rep_val), "BOTH": (rep and abs(r) >= tau and np.sign(r) == np.sign(rep_val), rep_val)}
    return res


def report(out, a):
    recs = [json.loads(l) for l in open(out / "items.jsonl")]
    rows = []
    budgets = sorted({int(k.split("_")[1]) for k in recs[0] if k.startswith(f"out{SEED_AUD1}_")})
    for rec in recs:
        for att in ATTRS:
            t = rec[f"truth_{att}"]
            rows.append(dict(set=rec["set"], split=rec["split"], attr=att, planted=int(rec["planted"] == att),
                             ref=rec[f"ref_{att}"], truth=t[0], truth_lo=t[1], truth_hi=t[2],
                             **{f"out1_{b}": rec[f"out{SEED_AUD1}_{b}_{att}"][0] for b in budgets},
                             **{f"out2_{b}": rec[f"out{SEED_AUD2}_{b}_{att}"][0] for b in budgets}))
    with open(out / "items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def spearman(u, v):
        ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v))
        return float(np.corrcoef(ru, rv)[0, 1])

    L = ["# MNIST side-effect audit\n",
         f"{len(recs)} (reference set, split) pairs x {len(ATTRS)} attributes = {len(recs) * len(ATTRS)} items. Effects per "
         f"1 SD of achieved thickness change. Audit on 5 classes, truth on the other 5 (seed {SEED_TRUTH}, "
         f"{a.truth_per_class} per class). Held-out thickness change: "
         f"{np.mean([r['truth_thickness'] for r in recs]):+.2f} SD on average "
         f"(range {min(r['truth_thickness'] for r in recs):+.2f} to {max(r['truth_thickness'] for r in recs):+.2f}).\n"]
    L += ["## How well each audit predicts held-out side effects\n",
          "| tau | budget per class | audit | true effects | flagged | hits | false alarms | precision | recall | "
          "rank corr. with truth |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tau in TAUS:
        truth = np.array([(abs(r["truth"]) >= tau) and (r["truth_lo"] > 0 or r["truth_hi"] < 0) for r in rows])
        tv = np.array([r["truth"] for r in rows])
        for b in budgets:
            preds = [flags(rec, b, tau, a.per_class) for rec in recs]
            for aud in ["REF", "OUT-1", "OUT-rep", "BOTH"]:
                if aud == "REF" and b != budgets[0]:
                    continue
                fl = np.array([p[att][aud][0] for p in preds for att in ATTRS])
                val = np.array([p[att][aud][1] for p in preds for att in ATTRS])
                hits = int((fl & truth).sum())
                fa = int((fl & ~truth).sum())
                prec = hits / fl.sum() if fl.sum() else float("nan")
                rec_ = hits / truth.sum() if truth.sum() else float("nan")
                L.append(f"| {tau} | {'-' if aud == 'REF' else b} | {aud} | {int(truth.sum())} | {int(fl.sum())} | {hits} | "
                         f"{fa} | {prec:.2f} | {rec_:.2f} | {spearman(val, tv):.2f} |")
    L.append("")
    L += ["## Planted nuisance: reference difference vs. effect on held-out classes (per SD thickness, mean of splits)\n",
          "| planted | q | reference difference | OUT-rep on audit classes (budget max) | true held-out effect | "
          "true effect on other attributes, largest |", "| --- | --- | --- | --- | --- | --- |"]
    bmax = budgets[-1]
    for name in dict.fromkeys(r["set"] for r in recs):
        rs = [r for r in recs if r["set"] == name]
        att = rs[0]["planted"]
        if att:
            refv = np.mean([r[f"ref_{att}"] for r in rs])
            outv = np.mean([(r[f"out{SEED_AUD1}_{bmax}_{att}"][0] + r[f"out{SEED_AUD2}_{bmax}_{att}"][0]) / 2 for r in rs])
            trv = np.mean([r[f"truth_{att}"][0] for r in rs])
        others = {t: np.mean([r[f"truth_{t}"][0] for r in rs]) for t in ATTRS if t != att}
        big = max(others, key=lambda t: abs(others[t]))
        L.append(f"| {att or 'natural'} | {rs[0]['q'] or '-'} | {refv:+.2f} | {outv:+.2f} | {trv:+.2f} | "
                 f"{big} {others[big]:+.2f} |" if att else
                 f"| natural | - | - | - | - | {big} {others[big]:+.2f} |")
    L.append("")
    L += ["## Natural set: every attribute (per SD thickness, mean of splits)\n",
          "| attribute | reference difference | OUT-rep (budget max) | true held-out effect |", "| --- | --- | --- | --- |"]
    rs = [r for r in recs if r["set"] == "natural"]
    for att in ATTRS:
        L.append(f"| {att} | {np.mean([r[f'ref_{att}'] for r in rs]):+.2f} | "
                 f"{np.mean([(r[f'out{SEED_AUD1}_{bmax}_{att}'][0] + r[f'out{SEED_AUD2}_{bmax}_{att}'][0]) / 2 for r in rs]):+.2f} | "
                 f"{np.mean([r[f'truth_{att}'][0] for r in rs]):+.2f} |")
    L.append("")
    L += ["## Does the audit-class effect transfer? Sign agreement, audit vs. held-out classes\n",
          "Items where the true held-out effect clears tau = 0.15 and its interval excludes 0: share where OUT-rep "
          f"(budget {bmax}) on the audit classes has the same sign, and median |audit - truth|.\n"]
    tr = [(r, att) for r in recs for att in ATTRS
          if abs(r[f"truth_{att}"][0]) >= 0.15 and (r[f"truth_{att}"][1] > 0 or r[f"truth_{att}"][2] < 0)]
    if tr:
        av = np.array([(r[f"out{SEED_AUD1}_{bmax}_{t}"][0] + r[f"out{SEED_AUD2}_{bmax}_{t}"][0]) / 2 for r, t in tr])
        tv = np.array([r[f"truth_{t}"][0] for r, t in tr])
        L.append(f"{len(tr)} items: same sign {np.mean(np.sign(av) == np.sign(tv)):.0%}, "
                 f"median |audit - truth| {np.median(np.abs(av - tv)):.2f}.\n")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
