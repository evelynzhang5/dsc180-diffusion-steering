"""MNIST reference-set bias under a limited budget of nuisance labels.

    python quarter2/mnist/mnist_bias_budget.py --out outputs/q2_mnist_bias_budget [--n 1000] [--draws 5]

Question: at the same achieved thickness change, how much slant leakage and identity damage remain as the number
of slant labels grows, for simple corrections of a direction learned from a slant-biased reference set?

Reference sets (real training digits, pool of 20,000): 60 thick + 60 thin per digit (thick / thin = top / bottom
third of thickness within the digit). A share q of the thick digits is drawn from high-slant ones (above the digit's
median) and the same share of thin digits from low-slant ones; q is chosen per set so the within-digit
correlation of thickness and slant in the set is near 0, +-0.2, +-0.4 or +-0.65 (actual values reported).
Thickness labels (thick / thin, and measured thickness) are known for every reference digit. Slant labels are
known for a random subset of B = 20 or 100 reference digits (5 independent draws each), or all 1,200 (upper bound).
Every label used to fit, tune or choose a correction comes from that subset.

Directions (u2 activations at t = 0.25, per-digit centered, top 256 PCs where a regression is needed):
- none: within-digit thick-minus-thin mean difference (no correction);
- stratify: predict slant for every reference digit (ridge on activations, fit and tuned on the labeled subset;
  true slant on labeled digits), split each digit into quarters of predicted slant, and average the thick-minus-thin
  difference within quarters that contain both groups;
- subtract: slant direction = ridge weights predicting slant from activations (labeled subset), projected out of
  the mean difference;
- controlled: ridge predicting measured thickness from activations with slant regressed out of both (labeled
  subset only).
Each direction is applied on [0.25, 0.375) (2 of 16 DDIM steps, strength a x |h|); a is tuned separately for
thickening and thinning on dev seed 0 to a mean change of +-0.3 SD, frozen, and evaluated on seed 2 (seed 3 stays
untouched). Resumable: finished directions are kept in OUT/runs.jsonl. Writes OUT/sets.csv and OUT/report.md.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_bias import center_by_digit  # noqa: E402
from mnist_calibrate import sample  # noqa: E402
from mnist_thickness import ROOT, Edit, activations, identity, measures, mg, within  # noqa: E402

LAYER = "u2"
T_LEARN = 0.25
WIN = (0.25, 0.375)
POOL = 20000
LEVELS = [0.0, 0.2, -0.2, 0.4, -0.4, 0.65, -0.65]
BUDGETS = [20, 100, "all"]
METHODS = ["stratify", "subtract", "controlled"]
N_PC = 256
SEED_DEV, SEED_EVAL = 0, 2


def pick_q(th, sl, y, per, q, rng):
    thick, thin = [], []
    for c in range(10):
        idx = np.where(y == c)[0]
        tc, sc = th[idx], sl[idx]
        lo_t, hi_t = np.quantile(tc, [1 / 3, 2 / 3])
        med = np.median(sc)
        T, N, H, L = tc >= hi_t, tc <= lo_t, sc >= med, sc < med
        k = int(round(q * per))
        thick += list(rng.choice(idx[T & H], k, replace=False)) + list(rng.choice(idx[T & L], per - k, replace=False))
        thin += list(rng.choice(idx[N & L], k, replace=False)) + list(rng.choice(idx[N & H], per - k, replace=False))
    return np.array(thick), np.array(thin)


def set_corr(idx, th, sl, y):
    return float(np.corrcoef(within(th[idx], y[idx]), within(sl[idx], y[idx]))[0, 1])


def ridge_cv(Z, t, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), folds=5, seed=0):
    """Ridge regression with the penalty picked by k-fold CV (MSE) on the same rows."""
    n, p = Z.shape
    scale = float((Z ** 2).sum() / n)
    perm = np.random.default_rng(seed).permutation(n)
    folds = min(folds, n)
    best, best_lam = np.inf, lams[0]
    for lam in lams:
        err = 0.0
        for f in range(folds):
            te = perm[f::folds]
            tr = np.setdiff1d(perm, te)
            w = torch.linalg.solve(Z[tr].T @ Z[tr] + lam * scale * torch.eye(p), Z[tr].T @ t[tr])
            err += float(((Z[te] @ w - t[te]) ** 2).sum())
        if err < best:
            best, best_lam = err, lam
    return torch.linalg.solve(Z.T @ Z + best_lam * scale * torch.eye(p), Z.T @ t)


def digit_center(v, y, rows):
    """Center v[rows] by per-digit means computed on those rows (digits without rows are left at 0)."""
    out = v[rows].astype(float).copy()
    for c in range(10):
        m = y[rows] == c
        if m.any():
            out[m] -= out[m].mean()
    return out


def directions(X, yy, grp, th_s, sl_s, lab, method):
    """X: set activations (n, D); yy digits; grp +1 thick / -1 thin; th_s, sl_s measured thickness / slant;
    lab: indices (into the set) with slant labels. Returns (unit direction, info)."""
    Xc = center_by_digit(X, yy)
    md = torch.stack([X[torch.from_numpy((yy == c) & (grp == 1))].mean(0)
                      - X[torch.from_numpy((yy == c) & (grp == -1))].mean(0) for c in range(10)]).mean(0)
    md = md / md.norm()
    if method == "none":
        return md, {}
    _, _, V = torch.pca_lowrank(Xc, q=N_PC, center=False)
    Z = Xc @ V
    s_lab = torch.tensor(digit_center(sl_s, yy, lab), dtype=torch.float32)
    if method in ("stratify", "subtract"):
        w = ridge_cv(Z[lab], s_lab)
    if method == "stratify":
        shat = (Z @ w).numpy()
        shat[lab] = s_lab.numpy()  # labeled digits use their true (centered) slant
        diffs, used = [], 0
        for c in range(10):
            m = yy == c
            edges = np.quantile(shat[m], [0.25, 0.5, 0.75])
            b = np.digitize(shat, edges)
            num, den = 0, 0
            for k in range(4):
                tk, tn = m & (b == k) & (grp == 1), m & (b == k) & (grp == -1)
                if tk.sum() >= 2 and tn.sum() >= 2:
                    nb = int(tk.sum() + tn.sum())
                    num = num + nb * (X[torch.from_numpy(tk)].mean(0) - X[torch.from_numpy(tn)].mean(0))
                    den += nb
            if den:
                used += 1
                diffs.append(num / den)
            else:  # no quarter has both groups: fall back to the plain difference for this digit
                diffs.append(X[torch.from_numpy(m & (grp == 1))].mean(0) - X[torch.from_numpy(m & (grp == -1))].mean(0))
        d = torch.stack(diffs).mean(0)
        return d / d.norm(), {"digits_stratified": used}
    if method == "subtract":
        s = V @ w
        s = s / s.norm()
        d = md - (md @ s) * s
        return d / d.norm(), {"cos_md_slant": round(float(md @ s), 3)}
    if method == "controlled":
        t_lab = torch.tensor(digit_center(th_s, yy, lab), dtype=torch.float32)
        Zl = Z[lab] - Z[lab].mean(0)
        sv = s_lab - s_lab.mean()
        Zr = Zl - torch.outer(sv, Zl.T @ sv / (sv @ sv))
        tr = t_lab - t_lab.mean()
        tr = tr - sv * (tr @ sv) / (sv @ sv)
        d = V @ ridge_cv(Zr, tr)
        d = d * torch.sign(d @ md)
        return d / d.norm(), {}
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--levels", default=",".join(map(str, LEVELS)))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    levels = [float(v) for v in a.levels.split(",")]
    out, dev = Path(a.out), a.device
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {dev}", flush=True)
    net, evaluator = mg.load_models(ROOT, dev)
    edits = {LAYER: Edit(getattr(net, LAYER))}

    from torchvision.datasets import MNIST
    ds = MNIST(ROOT / "downloads" / "mnist", train=True, download=False)
    perm = torch.randperm(len(ds.data), generator=torch.Generator().manual_seed(0))
    xr = ds.data.float().div(127.5).sub(1).unsqueeze(1)[perm][:POOL]
    yr = ds.targets[perm][:POOL]
    m = measures(xr)
    keep = ~np.isnan(m["thickness"]) & np.isfinite(m["slant"])
    xr, yr = xr[keep], yr[keep]
    th, sl, y = m["thickness"][keep], m["slant"][keep], yr.numpy()
    sd_th, sd_sl = float(within(th, y).std()), float(within(sl, y).std())

    # reference sets: choose q per level so the realized correlation is closest to the target
    sets, set_rows = {}, []
    for lev in levels:
        best = None
        for q in np.linspace(0, 1, 21):
            tk, tn = pick_q(th, sl, y, a.per, q, np.random.default_rng(int(1000 * q)))
            r = set_corr(np.concatenate([tk, tn]), th, sl, y)
            if best is None or abs(r - lev) < abs(best[0] - lev):
                best = (r, q, tk, tn)
        r, q, tk, tn = best
        sets[lev] = (tk, tn)
        set_rows.append(dict(target=lev, q=round(float(q), 2), corr=round(r, 3),
                             slant_gap=round(float((sl[tk].mean() - sl[tn].mean()) / sd_sl), 3),
                             thick_gap=round(float((th[tk].mean() - th[tn].mean()) / sd_th), 3)))
        print(f"level {lev:+.2f}: q = {q:.2f}, corr = {r:+.3f}", flush=True)
    with open(out / "sets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(set_rows[0]))
        w.writeheader()
        w.writerows(set_rows)

    union = np.unique(np.concatenate([np.concatenate(s) for s in sets.values()]))
    H = activations(net, edits, LAYER, xr[union], yr[union], T_LEARN, dev, seed=1)
    shape = edits[LAYER].last.shape[1:]
    pos = {j: i for i, j in enumerate(union)}

    y_gen = torch.arange(10).repeat_interleave(a.n // 10)
    x_dev = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(SEED_DEV))
    x_ev = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(SEED_EVAL))
    b_dev = measures(sample(net, edits, x_dev, y_gen, dev)[0])["thickness"]
    f_fin = sample(net, edits, x_ev, y_gen, dev)[0]
    f_m = measures(f_fin)
    ok0, _ = identity(evaluator, f_fin, y_gen, dev)

    done = set()
    runs_path = out / "runs.jsonl"
    if runs_path.exists():
        done = {json.loads(l)["key"] for l in open(runs_path)}

    def change(vec, amp):
        fin = sample(net, edits, x_dev, y_gen, dev, LAYER, vec, amp, WIN)[0]
        return float(np.nanmean(measures(fin)["thickness"] - b_dev)) / sd_th

    def evaluate(key, info, d):
        vec = d.view(shape).to(dev)
        res = dict(key=key, **info)
        for sign in (1, -1):
            goal, amp, lo, hi, hist = sign * a.target, 0.4, 0.0, None, []
            for _ in range(8):
                f = change(sign * vec, amp)
                hist.append((amp, f))
                if abs(f - goal) < 0.007:
                    break
                lo, hi = (amp, hi) if abs(f) < abs(goal) else (lo, amp)
                new = amp * goal / f if f * goal > 0 else amp * 2
                if hi is not None:
                    new = min(max(new, lo + 0.1 * (hi - lo)), hi - 0.1 * (hi - lo))
                amp = min(new, 6.0)
            amp, f_dev = min(hist, key=lambda h: abs(h[1] - goal))
            fin = sample(net, edits, x_ev, y_gen, dev, LAYER, sign * vec, amp, WIN)[0]
            mm = measures(fin)
            ok, _ = identity(evaluator, fin, y_gen, dev)
            ch = (mm["thickness"] - f_m["thickness"]) / sd_th
            slc = (mm["slant"] - f_m["slant"]) / sd_sl
            s = "+" if sign == 1 else "-"
            res.update({f"a{s}": round(amp, 4), f"dev{s}": round(f_dev, 4), f"tries{s}": len(hist),
                        f"change{s}": round(float(np.nanmean(ch)), 4), f"slant{s}": round(float(np.nanmean(slc)), 4),
                        f"absslant{s}": round(float(np.nanmean(np.abs(slc))), 4),
                        f"broken{s}": int(((ok0 == 1) & (ok == 0)).sum()),
                        f"slant_img{s}": [None if np.isnan(v) else round(float(v), 3) for v in slc]})
        with open(runs_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"{key}: slant {res['slant+']:+.3f} / {res['slant-']:+.3f}, a {res['a+']:.2f} / {res['a-']:.2f}",
              flush=True)

    for lev in levels:
        tk, tn = sets[lev]
        idx = np.concatenate([tk, tn])
        X = H[[pos[i] for i in idx]]
        yy = y[idx]
        grp = np.r_[np.ones(len(tk)), -np.ones(len(tn))]
        th_s, sl_s = th[idx], sl[idx]
        jobs = [(f"{lev:+.2f}|none|-|0", "none", None)]
        for B in BUDGETS:
            for r in (range(a.draws) if B != "all" else [0]):
                lab = (np.arange(len(idx)) if B == "all"
                       else np.random.default_rng(100 * r + 7).choice(len(idx), B, replace=False))
                for meth in METHODS:
                    jobs.append((f"{lev:+.2f}|{meth}|{B}|{r}", meth, lab))
        for key, meth, lab in jobs:
            if key in done:
                continue
            d, info = directions(X, yy, grp, th_s, sl_s, lab, meth)
            evaluate(key, info, d)

    report(out, a, sd_th, sd_sl, set_rows, levels)


def report(out, a, sd_th, sd_sl, set_rows, levels):
    runs = {}
    for l in open(out / "runs.jsonl"):
        r = json.loads(l)
        runs[r["key"]] = r

    def missed(r, s):
        return abs(r["dev" + s] - (a.target if s == "+" else -a.target)) > 0.03

    def cell(lev, meth, B, field):
        rs = [r for k, r in runs.items() if k.startswith(f"{lev:+.2f}|{meth}|{B}|")]
        vals = [r[field] for r in rs]
        if not vals:
            return "–"
        star = "*" * sum(missed(r, field[-1]) for r in rs) if field.startswith("slant") else ""
        return _cell(vals, field) + star

    def _cell(vals, field):
        if len(vals) == 1:
            return f"{vals[0]:+.3f}" if "slant" in field else f"{vals[0]}"
        v = np.array(vals, float)
        if "slant" in field:
            return f"{v.mean():+.3f} ({v.min():+.2f} to {v.max():+.2f})"
        return f"{v.mean():.1f} ({v.min():.0f}-{v.max():.0f})"

    def img_ci(lev, meth, B, s):
        r = runs.get(f"{lev:+.2f}|{meth}|{B}|0")
        if r is None:
            return ""
        v = np.array([x for x in r[f"slant_img{s}"] if x is not None])
        bm = np.random.default_rng(0).choice(v, (1000, len(v))).mean(1)
        return f" [{np.quantile(bm, 0.025):+.3f}, {np.quantile(bm, 0.975):+.3f}]"

    L = ["# MNIST reference-set bias under a limited slant-label budget\n",
         f"{a.per} thick + {a.per} thin reference digits per digit class. Directions from u2 at t = {T_LEARN}; edit on "
         f"t in [{WIN[0]}, {WIN[1]}), strength tuned per direction and sign on dev seed {SEED_DEV} to +-{a.target} SD "
         f"of thickness, evaluated on seed {SEED_EVAL} ({a.n} digits). Slant in within-digit SD of real slant "
         f"({sd_sl:.3f}); thickness SD {sd_th:.3f} px. Budgets of 20 and 100: mean over {a.draws} independent label draws "
         "(min to max across draws). Square brackets: 95% interval over evaluation images for one run "
         "(the uncorrected direction, or the full-label run). * = one draw whose tuned strength missed the thickness "
         "target on dev by more than 0.03 SD (direction too weak or wrong-signed); see the strength table.\n",
         "## Reference sets\n", "| target corr | share q | actual corr | thick - thin: slant (SD) | thick - thin: thickness (SD) |",
         "| --- | --- | --- | --- | --- |"]
    for r in set_rows:
        L.append(f"| {r['target']:+.2f} | {r['q']:.2f} | {r['corr']:+.3f} | {r['slant_gap']:+.2f} | {r['thick_gap']:+.2f} |")
    L.append("")
    for s, name in (("+", "thickening"), ("-", "thinning")):
        L += [f"## Signed slant change when {name} (matched thickness)\n",
              "| actual corr | none | " + " | ".join(f"{m} {B}" for m in METHODS for B in BUDGETS) + " |",
              "| --- | --- |" + " --- |" * (len(METHODS) * len(BUDGETS))]
        for lev, sr in zip(levels, set_rows):
            row = [f"{cell(lev, 'none', '-', 'slant' + s)}{img_ci(lev, 'none', '-', s)}"]
            for m in METHODS:
                for B in BUDGETS:
                    row.append(cell(lev, m, B, "slant" + s) + (img_ci(lev, m, B, s) if B == "all" else ""))
            L.append(f"| {sr['corr']:+.2f} | " + " | ".join(row) + " |")
        L.append("")
    L += ["## Stratify: digit classes where some slant quarter contains both thick and thin digits (of 10)\n",
          "| actual corr | " + " | ".join(f"budget {B}" for B in BUDGETS) + " |", "| --- |" + " --- |" * len(BUDGETS)]
    for lev, sr in zip(levels, set_rows):
        v = [np.mean([r["digits_stratified"] for k, r in runs.items() if k.startswith(f"{lev:+.2f}|stratify|{B}|")]
                     or [np.nan]) for B in BUDGETS]
        L.append(f"| {sr['corr']:+.2f} | " + " | ".join(f"{x:.1f}" for x in v) + " |")
    L.append("")
    L += ["## Identity damage: digits correct -> wrong (of 1000), thickening + thinning\n",
          "| actual corr | none | " + " | ".join(f"{m} {B}" for m in METHODS for B in BUDGETS) + " |",
          "| --- | --- |" + " --- |" * (len(METHODS) * len(BUDGETS))]
    for lev, sr in zip(levels, set_rows):
        def brk(m, B):
            v = [r["broken+"] + r["broken-"] for k, r in runs.items() if k.startswith(f"{lev:+.2f}|{m}|{B}|")]
            return "–" if not v else (f"{v[0]}" if len(v) == 1 else f"{np.mean(v):.1f} ({min(v)}-{max(v)})")
        L.append(f"| {sr['corr']:+.2f} | {brk('none', '-')} | " + " | ".join(brk(m, B) for m in METHODS for B in BUDGETS) + " |")
    L += ["", "## Strength needed (a, thickening / thinning) and achieved thickness on the evaluation seed\n",
          "| actual corr | none | " + " | ".join(f"{m} {B}" for m in METHODS for B in BUDGETS) + " |",
          "| --- | --- |" + " --- |" * (len(METHODS) * len(BUDGETS))]
    for lev, sr in zip(levels, set_rows):
        def st(m, B):
            v = [r for k, r in runs.items() if k.startswith(f"{lev:+.2f}|{m}|{B}|")]
            if not v:
                return "–"
            return (f"{np.mean([r['a+'] for r in v]):.2f} / {np.mean([r['a-'] for r in v]):.2f} "
                    f"({np.mean([r['change+'] for r in v]):+.2f} / {np.mean([r['change-'] for r in v]):+.2f})")
        L.append(f"| {sr['corr']:+.2f} | {st('none', '-')} | " + " | ".join(st(m, B) for m in METHODS for B in BUDGETS) + " |")
    L.append("")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
