"""MNIST reference-set bias test: does a thickness direction learned from a biased reference set also change slant,
at the same achieved thickness change?

    python quarter2/mnist/mnist_bias.py --out outputs/q2_mnist_bias [--n 1000] [--per 60] [--target 0.3]

Reference sets are drawn from real training digits (a pool of 20,000, disjoint roles: the generator is fixed and the
evaluation digits are generated). Within each digit class, "thick" = top third of thickness, "thin" = bottom third,
"high / low slant" = above / below that digit's median slant. Each set has `per` thick and `per` thin digits per class:
- natural: drawn from the thick / thin thirds without looking at slant;
- balanced: half high-slant and half low-slant in each group;
- aligned: thick digits all high-slant, thin digits all low-slant (thickness and slant confounded, +);
- opposed: thick digits all low-slant, thin digits all high-slant (confounded, -).
Directions (u2 activations at t = 0.25, per-digit centered):
- meandiff: within-digit thick-minus-thin mean difference, averaged over digits (the direction used so far);
- ridge: ridge-regression weights for thick (+1) vs. thin (-1), penalty by 5-fold CV, on the top 256 PCs;
- rfm: Recursive Feature Machine style, Laplace-kernel ridge with the Average Gradient Outer Product (AGOP) updated
  for 3 rounds, direction = top eigenvector of the AGOP (top 256 PCs); sign aligned with meandiff.
Each direction is applied on [0.25, 0.375) (2 of 16 DDIM steps, relative strength a x |h|), with a tuned on dev seed
0 to give a mean thickness change of +-target SD, frozen, and evaluated on seed 2. Main outcome: the signed slant
change at matched thickness (units: within-digit SD of real slant), overall and by digit; also identity transitions,
centroid shift and the directions' cosines. Writes OUT/sets.csv, OUT/per_image.csv and OUT/report.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_calibrate import boot, ci, col, sample, write  # noqa: E402
from mnist_thickness import ROOT, Edit, activations, identity, measures, mg, within  # noqa: E402

LAYER = "u2"
T_LEARN = 0.25
WIN = (0.25, 0.375)
POOL = 20000
SETS = ["natural", "balanced", "aligned", "opposed"]
LEARNERS = ["meandiff", "ridge", "rfm"]
N_PC = 256
SEED_DEV, SEED_EVAL = 0, 2  # seed 3 stays untouched


def pick(th, sl, y, per, kind, rng):
    """Indices of thick and thin reference digits for one set."""
    thick, thin = [], []
    for c in range(10):
        idx = np.where(y == c)[0]
        tc, sc = th[idx], sl[idx]
        lo_t, hi_t = np.quantile(tc, [1 / 3, 2 / 3])
        med = np.median(sc)
        cell = lambda tmask, smask: idx[tmask & smask]
        T, N = tc >= hi_t, tc <= lo_t
        H, L = sc >= med, sc < med
        A = np.ones_like(T)
        h = per // 2
        spec = {"natural": [(T, A, per)], "balanced": [(T, H, h), (T, L, per - h)],
                "aligned": [(T, H, per)], "opposed": [(T, L, per)]}[kind]
        spec_thin = {"natural": [(N, A, per)], "balanced": [(N, H, h), (N, L, per - h)],
                     "aligned": [(N, L, per)], "opposed": [(N, H, per)]}[kind]
        for out, sp in ((thick, spec), (thin, spec_thin)):
            for tm, sm, k in sp:
                out += list(rng.choice(cell(tm, sm), k, replace=False))
    return np.array(thick), np.array(thin)


def center_by_digit(X, y):
    X = X.clone()
    for c in range(10):
        m = torch.from_numpy(y == c)
        X[m] -= X[m].mean(0)
    return X


def ridge_dir(Z, lab, folds=5, seed=0):
    n, p = Z.shape
    scale = float((Z ** 2).sum() / n)
    perm = np.random.default_rng(seed).permutation(n)
    best, best_lam = -np.inf, None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        acc = []
        for f in range(folds):
            te = perm[f::folds]
            tr = np.setdiff1d(perm, te)
            w = torch.linalg.solve(Z[tr].T @ Z[tr] + lam * scale * torch.eye(p), Z[tr].T @ lab[tr])
            acc.append(float(((Z[te] @ w).sign() == lab[te]).float().mean()))
        if np.mean(acc) > best:
            best, best_lam = np.mean(acc), lam
    w = torch.linalg.solve(Z.T @ Z + best_lam * scale * torch.eye(p), Z.T @ lab)
    return w, best, best_lam


def rfm_dir(Z, lab, iters=3, reg=1e-3, bs=200):
    """Laplace-kernel RFM: alternate kernel ridge and M <- AGOP. Returns the top eigenvector of the final AGOP."""
    n, p = Z.shape
    Z = Z / Z.norm(dim=1).median()
    M = torch.eye(p)

    def dist(A, B):
        AM = A @ M
        d2 = (AM * A).sum(1)[:, None] + (B @ M * B).sum(1)[None] - 2 * AM @ B.T
        return d2.clamp_min(1e-12).sqrt()

    for _ in range(iters):
        D = dist(Z, Z)
        L = float(D.median())
        alpha = torch.linalg.solve(torch.exp(-D / L) + reg * torch.eye(n), lab)
        G = []
        for i in range(0, n, bs):
            xb = Z[i:i + bs]
            Db = dist(xb, Z).clamp_min(1e-6)
            coef = alpha[None] * torch.exp(-Db / L) / (Db * L)  # (b, n)
            # grad f(x) = -sum_j coef_j M (x - z_j)
            g = -(coef.sum(1, keepdim=True) * xb - coef @ Z) @ M
            G.append(g)
        G = torch.cat(G)
        M = G.T @ G / n
        M = M / M.trace() * p
    evals, evecs = torch.linalg.eigh(M)
    return evecs[:, -1], float(evals[-1] / evals.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    out, dev = Path(a.out), a.device
    out.mkdir(parents=True, exist_ok=True)
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
    nat_corr = float(np.corrcoef(within(th, y), within(sl, y))[0, 1])
    print(f"pool {len(y)}: within-digit corr(thickness, slant) = {nat_corr:+.3f}", flush=True)

    rng = np.random.default_rng(0)
    groups = {k: pick(th, sl, y, a.per, k, rng) for k in SETS}
    union = np.unique(np.concatenate([np.concatenate(g) for g in groups.values()]))
    H = activations(net, edits, LAYER, xr[union], yr[union], T_LEARN, dev, seed=1)
    shape = edits[LAYER].last.shape[1:]
    pos = {j: i for i, j in enumerate(union)}

    # a reference slant direction (high-minus-low slant, middle-third thickness, within digit) for cosines only
    mid = []
    for c in range(10):
        idx = np.where(y == c)[0]
        lo_t, hi_t = np.quantile(th[idx], [1 / 3, 2 / 3])
        mid.append(idx[(th[idx] > lo_t) & (th[idx] < hi_t)])
    mid = np.concatenate(mid)
    mid = rng.choice(mid, min(3000, len(mid)), replace=False)
    Hs = activations(net, edits, LAYER, xr[mid], yr[mid], T_LEARN, dev, seed=1)
    sdiff = []
    for c in range(10):
        mc = y[mid] == c
        med = np.median(sl[mid][mc])
        hi, lo = mc & (sl[mid] >= med), mc & (sl[mid] < med)
        sdiff.append(Hs[torch.from_numpy(hi)].mean(0) - Hs[torch.from_numpy(lo)].mean(0))
    slant_dir = torch.stack(sdiff).mean(0)
    slant_dir = slant_dir / slant_dir.norm()

    dirs, set_rows = {}, []
    for k in SETS:
        tk, tn = groups[k]
        idx = np.concatenate([tk, tn])
        X = H[[pos[i] for i in idx]]
        yy = y[idx]
        lab = torch.cat([torch.ones(len(tk)), -torch.ones(len(tn))])
        Xc = center_by_digit(X, yy)
        md = []
        for c in range(10):
            md.append(X[: len(tk)][torch.from_numpy(y[tk] == c)].mean(0) - X[len(tk):][torch.from_numpy(y[tn] == c)].mean(0))
        md = torch.stack(md).mean(0)
        dirs[(k, "meandiff")] = md / md.norm()
        U, S, V = torch.pca_lowrank(Xc, q=N_PC, center=False)
        Z = Xc @ V
        w, acc, lam = ridge_dir(Z, lab)
        d = V @ w
        dirs[(k, "ridge")] = d / d.norm()
        u, share = rfm_dir(Z, lab)
        d = V @ u
        d = d * torch.sign(d @ dirs[(k, "meandiff")])
        dirs[(k, "rfm")] = d / d.norm()
        st, ss = within(th[idx], yy), within(sl[idx], yy)
        set_rows.append(dict(set=k, n=len(idx), corr=round(float(np.corrcoef(st, ss)[0, 1]), 3),
                             thick_gap=round(float((th[tk].mean() - th[tn].mean()) / sd_th), 3),
                             slant_gap=round(float((sl[tk].mean() - sl[tn].mean()) / sd_sl), 3),
                             ridge_cv_acc=round(acc, 3), ridge_lam=lam, rfm_top_share=round(share, 3)))
        print(f"set {k}: {set_rows[-1]}", flush=True)
    write(out / "sets.csv", set_rows)

    # ------------------------------------------------------------ calibrate on dev, evaluate on a fresh seed
    y_gen = torch.arange(10).repeat_interleave(a.n // 10)
    x_dev = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(SEED_DEV))
    x_ev = torch.randn(len(y_gen), 1, 28, 28, generator=torch.Generator().manual_seed(SEED_EVAL))
    b_dev = measures(sample(net, edits, x_dev, y_gen, dev)[0])["thickness"]
    f_fin = sample(net, edits, x_ev, y_gen, dev)[0]
    f_m = measures(f_fin)
    ok0, p0 = identity(evaluator, f_fin, y_gen, dev)

    def change(vec, amp):
        fin = sample(net, edits, x_dev, y_gen, dev, LAYER, vec, amp, WIN)[0]
        return float(np.nanmean(measures(fin)["thickness"] - b_dev)) / sd_th

    rows, log = [], []
    for (k, learner), d in dirs.items():
        vec = d.view(shape).to(dev)
        for sign in (1, -1):
            goal, amp, lo, hi, hist = sign * a.target, 0.35, 0.0, None, []
            for _ in range(8):
                f = change(sign * vec, amp)
                hist.append((amp, f))
                if abs(f - goal) < 0.007:
                    break
                lo, hi = (amp, hi) if abs(f) < abs(goal) else (lo, amp)
                new = amp * goal / f if f * goal > 0 else amp * 2
                if hi is not None:
                    new = min(max(new, lo + 0.1 * (hi - lo)), hi - 0.1 * (hi - lo))
                amp = min(new, 4.0)
            amp, f_dev = min(hist, key=lambda h: abs(h[1] - goal))
            fin = sample(net, edits, x_ev, y_gen, dev, LAYER, sign * vec, amp, WIN)[0]
            mm = measures(fin)
            ok, p = identity(evaluator, fin, y_gen, dev)
            log.append(dict(set=k, learner=learner, sign=sign, strength=amp, dev_change=f_dev, tries=len(hist)))
            print(f"{k} {learner} {sign:+d}: a = {amp:.3f}, dev {f_dev:+.3f}", flush=True)
            for i in range(len(y_gen)):
                rows.append(dict(set=k, learner=learner, sign=sign, image=i, digit=int(y_gen[i]),
                                 change=round(float((mm["thickness"][i] - f_m["thickness"][i]) / sd_th), 4),
                                 slant=round(float((mm["slant"][i] - f_m["slant"][i]) / sd_sl), 4),
                                 shift=round(float(np.hypot(mm["cx"][i] - f_m["cx"][i], mm["cy"][i] - f_m["cy"][i])), 4),
                                 correct0=int(ok0[i]), correct=int(ok[i]), p_drop=round(float(p0[i] - p[i]), 4)))
    write(out / "per_image.csv", rows)
    report(rows, log, set_rows, dirs, slant_dir, nat_corr, sd_th, sd_sl, a, out)


def report(rows, log, set_rows, dirs, slant_dir, nat_corr, sd_th, sd_sl, a, out):
    L = ["# MNIST reference-set bias test\n",
         f"Pool of real digits: within-digit corr(thickness, slant) = {nat_corr:+.3f}. {a.per} thick + {a.per} thin "
         f"per digit per set. Directions from u2 at t = {T_LEARN}; edit on t in [{WIN[0]}, {WIN[1]}), strength tuned "
         f"on dev seed {SEED_DEV} to +-{a.target} SD of thickness, evaluated on seed {SEED_EVAL} ({a.n} digits). "
         f"Thickness SD {sd_th:.3f} px, slant SD {sd_sl:.3f}. Mean (95% interval over images).\n",
         "## Reference sets\n",
         "| set | within-digit corr(thickness, slant) | thick - thin: thickness (SD) | thick - thin: slant (SD) | "
         "ridge CV accuracy | RFM top-eigenvalue share |", "| --- | --- | --- | --- | --- | --- |"]
    for r in set_rows:
        L.append(f"| {r['set']} | {r['corr']:+.2f} | {r['thick_gap']:+.2f} | {r['slant_gap']:+.2f} | "
                 f"{r['ridge_cv_acc']:.2f} | {r['rfm_top_share']:.2f} |")
    L += ["", "## Directions: cosine with the balanced meandiff direction / with a slant direction\n",
          "| set | " + " | ".join(LEARNERS) + " |", "| --- |" + " --- |" * len(LEARNERS)]
    ref = dirs[("balanced", "meandiff")]
    for k in SETS:
        L.append(f"| {k} | " + " | ".join(f"{float(dirs[(k, l)] @ ref):.2f} / {float(dirs[(k, l)] @ slant_dir):+.2f}"
                                          for l in LEARNERS) + " |")
    L += ["", "## At matched thickness: slant change (thicken / thin)\n",
          "Signed slant change in SD of real slant; a confounded direction should move slant the same way as its "
          "reference set (aligned: thicker = higher slant).\n",
          "| set | learner | strength a (+ / -) | thickness change (+ / -) | signed slant change, thicken | "
          "signed slant change, thin | abs. slant change | centroid shift (px) | correct->wrong (+ / -) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in SETS:
        for l in LEARNERS:
            g = {s: next(r for r in log if r["set"] == k and r["learner"] == l and r["sign"] == s) for s in (1, -1)}
            kw = dict(set=k, learner=l)
            br = [int(((col(rows, "correct0", sign=s, **kw) == 1) & (col(rows, "correct", sign=s, **kw) == 0)).sum())
                  for s in (1, -1)]
            L.append(f"| {k} | {l} | {g[1]['strength']:.2f} / {g[-1]['strength']:.2f} | "
                     f"{np.nanmean(col(rows, 'change', sign=1, **kw)):+.2f} / "
                     f"{np.nanmean(col(rows, 'change', sign=-1, **kw)):+.2f} | "
                     f"{ci(col(rows, 'slant', sign=1, **kw), '+.3f')} | {ci(col(rows, 'slant', sign=-1, **kw), '+.3f')} | "
                     f"{np.nanmean(np.abs(col(rows, 'slant', **kw))):.3f} | {np.nanmean(col(rows, 'shift', **kw)):.3f} | "
                     f"{br[0]} / {br[1]} |")
    L += ["", "## Signed slant change when thickening, by digit (meandiff)\n",
          "| set | " + " | ".join(f"{c}" for c in range(10)) + " |", "| --- |" + " --- |" * 10]
    for k in SETS:
        L.append(f"| {k} | " + " | ".join(f"{np.nanmean(col(rows, 'slant', set=k, learner='meandiff', sign=1, digit=c)):+.2f}"
                                          for c in range(10)) + " |")
    L.append("")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
