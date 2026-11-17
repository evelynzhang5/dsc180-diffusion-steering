"""Round 27: is the patch-to-patch blur pattern real or a measurement effect?

Uses round 25's saved baseline outputs (fp32 PISA + original gate + radius curve; corrected runs on 12 held-out
patches x positions 0/2/4 x targets 3, 4, 5, 6, 8, 11.6, 15.5 px). No new BokehDiff generations.

Second estimator, shape-agnostic: per cell, the blur kernel is estimated as a non-negative radial profile
    y_lum ~ sum_b w_b (x_lum * ring_b) + offset,  w_b >= 0, rings of 1 px width up to R_MAX
(ring_0 = the identity), and its size is summarized by the second moment r2 = sqrt(sum w_b E_b[r^2] / sum w_b).
A disk of radius R has r2 = R / sqrt(2); a Gaussian of sigma s has r2 = s * sqrt(2).
Steps:
  V  validation: known disk / Gaussian kernels applied to these exact patches -> r2 must be recovered per patch
  A  real outputs: per-patch pattern of disk-fit radius (round 25) vs of r2; across-patch CV of each
  B  measurement-only reproduction: the median real kernel profile, applied synthetically to every patch, then read
     with the disk-fit estimator -> does measurement alone reproduce the real per-patch pattern?
Usage: python check27.py <round 25 output dir> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.optimize import lsq_linear
from scipy.signal import fftconvolve
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402

R25, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
R_MAX = 34
TARGETS = [3, 4, 5, 6, 8, 11.625, 15.5]
lib = PT.library()
groups = [[p for p in lib if p["split"] == "test"][0::2], [p for p in lib if p["split"] == "test"][1::2]]
med = lambda x: float(np.median(x))
lum = lambda a: a.astype(np.float64).mean(2)


def ring(b):
    if b == 0:
        return np.ones((1, 1))
    k_out, k_in = m18.disk(b + 0.5), m18.disk(b - 0.5)
    ao, ai = k_out * (np.pi * (b + 0.5) ** 2), k_in * (np.pi * (b - 0.5) ** 2) if b > 0.5 else k_in * 0
    p = (ao.shape[0] - ai.shape[0]) // 2
    ai = np.pad(ai, p)
    r = np.clip(ao - ai, 0, None)
    return r / r.sum()


RINGS = [ring(b) for b in range(R_MAX + 1)]
def _m2(k):
    n = k.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return float((k * (yy ** 2 + xx ** 2)).sum())
E2 = np.array([_m2(k) for k in RINGS])


def ring_stack(xl):
    p = R_MAX + 1
    xp = np.pad(xl, p, mode="reflect")
    return [fftconvolve(xp, k, mode="same")[p:-p, p:-p] for k in RINGS]


def fit_profile(yl, rs, mask):
    A = np.stack([r[mask] for r in rs] + [np.ones(mask.sum())], 1)
    lb = np.r_[np.zeros(len(rs)), -np.inf]
    sol = lsq_linear(A, yl[mask], bounds=(lb, np.full(len(rs) + 1, np.inf)), method="bvls")
    w = sol.x[:-1]
    r2 = float(np.sqrt((w * E2).sum() / max(w.sum(), 1e-12)))
    return w, r2


def disk_equiv(r2):
    return float(np.sqrt(2) * r2)              # disk radius with the same second moment


# ---------------- V: validation on known kernels
rng = np.random.default_rng(0)
val = []
known = [("disk", r, m18.disk(r), r / np.sqrt(2)) for r in (4, 8, 12)] + [("gauss", s, m18.gauss(s), s * np.sqrt(2)) for s in (2, 4, 6)]
for gi, grp in enumerate(groups):
    img, cid, where = PT.layout(grp, 0)
    im = img.astype(np.float32)
    rs = ring_stack(lum(im))
    ints = {k: D.interior(cid, k) for k in PT.FREE}
    for kind, par, ker, r2_true in known:
        y = m18.degrade(m18.blur(im, ker), rng)
        for k in PT.FREE:
            w, r2 = fit_profile(lum(y), rs, ints[k])
            val.append(dict(kind=kind, par=par, patch=where[k], r2=r2, r2_true=float(r2_true), rel_err=r2 / r2_true - 1))
val_by = {f"{k}{p}": dict(median_rel_err=med([v["rel_err"] for v in val if v["kind"] == k and v["par"] == p]),
                         max_abs_rel_err=float(max(abs(v["rel_err"]) for v in val if v["kind"] == k and v["par"] == p)),
                         cv=float(np.std(x := [v["r2"] for v in val if v["kind"] == k and v["par"] == p]) / np.mean(x)))
          for k, p in sorted({(v["kind"], v["par"]) for v in val})}
val_pass = all(abs(v["median_rel_err"]) <= 0.10 and v["cv"] <= 0.03 for v in val_by.values())
print("validation", "PASS" if val_pass else "FAIL", {k: {a: round(b, 3) for a, b in v.items()} for k, v in val_by.items()}, flush=True)

# ---------------- A: real outputs (round 25 baseline corrected runs)
rows = []
for gi, grp in enumerate(groups):
    for rot in (0, 2, 4):
        img, cid, where = PT.layout(grp, rot)
        im = img.astype(np.float32)
        rs = ring_stack(lum(im))
        st = m18.stack(im, "disk")
        ints = {k: D.interior(cid, k) for k in PT.FREE}
        Bs = {k: M.stack_at(st, ints[k]) for k in PT.FREE}
        del st
        for t in TARGETS:
            y = np.asarray(Image.open(f"{R25}/img/cor_g{gi}_r{rot}_G0_t{t:.1f}.png")).astype(np.float32)
            for k in PT.FREE:
                w, r2 = fit_profile(lum(y), rs, ints[k])
                fd = M.fit_fast(y[ints[k]], Bs[k])
                rows.append(dict(patch=where[k], rot=rot, cell=k, target=t, disk_r=fd["v"], r2=r2, disk_equiv=disk_equiv(r2),
                                 profile=(w / max(w.sum(), 1e-12)).tolist()))
    print("real outputs measured, group", gi, flush=True)

patches = sorted({r["patch"] for r in rows})
def pattern(key):
    """per patch: median over positions and targets of (measure / median measure of all patches at that target)"""
    refs = {t: med([r[key] for r in rows if r["target"] == t]) for t in TARGETS}
    return {p: med([r[key] / refs[r["target"]] for r in rows if r["patch"] == p]) for p in patches}
def cv_at(key, t):
    v = [np.mean([r[key] for r in rows if r["patch"] == p and r["target"] == t]) for p in patches]
    return float(np.std(v) / np.mean(v))
pat_disk, pat_r2 = pattern("disk_r"), pattern("r2")
cv_disk = {f"{t:g}": cv_at("disk_r", t) for t in TARGETS}
cv_r2 = {f"{t:g}": cv_at("r2", t) for t in TARGETS}
rho_A = float(spearmanr([pat_disk[p] for p in patches], [pat_r2[p] for p in patches]).correlation)
print("A: CV disk-fit", {k: round(v, 3) for k, v in cv_disk.items()}, "| CV second-moment", {k: round(v, 3) for k, v in cv_r2.items()}, flush=True)
print("A: per-patch disk", {p: round(v, 2) for p, v in pat_disk.items()}, flush=True)
print("A: per-patch r2  ", {p: round(v, 2) for p, v in pat_r2.items()}, "| Spearman", round(rho_A, 2), flush=True)

# ---------------- B: measurement-only reproduction with the median real kernel
prof = {}
for t in (5, 8, 11.625):
    P = np.median(np.array([r["profile"] for r in rows if r["target"] == t]), 0)
    prof[f"{t:g}"] = P.tolist()
syn = []
for t, P in prof.items():
    P = np.array(P)
    ker = sum(P[b] * RINGS[b] if RINGS[b].shape[0] == RINGS[-1].shape[0] else
              P[b] * np.pad(RINGS[b], (RINGS[-1].shape[0] - RINGS[b].shape[0]) // 2) for b in range(len(RINGS)))
    ker = ker / ker.sum()
    for gi, grp in enumerate(groups):
        for rot in (0, 2, 4):
            img, cid, where = PT.layout(grp, rot)
            im = img.astype(np.float32)
            st = m18.stack(im, "disk")
            ints = {k: D.interior(cid, k) for k in PT.FREE}
            y = m18.degrade(m18.blur(im, ker), rng)
            for k in PT.FREE:
                syn.append(dict(target=t, patch=where[k], disk_r=M.fit_fast(y[ints[k]], M.stack_at(st, ints[k]))["v"]))
            del st
refs = {t: med([s["disk_r"] for s in syn if s["target"] == t]) for t in prof}
pat_syn = {p: med([s["disk_r"] / refs[s["target"]] for s in syn if s["patch"] == p]) for p in patches}
cv_syn = {t: float(np.std(v := [np.mean([s["disk_r"] for s in syn if s["patch"] == p and s["target"] == t]) for p in patches]) / np.mean(v)) for t in prof}
rho_B = float(spearmanr([pat_disk[p] for p in patches], [pat_syn[p] for p in patches]).correlation)
print("B: identical known kernel on every patch -> disk-fit pattern", {p: round(v, 2) for p, v in pat_syn.items()},
      "| CV", {k: round(v, 3) for k, v in cv_syn.items()}, "| Spearman with real", round(rho_B, 2), flush=True)

# ---------------- frozen decision
cvd, cvr = med(list(cv_disk.values())), med(list(cv_r2.values()))
cvs = med(list(cv_syn.values()))
if not val_pass:
    outcome = "V: the shape-agnostic estimator fails validation; no conclusion"
elif cvr <= 0.5 * cvd and rho_B >= 0.7:
    outcome = "K1: mostly measurement - the patch pattern disappears with a shape-agnostic size and is reproduced by the disk estimator on an identical kernel"
elif cvr >= 0.8 * cvd and cvs <= 0.5 * cvd:
    outcome = "K2: real model effect - the pattern persists with a shape-agnostic size, and an identical kernel does not reproduce it"
else:
    outcome = "K3: mixed - part measurement, part model (see numbers)"
summary = dict(outcome=outcome, validation_pass=val_pass, validation=val_by, cv_disk_fit=cv_disk, cv_second_moment=cv_r2,
               cv_disk_fit_on_identical_kernel=cv_syn, cv_medians=dict(disk=cvd, r2=cvr, synthetic=cvs),
               pattern_disk=pat_disk, pattern_r2=pat_r2, pattern_synthetic=pat_syn, spearman_disk_vs_r2=rho_A,
               spearman_disk_vs_synthetic=rho_B, median_profiles=prof,
               high_contrast_profiles={p: np.median(np.array([r["profile"] for r in rows if r["patch"] == p and r["target"] == 8]), 0).tolist()
                                       for p in ("coins_0", "coins_1", "rocket_1", "gravel_0", "chelsea_0")})
json.dump(summary, open(f"{res}/kernel_check.json", "w"), indent=1)
json.dump(dict(validation=val, rows=rows, synthetic=syn), open(f"{res}/kernel_rows.json", "w"))
print("OUTCOME", outcome, "| CV medians", {k: round(v, 3) for k, v in summary["cv_medians"].items()}, flush=True)

# profile figure: median radial profile at target 8 for typical vs high-contrast patches
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for p, P in summary["high_contrast_profiles"].items():
    ax[0].plot(range(R_MAX + 1), P, label=p)
ax[0].set_xlabel("ring radius (px)"); ax[0].set_ylabel("weight"); ax[0].set_title("Kernel profile, target 8 px"); ax[0].legend(fontsize=7)
ax[1].scatter([pat_disk[p] for p in patches], [pat_r2[p] for p in patches], label="second moment")
ax[1].scatter([pat_disk[p] for p in patches], [pat_syn[p] for p in patches], marker="x", label="identical kernel (disk fit)")
for p in patches:
    ax[1].annotate(p, (pat_disk[p], pat_r2[p]), fontsize=6)
ax[1].axline((1, 1), slope=1, color="gray", lw=0.5)
ax[1].set_xlabel("real per-patch ratio, disk fit"); ax[1].set_ylabel("ratio"); ax[1].legend(fontsize=7); ax[1].set_title("Per-patch pattern")
fig.tight_layout()
fig.savefig(f"{res}/kernel_check.png", dpi=130)
print("figure written")
