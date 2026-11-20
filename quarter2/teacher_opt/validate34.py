"""Validate round 34's revised error against known corruptions (CPU only; real patches of round 29's 8 cases).

For each case cell and target t, synthetic outputs are built from the input with known operations and scored by
the old score (aff_mae, round 29) and the new primary (off_mae):
  contrast   target with contrast scaled by 0.5 / 0.75 about its mean       -> must not score better than target
  real-contrast  BokehDiff's saved baseline output with contrast 0.5      -> new score must not drop
  texture    target further smoothed (Gaussian sigma 2, 4 px)              -> must score worse than target
  shift      target shifted 2 / 4 px                                       -> must score worse, monotone
  radius     disk blur at t-3..t+3                                         -> new score monotone in |r - t|
  kernel     Gaussian with the disk's second moment; ring (rim-weighted)   -> must score worse than target
  offset     target + per-channel offset (+8, -5, +3)                      -> new score unchanged (to 1e-3; float32 rounding)
Validation PASSES if every "must" holds in all 8 cases. Writes results/metric_validation.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, shift as nd_shift

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import metrics34 as Q  # noqa: E402

R25 = sys.argv[1] if len(sys.argv) > 1 else "../../outputs/q2_lora_scale"
img, cid, _ = Q.scene()
im = img.astype(np.float32)
clip = lambda x: np.clip(x, 0, 255)


def ring(r):
    k = m18.disk(r).astype(np.float64)
    n = k.shape[0] // 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    k = k * (0.4 + 0.6 * np.sqrt(xx ** 2 + yy ** 2) / r)
    return (k / k.sum()).astype(np.float32)


def gauss_like(r):
    s = r / 2.0                                    # disk second moment r^2/2 per axis -> sigma = r/sqrt(2)... per-axis var r^2/4
    return lambda x: np.stack([gaussian_filter(x[..., c], s) for c in range(3)], -1)


checks, table = [], {}
for t in Q.TARGETS:
    tg = Q.target(img, t)
    base = np.asarray(Image.open(f"{R25}/img/base_t{t:.1f}.png")).astype(np.float32)
    for k, p in Q.CASES:
        m = D.interior(cid, k)
        mu = tg[m].mean(0)
        S = {}
        S["target"] = tg
        for c in (0.5, 0.75):
            S[f"contrast{c}"] = clip(mu + c * (tg - mu))
        for s in (2, 4):
            S[f"texture_sigma{s}"] = np.stack([gaussian_filter(tg[..., ch], s) for ch in range(3)], -1)
        for d in (2, 4):
            S[f"shift{d}"] = nd_shift(tg, (d, d, 0), order=1, mode="nearest")
        for dr in (-3, -2, -1, 1, 2, 3):
            S[f"radius{dr:+d}"] = clip(m18.blur(im, m18.disk(t + dr)))
        S["kernel_gauss"] = clip(gauss_like(t)(im))
        S["kernel_ring"] = clip(m18.blur(im, ring(t)))
        S["offset"] = tg + np.array([8, -5, 3], np.float32)
        bm = base[m].mean(0)
        S["real_base"] = base
        S["real_base_contrast0.5"] = clip(bm + 0.5 * (base - bm))
        sc = {n: dict(old=Q.aff_mae(y, tg, m), new=Q.off_mae(y, tg, m), shape=Q.shape_mae(y, tg, m),
                      contrast=Q.contrast(y, tg, m)) for n, y in S.items()}
        table[f"{p}@{t:g}"] = sc
        new = {n: v["new"] for n, v in sc.items()}
        key = f"{p}@{t:g}"
        checks += [
            (key, "contrast0.5 worse than target", new["contrast0.5"] > new["target"] + 0.5),
            (key, "contrast0.75 worse than target", new["contrast0.75"] > new["target"] + 0.25),
            (key, "real contrast0.5 not better", new["real_base_contrast0.5"] >= new["real_base"]),
            (key, "texture sigma2 worse", new["texture_sigma2"] > new["target"]),
            (key, "texture sigma4 worse than sigma2", new["texture_sigma4"] > new["texture_sigma2"]),
            (key, "shift monotone", new["target"] < new["shift2"] < new["shift4"]),
            (key, "radius monotone", all(new[f"radius{s}{a}"] < new[f"radius{s}{a + 1}"]
                                         for s in "+-" for a in (1, 2))),
            (key, "gaussian kernel worse", new["kernel_gauss"] > new["target"]),
            (key, "ring kernel worse", new["kernel_ring"] > new["target"]),
            (key, "offset invariant", abs(new["offset"] - new["target"]) < 1e-3),
        ]
fails = [c for c in checks if not c[2]]
old_reward = {k: dict(old_real=v["real_base"]["old"], old_real_c05=v["real_base_contrast0.5"]["old"],
                      new_real=v["real_base"]["new"], new_real_c05=v["real_base_contrast0.5"]["new"])
              for k, v in table.items()}
res = dict(passed=not fails, n_checks=len(checks), fails=fails, old_score_contrast_reward=old_reward, table=table)
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/metric_validation.json", "w"), indent=1)
print("VALIDATION", "PASS" if not fails else "FAIL", f"{len(checks) - len(fails)}/{len(checks)}")
for f in fails:
    print("  fail:", f)
print("old vs new on BokehDiff's baseline, contrast halved:")
for k, v in old_reward.items():
    print(f"  {k:18s} old {v['old_real']:.2f} -> {v['old_real_c05']:.2f} | new {v['new_real']:.2f} -> {v['new_real_c05']:.2f}")
