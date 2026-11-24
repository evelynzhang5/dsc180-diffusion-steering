"""Round 39 post hoc (CPU, references only): a positive control for setting A with TEXTURED backgrounds.

Setting A (foreground sharp at disparity 1, background behind and blurred) was degenerate in step 1: a uniform or
gradient background is unchanged by blur, so the correct output was the input. Before any model run with textured
backgrounds, check on the reference alone that
  (1) the reference changes visibly as the background blur increases (r = 0, 5.8, 7.75, 11.6, 15.5 px),
  (2) the delivered background blur is measurable: the linear-light radius estimator, fitted to the reference's
      background interior (> 32 px outside the edge), recovers r,
  (3) leaving the input unchanged fails: its error to the reference, in the background interior and the outer edge
      bands, is far above the reference's own sampling noise,
  (4) hidden background: the true-background reference vs an available-information reference whose background
      behind the foreground is filled from visible pixels only (multi-scale normalised convolution) differs mainly
      near the edge; reported by band.
Scenes: 3 foregrounds (disk/gravel, ellipse/astronaut, rounded square/chelsea) over 3 development textures
(grass, brick, camera; tiled, tinted), input composited in linear light. Ray tracer: 256 aperture samples (numpy).
usage: python a_control.py <results dir>
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lens39 as LN  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import scenes39 as S39  # noqa: E402

RES = sys.argv[1]
NS = 16
RADII = [0.0, 5.8, 7.75, 11.625, 15.5]
BGTEX = {"grass": (0.9, 1.0, 0.7), "brick": (1.0, 0.8, 0.7), "camera": (0.8, 0.9, 1.0)}
CASES = [("disk", "gravel", "grass", (256, 256)), ("ellipse", "astronaut", "brick", (240, 272)),
         ("rsquare", "chelsea", "camera", (272, 240))]


def bg_texture(name, tint):
    a = getattr(data, name)().astype(np.float64)
    a = a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
    return np.clip(a[:512, :512] * np.array(tint), 0, 255)


def fill_hidden(img, vis):
    """available-information background: hidden pixels filled from visible ones by normalised convolution,
    coarse to fine (in linear light)."""
    lin, w = S39.to_lin(img), vis.astype(np.float64)
    out = lin.copy()
    filled = vis.copy()
    for s in (64, 32, 16, 8, 4, 2):
        num = np.stack([gaussian_filter(lin[..., c] * w, s) for c in range(3)], -1)
        den = gaussian_filter(w, s)[..., None]
        est = num / np.maximum(den, 1e-6)
        out = np.where(filled[..., None], out, est)
    return S39.to_srgb(out)


def bands(a, c, bm):
    d = np.abs(a - c).mean(2)
    return {n: round(float(d[m].mean()), 3) for n, m in bm.items()}


out = []
for shape, tex, bgn, ctr in CASES:
    a = S39.alpha_of(shape, *ctr)
    F, B = S39.texture(tex), bg_texture(bgn, BGTEX[bgn])
    img = np.round(S39.to_srgb(a[..., None] * S39.to_lin(F) + (1 - a[..., None]) * S39.to_lin(B)))
    inside = a >= 0.5
    from scipy.ndimage import distance_transform_edt
    dist = np.where(inside, distance_transform_edt(inside), -distance_transform_edt(~inside))
    bm = S39.band_masks(dist)
    Bvis = fill_hidden(img, a <= 0)
    hid = bands(Bvis, B, {"hidden": inside})
    lin = S39.to_lin(img)
    st = [S39.to_srgb(m18.blur(lin, m18.disk(r)))[bm["out>32"]] for r in m18.R_GRID]
    Bst = np.stack(st)
    fg = dict(F=F, alpha=a, d=1.0)
    rows = []
    prev = None
    for r in RADII:
        bgl = dict(F=B, alpha=np.ones_like(a), d=0.0)
        ref = LN.raytrace_np_radius([fg, bgl], 1.0, NS, {1: r})
        refv = ref if r == 0 else LN.raytrace_np_radius([fg, dict(bgl, F=Bvis)], 1.0, NS, {1: r})
        rhat = float(M.fit_fast(ref[bm["out>32"]], Bst)["v"])
        rows.append(dict(r=r, input_vs_ref=bands(img, ref, bm), avail_vs_true=bands(refv, ref, bm), r_hat_ref=rhat,
                         step_from_prev=bands(ref, prev, bm) if prev is not None else None))
        prev = ref
        print(f"{shape}/{tex} over {bgn}: r {r:5.2f}  r_hat(ref bg) {rhat:5.2f}  input-vs-ref out>32 "
              f"{rows[-1]['input_vs_ref']['out>32']:.2f} out0-8 {rows[-1]['input_vs_ref']['out0-8']:.2f} "
              f"in0-8 {rows[-1]['input_vs_ref']['in0-8']:.2f} | avail-vs-true out0-8 "
              f"{rows[-1]['avail_vs_true']['out0-8']:.2f} out8-16 {rows[-1]['avail_vs_true']['out8-16']:.2f} "
              f"out>32 {rows[-1]['avail_vs_true']['out>32']:.2f}", flush=True)
    # sampling noise of the reference itself at r = 15.5: 256 vs 1024 samples
    bgl = dict(F=B, alpha=np.ones_like(a), d=0.0)
    noise = bands(LN.raytrace_np_radius([fg, bgl], 1.0, NS, {1: 15.5}), LN.raytrace_np_radius([fg, bgl], 1.0, 32, {1: 15.5}), bm)
    out.append(dict(shape=shape, fg=tex, bg=bgn, fill_error=hid, rows=rows, ref_noise_256_vs_1024=noise))
    print("   hidden-background fill error (mean |.|, 0-255):", hid, "| ref noise 256 vs 1024:",
          {k: noise[k] for k in ("out0-8", "in0-8", "out>32")}, flush=True)
json.dump(out, open(f"{RES}/a_control.json", "w"), indent=1)
