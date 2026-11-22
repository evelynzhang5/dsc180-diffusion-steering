"""Round 39 step 0 (CPU): validate the boundary reference before using it.

Scene 512^2: foreground = textured disk (astronaut crop, antialiased edge, radius 140) over a background that is
  uniform grey | smooth gradient | texture (brick). Foreground at disparity 0.7, background at 0.3.
Focus settings: A fg in focus (d_f 0.7: sharp fg over blurred bg, bg radius 15.5 px)
                B bg in focus (d_f 0.3: blurred fg over sharp bg)
                C focus between (d_f 0.5: both blurred, fg in front, bg behind; radii 7.75 px each)
Renderers (lens39): raytrace (aperture-sampled thin lens, linear light; the reference), composite (premultiplied
analytic layering), naive (non-premultiplied). Also:
  hidden    raytrace with the background layer replaced by the all-in-focus composite image (the background behind the
            foreground is unknown to BokehDiff; this is the best a renderer without it could use)
  srgb      composite computed in sRGB instead of linear light
  kernel    the sampled aperture kernel vs measure18's analytic disk (single layer, linear light)
Reported: mean |difference| to raytrace (0-255) in bands of signed distance to the foreground edge.
usage: python validate39.py <results dir>
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lens39 as LN  # noqa: E402
import measure18 as m18  # noqa: E402

RES = sys.argv[1]
os.makedirs(RES, exist_ok=True)
S, NS = 512, 32
yy, xx = np.mgrid[:S, :S]
ss = 4                                                     # antialiased disk by 4x4 supersampling
sub = (np.arange(ss) + 0.5) / ss
alpha = np.zeros((S, S))
for oy in sub:
    for ox in sub:
        alpha += ((yy + oy - S / 2) ** 2 + (xx + ox - S / 2) ** 2 <= 140 ** 2)
alpha /= ss * ss
inside = alpha >= 0.5
dist = np.where(inside, distance_transform_edt(inside), -distance_transform_edt(~inside))   # + inside, - outside
BANDS = [(-1e9, -32), (-32, -16), (-16, -8), (-8, 0), (0, 8), (8, 16), (16, 32), (32, 1e9)]
bname = ["out>32", "out16-32", "out8-16", "out0-8", "in0-8", "in8-16", "in16-32", "in>32"]
F = data.astronaut()[0:S, 0:S].astype(np.float64)
BG = {"uniform": np.full((S, S, 3), 128.0),
      "gradient": np.stack([np.tile(np.linspace(40, 220, S), (S, 1))] * 3, -1) * np.array([1.0, 0.85, 0.7]),
      "texture": np.repeat(np.tile(data.brick(), (2, 2))[:S, :S, None], 3, 2).astype(np.float64)}
CONF = {"A_fg_focus": 0.7, "B_bg_focus": 0.3, "C_between": 0.5}


def bands(a, b):
    d = np.abs(a - b).mean(2)
    return {n: round(float(d[(dist > lo) & (dist <= hi)].mean()), 3) for n, (lo, hi) in zip(bname, BANDS)}


out = {}
# kernel check
lin = LN.to_lin(BG["texture"])
for r in (7.75, 15.5):
    kb = LN.to_srgb(LN.kernel_blur(lin, r, NS))
    an = LN.to_srgb(m18.blur(lin, m18.disk(r)))
    out[f"kernel_r{r}"] = float(np.abs(kb - an)[40:-40, 40:-40].mean())
print("kernel check (sampled vs analytic disk, interior):", {k: round(v, 3) for k, v in out.items()}, flush=True)
for bn, B in BG.items():
    for cn, d_f in CONF.items():
        t0 = time.time()
        fg = dict(F=F, alpha=alpha, d=0.7)
        bg = dict(F=B, alpha=np.ones((S, S)), d=0.3)
        ref = LN.raytrace([fg, bg], d_f, NS)
        comp = LN.composite(fg, bg, d_f, NS)
        nai = LN.naive(fg, bg, d_f, NS)
        allin = alpha[..., None] * F + (1 - alpha[..., None]) * B                     # BokehDiff's input (sRGB mix)
        hid = LN.raytrace([fg, dict(F=allin, alpha=np.ones((S, S)), d=0.3)], d_f, NS)
        cs = LN.R
        srgb = (lambda cf, cb: LN.kernel_blur(F * alpha[..., None], cf, NS) +
                (1 - LN.kernel_blur(alpha[..., None], cf, NS)) * LN.kernel_blur(B, cb, NS))(cs * (0.7 - d_f), cs * (0.3 - d_f))
        out[f"{bn}/{cn}"] = dict(composite=bands(comp, ref), naive=bands(nai, ref), hidden=bands(hid, ref),
                                 srgb_composite=bands(srgb, ref))
        np.save(f"{RES}/ref_{bn}_{cn}.npy", ref.astype(np.float32))
        print(f"{bn:8s} {cn:10s} ({time.time() - t0:.0f}s)", flush=True)
        for k, v in out[f"{bn}/{cn}"].items():
            print(f"   {k:15s}", v, flush=True)
json.dump(out, open(f"{RES}/validate39.json", "w"), indent=1)
