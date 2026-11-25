"""Round 46 (CPU): build the controlled transparency composites. FROZEN before any round 46 render.

Linear-light compositing model: I = alpha * F + (1 - alpha) * B, inside a pane (rounded rectangle) in front of a
background image B; outside the pane alpha = 0.
  B   6 independently generated, all-in-focus backgrounds (round 45 fresh scenes, chosen for full-frame texture):
      f_fur_0, f_hair_3, f_highlights_0, f_highlights_2, f_thin_1, f_hair_5. These are the independent units.
  F, alpha   4 surface types, the same pane geometry for every background:
      etched    curved strokes, alpha 0.85 on strokes, 0.06 elsewhere; F near-white frost (0.85)
      grid      2 px lines every 24 px, alpha 0.70 on lines, 0.06 elsewhere; F light grey (0.80)
      specular  soft diagonal streaks, alpha up to 0.60 (smooth); F white (1.0)
      tinted    uniform tint alpha 0.35 modulated +-0.10 by fine scratch texture, rim (6 px) alpha 0.90;
                F blue tint (0.20, 0.45, 0.80) with white rim
  Regions (inside the pane): surface-dominant alpha >= 0.8, transmission-dominant alpha <= 0.2, overlap in between;
  a region with < 2,000 px (768 scale) is marked unavailable.
Depth, in BokehDiff's processing convention (training convention after the flip: larger = nearer, as bd40):
  d_B = B's own prepared disparity (bd40 preprocessing: min-max, 3x3 erode/dilate clip, flip), rescaled to [0.05, 0.70];
  pane depth = focus f = 0.95 (nearer than everything in B). K: round 45's monotone rule for 12 px with
  gap = median |d_B - f| over the image (approximate: a different normalisation from the rule's fit).
  The SAME f, K, normalisation, resolution and seed are used for every render of a background, including the reference.
Control composite: flat grey background (linear 0.5, d_B = 0.3 everywhere) with the tinted surface (flat-colour control).
usage (repo root): python quarter2/feature46/compose46.py <round 45 data> <round 46 data>
"""
import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sw
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "confirm45"))
import fit45  # noqa: E402

SRC, DST = sys.argv[1], sys.argv[2]
BGS = ["f_fur_0", "f_hair_3", "f_highlights_0", "f_highlights_2", "f_thin_1", "f_hair_5"]
TYPES = ["etched", "grid", "specular", "tinted"]
N = 768
PANE = (200, 150, 568, 620)          # x0, y0, x1, y1 at 768
F_FOCUS, D_LO, D_HI = 0.95, 0.05, 0.70
RULE = json.load(open(os.path.join(HERE, "..", "confirm45", "results", "rule45.json")))
for d in ["input", "alpha", "surf", "maps"]:
    os.makedirs(f"{DST}/{d}", exist_ok=True)


def to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _mm(d, fn):
    return fn(sw(np.pad(d, 1, mode="edge"), (3, 3)), axis=(-2, -1))


def prepared(i):
    d = np.load(f"{SRC}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = _mm(_mm(d, np.min), np.max)
    d = 1 - np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((N, N), resample=Image.BILINEAR))
    return D_LO + (d - d.min()) / (d.max() - d.min()) * (D_HI - D_LO)


def pane_mask():
    m = Image.new("L", (N, N), 0)
    ImageDraw.Draw(m).rounded_rectangle(PANE, radius=40, fill=255)
    return np.asarray(m) / 255.0


def surface(kind, pane, rng):
    din = pane.copy()
    a = np.zeros((N, N))
    F = np.ones((N, N, 3))
    if kind == "etched":
        im = Image.new("L", (N, N), 0)
        dr = ImageDraw.Draw(im)
        for _ in range(14):
            pts = [(rng.uniform(PANE[0], PANE[2]), rng.uniform(PANE[1], PANE[3])) for _ in range(4)]
            t = np.linspace(0, 1, 60)[:, None]
            p = np.array(pts)
            curve = ((1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3])
            dr.line([tuple(q) for q in curve], fill=255, width=4)
        s = np.asarray(im.filter(ImageFilter.GaussianBlur(0.6))) / 255.0
        a = 0.06 + (0.85 - 0.06) * s
        F *= 0.85
    elif kind == "grid":
        s = np.zeros((N, N))
        for k in range(PANE[0], PANE[2], 24):
            s[:, k:k + 2] = 1
        for k in range(PANE[1], PANE[3], 24):
            s[k:k + 2, :] = 1
        a = 0.06 + (0.70 - 0.06) * s
        F *= 0.80
    elif kind == "specular":
        yy, xx = np.mgrid[0:N, 0:N]
        u = (xx + yy) / np.sqrt(2)
        s = np.zeros((N, N))
        for c, w in [(560, 18), (610, 8), (700, 30)]:
            s = np.maximum(s, np.exp(-((u - c) / w) ** 2))
        a = 0.60 * s
        F *= 1.0
    elif kind == "tinted":
        tex = rng.normal(0, 1, (N // 2, N // 2))
        tex = np.asarray(Image.fromarray(tex.astype(np.float32)).resize((N, N), resample=Image.BILINEAR))
        tex = np.clip(tex / 3, -1, 1)
        im = Image.new("L", (N, N), 0)
        ImageDraw.Draw(im).rounded_rectangle(PANE, radius=40, outline=255, width=6)
        rim = np.asarray(im) / 255.0
        a = np.clip(0.35 + 0.10 * tex, 0.2 + 1e-3, 0.8 - 1e-3)
        a = np.where(rim > 0.5, 0.90, a)
        F = np.where(rim[..., None] > 0.5, 1.0, np.array([0.20, 0.45, 0.80])[None, None, :] * np.ones((N, N, 1)))
    return np.clip(a * din, 0, 1), F


pane = pane_mask()
meta = dict(bgs=BGS, types=TYPES, pane=PANE, focus=F_FOCUS, depth_range=[D_LO, D_HI], composites={}, backgrounds={})
rng = np.random.default_rng(46)
surfs = {k: surface(k, pane, rng) for k in TYPES}
for bg in BGS + ["flat"]:
    if bg == "flat":
        B = np.full((N, N, 3), 0.5)
        dB = np.full((N, N), 0.30)
    else:
        B = to_lin(np.asarray(Image.open(f"{SRC}/input/{bg}.png").convert("RGB").resize((N, N), Image.LANCZOS)) / 255.0)
        dB = prepared(bg)
    gap = float(np.median(np.abs(dB - F_FOCUS)))
    K, info = fit45.plan_K(RULE, 12.0, gap)
    np.save(f"{DST}/maps/{bg}_dB.npy", dB.astype(np.float32))
    Image.fromarray((to_srgb(B) * 255).round().astype(np.uint8)).save(f"{DST}/input/{bg}__bgonly.png")
    meta["backgrounds"][bg] = dict(K=K, gap=gap, rule_info=info)
    for kind in (TYPES if bg != "flat" else ["tinted"]):
        a, F = surfs[kind]
        I = a[..., None] * F + (1 - a[..., None]) * B
        c = f"{bg}__{kind}"
        Image.fromarray((to_srgb(I) * 255).round().astype(np.uint8)).save(f"{DST}/input/{c}.png")
        np.save(f"{DST}/alpha/{c}.npy", a.astype(np.float32))
        np.save(f"{DST}/surf/{c}.npy", F.astype(np.float16))
        inside = pane > 0.5
        reg = dict(surface=int((inside & (a >= 0.8)).sum()), transmission=int((inside & (a <= 0.2)).sum()),
                   overlap=int((inside & (a > 0.2) & (a < 0.8)).sum()))
        meta["composites"][c] = dict(bg=bg, type=kind, regions_px=reg, control=bg == "flat")
        print(c, f"K {K:.1f}", reg, flush=True)
np.save(f"{DST}/maps/pane.npy", pane.astype(np.float32))
json.dump(meta, open(f"{DST}/compose46.json", "w"), indent=1)
