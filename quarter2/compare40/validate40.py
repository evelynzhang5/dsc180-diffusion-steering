"""Round 40 stage 2 (CPU): validate the boundary measures (measures40.boundary) before using them on generated images.

Correct outputs: aperture-sampled thin-lens references (lens39, linear light, 256 samples) for an in-focus textured
foreground in front of a blurred textured background (round 39's setting A with textured backgrounds), at background
blur r = 7.75 and 15.5 px. Four scenes: round 39's three textured controls plus a thin-structure scene (a wheel: rim,
12 spokes 2 px wide, hub) over grass.
Injected defects, each applied to the correct output:
  rim     +10/255 on all channels in the 2 px just inside the contour                 -> rim should rise
  bleed   outer 0-8 px pulled toward the subject edge colour, weight 0.4 (1 - d/8)     -> bleed should rise
  halo    outer 0-12 px left sharp: replaced by the input, faded out over 12 px        -> halo_sharp should rise
  thin    inner 0-6 px band Gaussian-blurred (sigma 1.5 px)                             -> thin_loss should rise
Rule (frozen before running): a measure is VALIDATED if, in every scene and radius, its own defect moves it in the
expected direction by at least 2x the largest |value| it takes on the correct outputs, and its median shift under its
own defect is at least 2x its median |shift| under each other defect. Otherwise it stays exploratory.
Depth gating (added after review, before stage 3): all measures receive the scene's depth (subject layer = focus 1.0,
background 0.0), so halo_sharp uses only band pixels off the focus plane; the validation above is re-run with it.
Extra control "branch": the disk scene plus a sharp textured horizontal bar (smooth rocket or high-detail grass texture; half-width 3, 12 or 24 px) at the FOCUS
depth, outside the subject mask, crossing the outer band on both sides (like a branch the subject sits on). The thin-lens reference keeps it sharp, which is
correct. Expected: ungated halo_sharp > 0 (false positive), gated halo_sharp ~ 0. Reported, not part of the rule.
usage: python validate40.py <results dir>
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lens39 as LN  # noqa: E402
import measures40 as MS  # noqa: E402
import scenes39 as S39  # noqa: E402

RES = sys.argv[1]
NS, RADII = 16, [7.75, 15.5]
EXPECT = {"rim": "rim", "bleed": "bleed", "halo": "halo_sharp", "thin": "thin_loss"}
BGTEX = {"grass": (0.9, 1.0, 0.7), "brick": (1.0, 0.8, 0.7), "camera": (0.8, 0.9, 1.0)}
CASES = [("disk", "gravel", "grass", (256, 256)), ("ellipse", "astronaut", "brick", (240, 272)),
         ("rsquare", "chelsea", "camera", (272, 240)), ("wheel", "rocket", "grass", (256, 256))]


def bg_texture(name, tint):
    a = getattr(data, name)().astype(np.float64)
    a = a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
    return np.clip(a[:512, :512] * np.array(tint), 0, 255)


def wheel_alpha(cy, cx):
    yy, xx = np.mgrid[:512, :512].astype(np.float64)
    sub = (np.arange(4) + 0.5) / 4
    acc = 0
    for oy in sub:
        for ox in sub:
            y, x = yy + oy - cy, xx + ox - cx
            r = np.hypot(y, x)
            ring = (r >= 104) & (r <= 114)
            hub = r <= 14
            th = np.arctan2(y, x)
            spoke = np.zeros_like(r, bool)
            for k in range(12):
                a = k * np.pi / 6
                d = np.abs(-np.sin(a) * x + np.cos(a) * y)
                along = np.cos(a) * x + np.sin(a) * y
                spoke |= (d <= 1.0) & (along >= 0) & (r <= 106)
            acc = acc + (ring | hub | spoke)
    return acc / 16


def tiles_for(subj, img):
    _, dout = MS.geometry(subj)
    lum = img @ MS.LUM
    gy, gx = np.gradient(lum)
    g = np.hypot(gx, gy)
    c = []
    for y in range(0, 512 - 64 + 1, 32):
        for x in range(0, 512 - 64 + 1, 32):
            if dout[y:y + 64, x:x + 64].min() >= 24:
                c.append((float(g[y:y + 64, x:x + 64].mean()), y, x))
    c.sort(reverse=True)
    out = []
    for t, y, x in c:
        if all(abs(y - o["y"]) >= 64 or abs(x - o["x"]) >= 64 for o in out):
            out.append(dict(y=y, x=x))
        if len(out) == 8:
            break
    return out


def inject(kind, ref, x, subj):
    din, dout = MS.geometry(subj)
    y = ref.copy()
    if kind == "rim":
        y[(din > 0) & (din <= 2)] += 10
    elif kind == "bleed":
        c_s = x[(din > 0) & (din <= 8)].mean(0)
        w = np.clip(0.4 * (1 - dout / 8), 0, None) * ((dout > 0) & (dout <= 8))
        y = y + w[..., None] * (c_s - y)
    elif kind == "halo":
        w = np.clip(1 - dout / 12, 0, None) * (dout > 0)
        y = (1 - w[..., None]) * y + w[..., None] * x
    elif kind == "thin":
        b = np.stack([gaussian_filter(ref[..., c], 1.5) for c in range(3)], -1)
        m = (din > 0) & (din <= 6)
        y[m] = b[m]
    return np.clip(y, 0, 255)


rows = []
for shape, tex, bgn, ctr in CASES:
    a = wheel_alpha(*ctr) if shape == "wheel" else S39.alpha_of(shape, *ctr)
    F, B = S39.texture(tex), bg_texture(bgn, BGTEX[bgn])
    x = np.round(S39.to_srgb(a[..., None] * S39.to_lin(F) + (1 - a[..., None]) * S39.to_lin(B)))
    subj = a >= 0.5
    tiles = tiles_for(subj, x)
    for r in RADII:
        ref = LN.raytrace_np_radius([dict(F=F, alpha=a, d=1.0), dict(F=B, alpha=np.ones_like(a), d=0.0)], 1.0, NS, {1: r})
        dep = subj.astype(np.float64)
        row = dict(scene=f"{shape}/{tex}/{bgn}", r=r, correct=MS.boundary(x, ref, subj, tiles, dep, 1.0))
        for k in EXPECT:
            row[k] = MS.boundary(x, inject(k, ref, x, subj), subj, tiles, dep, 1.0)
        row["identity"] = MS.boundary(x, x, subj, tiles, dep, 1.0)
        rows.append(row)
        print(row["scene"], r, " | ".join(f"{k}: " + " ".join(f"{m}={row[k][m]:+.3f}" for m in EXPECT.values())
                                         for k in ["correct"] + list(EXPECT)), flush=True)

verdict = {}
for k, m in EXPECT.items():
    base = max(abs(r["correct"][m]) for r in rows)
    own = [r[k][m] - r["correct"][m] for r in rows]
    others = {o: float(np.median([abs(r[o][m] - r["correct"][m]) for r in rows])) for o in EXPECT if o != k}
    ok = all(s > 2 * base for s in own) and all(np.median(own) >= 2 * v for v in others.values())
    verdict[m] = dict(validated=bool(ok), max_abs_on_correct=base, own_shift_min=float(min(own)),
                      own_shift_median=float(np.median(own)), other_shift_median=others)
    print(f"{m:10s} {'VALIDATED' if ok else 'exploratory'}: |correct| max {base:.3f}; own shift min {min(own):+.3f} "
          f"median {np.median(own):+.3f}; other defects median |shift| " + " ".join(f"{o} {v:.3f}" for o, v in others.items()))
# extra control: in-focus branch outside the subject mask
a = S39.alpha_of("disk", 256, 256)
subj = a >= 0.5
_, dout = MS.geometry(subj)
yy, xx = np.mgrid[:512, :512]
F, Bt = S39.texture("gravel"), bg_texture("grass", BGTEX["grass"])
lin = S39.to_lin
branch = []
for btex, hw, r in [(t, hw, r) for t in ["rocket", "grass"] for hw in [3, 12, 24] for r in RADII]:
    Fb = S39.texture("rocket") if btex == "rocket" else bg_texture("grass", (0.7, 0.6, 0.5))
    bar = ((np.abs(yy - 256) <= hw) & (dout >= 1)).astype(np.float64)
    fg = np.maximum(a, bar)
    x = np.round(S39.to_srgb(a[..., None] * lin(F) + (1 - a[..., None]) * (bar[..., None] * lin(Fb)
                                                                           + (1 - bar[..., None]) * lin(Bt))))
    tiles = tiles_for(subj | (bar > 0), x)
    dep = (fg >= 0.5).astype(np.float64)
    ref = LN.raytrace_np_radius([dict(F=F, alpha=a, d=1.0), dict(F=Fb, alpha=bar, d=1.0),
                                 dict(F=Bt, alpha=np.ones_like(a), d=0.0)], 1.0, NS, {2: r})
    ung = MS.boundary(x, ref, subj, tiles)
    gat = MS.boundary(x, ref, subj, tiles, dep, 1.0)
    branch.append(dict(bar_texture=btex, half_width=hw, r=r, ungated=ung["halo_sharp"], gated=gat["halo_sharp"], gated_px=gat["halo_px"],
                       gated_kept=gat["halo_kept"]))
    print(f"branch control {btex} half-width {hw} r={r}: halo_sharp ungated {ung['halo_sharp']:+.3f}  gated {gat['halo_sharp']:+.3f} "
          f"(band kept {gat['halo_kept']:.2f})", flush=True)
json.dump(dict(rows=rows, verdict=verdict, branch_control=branch), open(f"{RES}/validate40.json", "w"), indent=1)
