"""Round 39 post hoc (CPU, saved outputs): what is the change to a sharp foreground when only the background's blur is
requested (setting A)?

Compared: BokehDiff's calibrated setting-A output (global x1.0) vs its zero-request output k0 of the same scene. The
two differ only in the background's blur request, so any foreground change is caused by that request.
Foreground pixels by distance inside the edge: 2-8, 8-16, 16-32, 32-64, >64 px.
  shift      per-channel median difference (uniform colour/brightness shift), reported as mean |.| over channels and
             as signed luminance
  texture    mean |difference| after removing the whole-foreground per-channel median shift (non-uniform change)
  sharpness  high-frequency energy ratio: E|y - G1*y|^2 / E|k0 - G1*k0|^2 (G1 = Gaussian sigma 1 px); < 1 = softer
  contrast   per-channel std ratio y / k0
Also: dose response of the foreground change (fg_focus, vs k0) over the global request multiplier 0.6-1.4.
usage: python fg_change.py <R39_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes39 as S39  # noqa: E402

OUT, RES = sys.argv[1], sys.argv[2]
LUM = np.array([0.2126, 0.7152, 0.0722])
BANDS = [(2, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]
hf = lambda x: x - np.stack([gaussian_filter(x[..., c], 1.0) for c in range(3)], -1)
rows = []
for sc in S39.plan():
    b = S39.build(sc)
    k0 = np.asarray(Image.open(f"{OUT}/img/{sc['name']}_k0.png")).astype(np.float64)
    fgm = b["dist"] > 2
    for lv in S39.LEVELS:
        y = np.asarray(Image.open(f"{OUT}/img/{sc['name']}_A_{lv}_global_g1.00_e1.00.png")).astype(np.float64)
        d = y - k0
        off = np.median(d[fgm], 0)
        hy, hk = hf(y), hf(k0)
        r = dict(scene=sc["name"], tex=sc["tex"], bg=sc["bg_kind"], level=lv,
                 whole=dict(shift=float(np.abs(off).mean()), shift_lum=float(off @ LUM),
                            texture=float(np.abs(d[fgm] - off).mean()), raw=float(np.abs(d[fgm]).mean())), bands={})
        for lo, hi in BANDS:
            m = (b["dist"] > lo) & (b["dist"] <= hi)
            if m.sum() < 50:
                continue
            ob = np.median(d[m], 0)
            r["bands"][f"{lo}-{hi if hi < 1e8 else 'inf'}"] = dict(
                n=int(m.sum()), shift=float(np.abs(ob).mean()), shift_lum=float(ob @ LUM),
                texture=float(np.abs(d[m] - off).mean()), texture_local=float(np.abs(d[m] - ob).mean()),
                sharp=float((hy[m] ** 2).mean() / max((hk[m] ** 2).mean(), 1e-9)),
                contrast=float(np.mean(y[m].std(0) / (k0[m].std(0) + 1e-6))))
        rows.append(r)
C = json.load(open(f"{OUT}/conds.json"))["rows"]
dose = {}
for mg in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4):
    v = [x["fg_focus"] for x in C if x["setting"] == "A" and x["kind"] == "global" and abs(x["mg"] - mg) < 1e-9]
    dose[str(mg)] = float(np.median(v))
json.dump(dict(rows=rows, dose=dose), open(f"{RES}/fg_change.json", "w"), indent=1)

med = lambda xs: float(np.median(xs))
print("whole foreground, median over 48 (scene, level):",
      {k: round(med([r["whole"][k] for r in rows]), 2) for k in ("raw", "shift", "shift_lum", "texture")})
for t in S39.TEXTURES:
    R = [r for r in rows if r["tex"] == t]
    print(f"  {t:10s}", {k: round(med([r["whole"][k] for r in R]), 2) for k in ("raw", "shift", "shift_lum", "texture")})
print("by distance inside the edge (median):")
for bn in rows[0]["bands"]:
    R = [r["bands"][bn] for r in rows if bn in r["bands"]]
    print(f"  {bn:8s}", {k: round(med([x[k] for x in R]), 3) for k in ("shift", "shift_lum", "texture", "texture_local", "sharp", "contrast")})
print("L1 vs L2 whole texture change:", {lv: round(med([r["whole"]["texture"] for r in rows if r["level"] == lv]), 2) for lv in S39.LEVELS})
print("dose (median fg change vs k0 by global multiplier):", {k: round(v, 2) for k, v in dose.items()})
