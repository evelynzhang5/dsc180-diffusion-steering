"""Round 38 D2b (post hoc, CPU): is the remaining error partly the reference's colour space? BokehDiff's outputs vs a
disk blur done in sRGB (rounds 18-37's reference) and in linear light (sRGB -> linear, blur, -> sRGB), each at its own
best-fitting radius (0.5 px grid), on round 36's calibrated-request outputs. Revised error (offset-only MAE).
usage: python linear_check.py <R36_OUT> <results dir>"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D, measure18 as m18, metrics34 as Q, patches as PT, scenes36 as SC  # noqa: E401,E402

to_lin = lambda s: np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
to_srgb = lambda l: np.where(l <= 0.0031308, 12.92 * l, 1.055 * np.clip(l, 0, None) ** (1 / 2.4) - 0.055)
R36, RES = sys.argv[1], sys.argv[2]
GRID = np.arange(0, 24.01, 0.5)
out = []
for split in ("test", "dev"):
    for sc in SC.plan(split):
        y = np.asarray(Image.open(f"{R36}/img/{sc['name']}_a0.png")).astype(np.float32)
        src = sc["img"].astype(np.float32)
        lin = to_lin(src / 255.0)
        st_s = [m18.blur(src, m18.disk(r)) for r in GRID]
        st_l = [np.clip(to_srgb(m18.blur(lin, m18.disk(r))), 0, 1) * 255 for r in GRID]
        for k in PT.FREE:
            m = D.interior(sc["cid"], k)
            es = np.array([Q.off_mae(y, b, m) for b in st_s])
            el = np.array([Q.off_mae(y, b, m) for b in st_l])
            rs = SC.rstar(sc["gaps"][k])
            j = int(np.argmin(np.abs(GRID - rs)))
            hl = float((src[m].max(1) > 230).mean())                    # share of near-saturated source pixels
            out.append(dict(split=split, scene=sc["name"], cell=k, src=sc["where"][k].rsplit("_", 1)[0], r_star=rs,
                            srgb_best=float(es.min()), lin_best=float(el.min()), srgb_r=float(GRID[es.argmin()]),
                            lin_r=float(GRID[el.argmin()]), srgb_at_rstar=float(es[j]), lin_at_rstar=float(el[j]),
                            highlight_share=hl))
        print(sc["name"], flush=True)
json.dump(out, open(f"{RES}/linear_check.json", "w"), indent=1)
for split in ("test", "dev"):
    L = [r for r in out if r["split"] == split]
    g = lambda k: np.array([r[k] for r in L])
    ratio = g("lin_best") / g("srgb_best")
    print(f"== {split}: median best error sRGB {np.median(g('srgb_best')):.2f} linear {np.median(g('lin_best')):.2f}; "
          f"linear better in {int((ratio < 1).sum())}/{len(L)}; ratio 10/50/90 % {np.percentile(ratio, [10, 50, 90]).round(3)}; "
          f"corr(ratio, highlight share) {np.corrcoef(ratio, g('highlight_share'))[0, 1]:.2f}")
    for s in sorted({r["src"] for r in L}):
        Ls = [r for r in L if r["src"] == s]
        print(f"   {s:22s} sRGB {np.median([r['srgb_best'] for r in Ls]):.2f} (r {np.median([r['srgb_r'] - r['r_star'] for r in Ls]):+.1f}) "
              f"linear {np.median([r['lin_best'] for r in Ls]):.2f} (r {np.median([r['lin_r'] - r['r_star'] for r in Ls]):+.1f}) "
              f"highlights {np.median([r['highlight_share'] for r in Ls]):.3f}")
