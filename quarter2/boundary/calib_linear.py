"""Round 39 step 0b (CPU): recheck calibration and radius validity under the linear-light reference.

The radius estimator (round 18/19: disk radius 0-48 px step 0.5, per-channel gain + offset, parabolic refinement) fits
the output against a stack of disk blurs of the input. Until now that stack was blurred in sRGB. Here it is rebuilt in
linear light (sRGB -> linear -> disk blur -> sRGB) and everything that depends on it is recomputed:

  C1   round 25's development calibration (G0 gate, 12 dev patches x positions 0/3 x 11 request levels), refitted with
       each estimator. Check: the sRGB refit must reproduce round 25's frozen calib.json exactly.
  V36  round 36's 72 test cells (calibrated request, a = 0): r_hat with each estimator, within tolerance max(2, 20 %)
       of r*. Check: the sRGB count must reproduce round 36's 66/72.
Outputs results/calib_linear.json (both curves, per-row r_hat, per-cell validity). The linear curve is the one
frozen for step 1.
usage: python calib_linear.py <R25_OUT> <R36_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import lens39 as LN  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402
import scenes36 as SC  # noqa: E402

R25, R36, RES = sys.argv[1:4]
FLOOR = 1.5


def stacks(img):
    img = img.astype(np.float64)
    lin = LN.to_lin(img)
    s = [m18.blur(img, m18.disk(r)) for r in m18.R_GRID]
    l_ = [LN.to_srgb(m18.blur(lin, m18.disk(r))) for r in m18.R_GRID]
    return s, l_


out = dict(c1_rows=[], v36=[])
# ---- C1 refit on round 25's development sweep
conds = json.load(open(f"{R25}/conds.json"))["conds"]
lib = PT.library()
dev = [p for p in lib if p["split"] == "dev"]
for gi, grp in enumerate([dev[0::2], dev[1::2]]):
    for rot in (0, 3):
        img, cid, where = PT.layout(grp, rot)
        st_s, st_l = stacks(img)
        ints = {k: D.interior(cid, k) for k in PT.FREE}
        for c in [c for c in conds if c.get("stage") == "dev" and c["group"] == gi and c["rot"] == rot
                  and c["gate"] == "G0"]:
            y = np.asarray(Image.open(f"{R25}/{c['file']}")).astype(np.float64)
            for k in PT.FREE:
                fs = M.fit_fast(y[ints[k]], np.stack([b[ints[k]] for b in st_s]))
                fl = M.fit_fast(y[ints[k]], np.stack([b[ints[k]] for b in st_l]))
                out["c1_rows"].append(dict(r_req=c["r_req"], patch=where[k], srgb=fs["v"], srgb_cap=fs["at_cap"],
                                           lin=fl["v"], lin_cap=fl["at_cap"]))
        print("dev layout", gi, rot, flush=True)
rows = out["c1_rows"]
x = np.array([r["r_req"] for r in rows])
curves = {k: M.Curve(x, np.array([48.0 if r[k + "_cap"] else r[k] for r in rows]), FLOOR) for k in ("srgb", "lin")}
orig = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
out["curve_srgb"], out["curve_lin"], out["curve_r25"] = curves["srgb"].as_dict(), curves["lin"].as_dict(), orig
out["check_c1_reproduces_r25"] = float(np.abs(np.array(orig["y"]) - curves["srgb"].y).max()) if \
    np.array_equal(np.array(orig["x"]), curves["srgb"].x) else None
TARGETS = [3, 4, 5, 6, 8, 9.69, 11.625, 13.56, 15.5]
out["requests"] = {str(t): dict(srgb=curves["srgb"].inverse(t), lin=curves["lin"].inverse(t)) for t in TARGETS}
print("C1 check (sRGB refit vs round 25, max |dy|):", out["check_c1_reproduces_r25"])
print("curve x  ", curves["srgb"].x.tolist())
print("srgb y   ", np.round(curves["srgb"].y, 2).tolist())
print("linear y ", np.round(curves["lin"].y, 2).tolist())
for t, v in out["requests"].items():
    print(f"  target {t:>6s}: request sRGB {v['srgb'][0]:.2f}  linear {v['lin'][0]:.2f}  ({v['lin'][1]})")

# ---- round 36 test cells: validity under each estimator
for sc in SC.plan("test"):
    y = np.asarray(Image.open(f"{R36}/img/{sc['name']}_a0.png")).astype(np.float64)
    st_s, st_l = stacks(sc["img"])
    for k in PT.FREE:
        m = D.interior(sc["cid"], k)
        rs = SC.rstar(sc["gaps"][k])
        fs = M.fit_fast(y[m], np.stack([b[m] for b in st_s]))["v"]
        fl = M.fit_fast(y[m], np.stack([b[m] for b in st_l]))["v"]
        out["v36"].append(dict(scene=sc["name"], cell=k, r_star=rs, srgb=fs, lin=fl,
                               ok_srgb=bool(abs(fs - rs) <= m18.TOL(rs)), ok_lin=bool(abs(fl - rs) <= m18.TOL(rs))))
    print(sc["name"], flush=True)
V = out["v36"]
d = np.array([r["lin"] - r["srgb"] for r in V])
out["v36_summary"] = dict(n=len(V), ok_srgb=sum(r["ok_srgb"] for r in V), ok_lin=sum(r["ok_lin"] for r in V),
                          both=sum(r["ok_srgb"] and r["ok_lin"] for r in V),
                          lin_minus_srgb_median=float(np.median(d)), lin_minus_srgb_maxabs=float(np.abs(d).max()))
print("V36:", out["v36_summary"])
json.dump(out, open(f"{RES}/calib_linear.json", "w"), indent=1)
