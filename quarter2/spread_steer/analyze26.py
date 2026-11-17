"""Round 26 analysis. Usage: python analyze26.py <R26_OUT> <results dir>"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402

out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
calib = json.load(open(f"{out}/calib.json"))
conds = meta["conds"]
lib = PT.library()
groups = lambda split: [[p for p in lib if p["split"] == split][0::2], [p for p in lib if p["split"] == split][1::2]]
med = lambda x: float(np.median(x))
MILD, LARGE = [3, 4, 5, 6], [8, 11.625, 15.5]
HARD = ["coins_0", "coins_1", "rocket_1"]           # round 24/25 high-contrast misses

val = M.validate(D)
st_err = [s["median_abs_err"] for c in conds if c.get("steer_stats") for s in c["steer_stats"]]
st_max = [s["max_abs_err"] for c in conds if c.get("steer_stats") for s in c["steer_stats"]]
gate = dict(estimator=val["passed"], baseline_reproduces_round25=meta["base_repro_max_diff"] <= 1.0,
            repeat=meta["repeat_max_abs_diff"] <= 1.0, outputs_ok=all(c["finite"] and not c["black"] for c in conds),
            steering_hits_target_share=bool(st_err and med(st_err) <= 0.005))
gate["passed"] = all(gate.values())

rows = []
for gi, grp in enumerate(groups("test")):
    for rot in sorted({c["rot"] for c in conds if c.get("stage") in ("raw", "corrected") and c["group"] == gi}):
        img, cid, where = PT.layout(grp, rot)
        im = img.astype(np.float32)
        st = m18.stack(im, "disk")
        ints = {k: D.interior(cid, k) for k in PT.FREE + [4]}
        Bs = {k: M.stack_at(st, ints[k]) for k in PT.FREE + [4]}
        del st
        for c in [c for c in conds if c.get("stage") in ("raw", "corrected") and c["group"] == gi and c["rot"] == rot]:
            y = np.asarray(Image.open(os.path.join(out, c["file"]))).astype(np.float32)
            for k in PT.FREE:
                f = M.fit_fast(y[ints[k]], Bs[k])
                yv = y[ints[k]]
                rows.append(dict(stage=c["stage"], method=c["method"], target=c.get("target"), m=c.get("m"), patch=where[k],
                                 r_hat=f["v"], at_cap=f["at_cap"], resid=f["mse"] / float(yv.var(0).mean() + 1e-9), cell=k))
            rows.append(dict(stage=c["stage"], method=c["method"], target=c.get("target"), m=c.get("m"), patch="focus", cell=4,
                             mae_focus=float(np.abs(y - im)[ints[4]].mean())))
    print("measured group", gi, flush=True)

B = [r for r in rows if r["cell"] != 4]
F = [r for r in rows if r["cell"] == 4]
C = [r for r in B if r["stage"] == "corrected"]
within1 = lambda r: (not r["at_cap"]) and abs(r["r_hat"] - r["target"]) <= 1.0
agree = lambda r: (not r["at_cap"]) and abs(r["r_hat"] - r["target"]) <= M.TOL(r["target"])
METH = ("base", "steer")
mild = {m: float(np.mean([within1(r) for r in C if r["method"] == m and r["target"] in MILD])) for m in METH}
large = {m: float(np.mean([agree(r) for r in C if r["method"] == m and r["target"] in LARGE])) for m in METH}
def cv_t(m, t):
    v = [np.mean([r["r_hat"] for r in C if r["method"] == m and r["target"] == t and r["patch"] == p]) for p in {r["patch"] for r in C}]
    return float(np.std(v) / np.mean(v))
cv = {m: {f"{t:g}": cv_t(m, t) for t in MILD + LARGE} for m in METH}
cv_med = {m: med(list(cv[m].values())) for m in METH}
hard = {m: med([abs(r["r_hat"] / r["target"] - 1) for r in C if r["method"] == m and r["patch"] in HARD]) for m in METH}
hard_ratio = {m: {p: med([r["r_hat"] / r["target"] for r in C if r["method"] == m and r["patch"] == p]) for p in sorted({r["patch"] for r in C})} for m in METH}
by_t = {m: {f"{t:g}": dict(within1=float(np.mean([within1(r) for r in C if r["method"] == m and r["target"] == t])),
                          agree=float(np.mean([agree(r) for r in C if r["method"] == m and r["target"] == t])),
                          median_r=med([r["r_hat"] for r in C if r["method"] == m and r["target"] == t])) for t in MILD + LARGE} for m in METH}
focus = {m: med([r["mae_focus"] for r in F if r["method"] == m and r["stage"] == "corrected"]) for m in METH}
resid = {m: med([r["resid"] for r in C if r["method"] == m and not r["at_cap"]]) for m in METH}
raw = [r for r in B if r["stage"] == "raw"]
raw_curve = [(m_, med([r["r_hat"] for r in raw if r["m"] == m_]),
              float(np.std(v := [np.mean([r["r_hat"] for r in raw if r["m"] == m_ and r["patch"] == p]) for p in {r["patch"] for r in raw}]) / np.mean(v)))
             for m_ in sorted({r["m"] for r in raw})]

a_ = mild["steer"] >= mild["base"] - 0.05 and large["steer"] >= large["base"] - 0.05
b_ = cv_med["steer"] <= 0.8 * cv_med["base"]
c_ = hard["steer"] <= 0.5 * hard["base"]
d_ = focus["steer"] - focus["base"] <= 0.5 and resid["steer"] <= 1.2 * resid["base"]
if a_ and b_ and c_ and d_:
    outcome = "S: spread steering matches calibration accuracy, cuts patch-to-patch spread >= 20 % and halves the high-contrast error, quality kept"
elif b_ or c_:
    outcome = "Partial: spread steering improves consistency or the high-contrast patches, but not every condition holds"
else:
    outcome = "F: spread steering does not improve consistency or the high-contrast patches; attention-level steering of this mechanism is closed"
summary = dict(gate=gate, outcome=outcome, conditions=dict(accuracy_not_worse=a_, cv_reduced_20=b_, hard_halved=c_, quality_kept=d_),
               mild_within1=mild, large_agree=large, cv_by_target=cv, cv_median=cv_med, hard_abs_err=hard, ratio_by_patch=hard_ratio,
               by_target=by_t, focus_mae_in=focus, fit_resid=resid, raw_steer_curve_m_r_cv=raw_curve,
               steer_share_err=dict(median=med(st_err) if st_err else None, max=max(st_max) if st_max else None),
               calib=dict(steer=calib["steer_curve"], base=calib["base_curve"]), seconds=meta["seconds"])
json.dump(summary, open(f"{res}/steer.json", "w"), indent=1)
json.dump(rows, open(f"{res}/cells.json", "w"))
print("gate", gate)
print("OUTCOME", outcome, summary["conditions"])
print("mild within 1 px", mild, "| large agree", large, "| CV median", cv_med, "| hard |ratio-1|", hard, "| focus", focus, "| resid", resid)
for m in METH:
    print(m, "by target", {k: {a: round(b, 2) for a, b in v.items()} for k, v in by_t[m].items()})
    print(m, "ratio by patch", {k: round(v, 2) for k, v in hard_ratio[m].items()})
print("raw steering (m, median r, patch CV)", [(a, round(b, 2), round(c, 3)) for a, b, c in raw_curve])
print("steering share error", summary["steer_share_err"], "| s/img", meta["seconds"])

# sheet: two test layouts, base vs steer at targets 4, 8, 15.5 (+ input): 2 x 7
T = 150
cols = [("base", 4), ("steer", 4), ("base", 8), ("steer", 8), ("base", 15.5), ("steer", 15.5)]
sheet = Image.new("RGB", ((len(cols) + 1) * (T + 4), 2 * (T + 16)), "white")
dr = ImageDraw.Draw(sheet)
for i, gi in enumerate((0, 1)):
    img, cid, where = PT.layout(groups("test")[gi], 0)
    sheet.paste(Image.fromarray(img).resize((T, T)), (0, i * (T + 16) + 14))
    dr.text((2, i * (T + 16) + 1), f"g{gi} input", fill="black")
    for j, (mth, t) in enumerate(cols):
        sheet.paste(Image.open(f"{out}/img/cor_g{gi}_r0_{mth}_t{t:.1f}.png").resize((T, T)), ((j + 1) * (T + 4), i * (T + 16) + 14))
        dr.text(((j + 1) * (T + 4) + 2, i * (T + 16) + 1), f"{mth} {t:g}px", fill="black")
sheet.save(f"{res}/steer_sheet.png")
print("sheet written")
