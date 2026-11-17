"""Round 25 analysis. Usage: python analyze25.py <R25_OUT> <results dir>"""
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
GATES = ("G0", "G1")
MILD, LARGE = [3, 4, 5, 6], [8, 11.625, 15.5]

val = M.validate(D)
gate = dict(estimator=val["passed"], reimpl_matches_original=meta["check_max_abs_diff"] <= 1.0,
            repeat=meta["repeat_max_abs_diff"] <= 1.0, outputs_ok=all(c["finite"] and not c["black"] for c in conds))
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
                rows.append(dict(stage=c["stage"], gate=c["gate"], target=c["target"], r_req=c["r_req"], patch=where[k], cell=k,
                                 r_hat=f["v"], at_cap=f["at_cap"], resid=f["mse"] / float(yv.var(0).mean() + 1e-9),
                                 status=c.get("status", "raw")))
            rows.append(dict(stage=c["stage"], gate=c["gate"], target=c["target"], patch="focus", cell=4,
                             mae_focus=float(np.abs(y - im)[ints[4]].mean())))
    print("measured group", gi, flush=True)

B = [r for r in rows if r["cell"] != 4]
F = [r for r in rows if r["cell"] == 4]
# raw response curves (median over patches and positions)
curve = {g: [(t, med([r["r_hat"] for r in B if r["stage"] == "raw" and r["gate"] == g and r["target"] == t]))
             for t in sorted({r["target"] for r in B if r["stage"] == "raw"})] for g in GATES}
def window(cv):
    """request range over which the raw median goes from 2 to 5 px (linear interpolation); None if not spanned"""
    xs, ys = np.array([a for a, _ in cv]), np.array([b for _, b in cv])
    def cross(v):
        j = next((i for i in range(1, len(ys)) if ys[i] >= v and ys[i - 1] < v), None)
        return None if j is None else float(np.interp(v, [ys[j - 1], ys[j]], [xs[j - 1], xs[j]]))
    a, b = cross(2.0), cross(5.0)
    return dict(req_at_2px=a, req_at_5px=b, width=None if a is None or b is None else b - a)
steps = {g: float(max(abs(curve[g][i + 1][1] - curve[g][i][1]) for i in range(len(curve[g]) - 1) if curve[g][i + 1][0] <= 8)) for g in GATES}
raw_cv = {g: {f"{t:g}": float(np.std(v := [np.mean([r["r_hat"] for r in B if r["stage"] == "raw" and r["gate"] == g and r["target"] == t and r["patch"] == p])
                                            for p in {r["patch"] for r in B}]) / np.mean(v)) for t, _ in curve[g]} for g in GATES}
# control on held-out patches (corrected runs)
C = [r for r in B if r["stage"] == "corrected"]
within1 = lambda r: (not r["at_cap"]) and abs(r["r_hat"] - r["target"]) <= 1.0
agree = lambda r: (not r["at_cap"]) and abs(r["r_hat"] - r["target"]) <= M.TOL(r["target"])
mild_acc = {g: float(np.mean([within1(r) for r in C if r["gate"] == g and r["target"] in MILD])) for g in GATES}
large_acc = {g: float(np.mean([agree(r) for r in C if r["gate"] == g and r["target"] in LARGE])) for g in GATES}
by_t = {g: {f"{t:g}": dict(within1=float(np.mean([within1(r) for r in C if r["gate"] == g and r["target"] == t])),
                          agree=float(np.mean([agree(r) for r in C if r["gate"] == g and r["target"] == t])),
                          median_r=med([r["r_hat"] for r in C if r["gate"] == g and r["target"] == t]),
                          request=med([r["r_req"] for r in C if r["gate"] == g and r["target"] == t]))
            for t in sorted({r["target"] for r in C})} for g in GATES}
cor_cv = {g: med([np.std(v := [np.mean([r["r_hat"] for r in C if r["gate"] == g and r["target"] == t and r["patch"] == p])
                                for p in {r["patch"] for r in C}]) / np.mean(v) for t in MILD]) for g in GATES}
focus = {g: med([r["mae_focus"] for r in F if r["gate"] == g]) for g in GATES}
resid = {g: med([r["resid"] for r in C if r["gate"] == g and not r["at_cap"]]) for g in GATES}

W = (mild_acc["G1"] >= mild_acc["G0"] + 0.10 and large_acc["G1"] >= large_acc["G0"] - 0.05 and cor_cv["G1"] <= 0.8 * cor_cv["G0"]
     and focus["G1"] - focus["G0"] <= 0.5 and resid["G1"] <= 1.2 * resid["G0"])
if W:
    outcome = "W: the cell-transfer gate improves mild-blur control and consistency without hurting larger blur or quality"
elif mild_acc["G1"] >= mild_acc["G0"] + 0.10:
    outcome = "P: mild-blur control improves by >= 10 points, but another condition fails (see breakdown)"
else:
    outcome = "N: no meaningful improvement in mild-blur control (< 10 points)"
summary = dict(gate=gate, outcome=outcome, raw_curve=curve, window=  {g: window(curve[g]) for g in GATES}, max_step_up_to_8px=steps,
               raw_patch_cv=raw_cv, mild_within1=mild_acc, large_agree=large_acc, by_target=by_t, corrected_mild_patch_cv=cor_cv,
               focus_mae_in=focus, fit_resid=resid, calib=calib["C1"], seconds=meta["seconds"], check_diff=meta["check_max_abs_diff"])
json.dump(summary, open(f"{res}/gate.json", "w"), indent=1)
json.dump(rows, open(f"{res}/cells.json", "w"))
print("gate", gate)
print("OUTCOME", outcome)
for g in GATES:
    print(g, "raw curve", [(a, round(b, 2)) for a, b in curve[g]], "window", summary["window"][g], "max step <=8px %.2f" % steps[g])
    print("   raw patch CV", {k: round(v, 3) for k, v in raw_cv[g].items()})
    print("   corrected:", {k: {a: round(b, 2) for a, b in v.items()} for k, v in by_t[g].items()})
print("mild within 1 px", mild_acc, "| large agree", large_acc, "| corrected mild patch CV", cor_cv, "| focus", focus, "| resid", resid, "| s/img", meta["seconds"])

# sheet: crops of one swept cell, raw requests 1-12 px, G0 vs G1 (2 x 9) + input
idx = [1, 2, 3, 4, 5, 6, 8, 10, 12]
img, cid, where = PT.layout(groups("test")[0], 0)
y0, x0 = 30, 200
crop = lambda a: Image.fromarray(np.asarray(a)[y0:y0 + 110, x0:x0 + 110])
T = 110
sheet = Image.new("RGB", ((len(idx) + 1) * (T + 4), 2 * (T + 16)), "white")
dr = ImageDraw.Draw(sheet)
sheet.paste(crop(img), (0, 14))
dr.text((2, 1), "input", fill="black")
for i, g in enumerate(GATES):
    for j, r in enumerate(idx):
        sheet.paste(crop(Image.open(f"{out}/img/raw_g0_r0_{g}_req{r}.png")), ((j + 1) * (T + 4), i * (T + 16) + 14))
        dr.text(((j + 1) * (T + 4) + 2, i * (T + 16) + 1), f"{g} {r}px", fill="black")
sheet.save(f"{res}/gate_sheet.png")
print("sheet written")
