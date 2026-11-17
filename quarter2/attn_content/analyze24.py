"""Round 24 analysis. Usage: python analyze24.py <R24_OUT> <results dir>"""
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
pix = {p["id"]: p for p in lib}
groups = lambda split: [[p for p in lib if p["split"] == split][0::2], [p for p in lib if p["split"] == split][1::2]]
ld = lambda f: np.asarray(Image.open(os.path.join(out, f))).astype(np.float32)
METHODS = ["A0-raw", "A1-raw", "A2-raw", "A0-C1", "A0-CP", "A1-C1", "A2-C1"]
med = lambda x: float(np.median(x))

# ---- protection 2: estimator on these patches (known disk and Gaussian blur, all cells blurred alike)
rng = np.random.default_rng(0)
est = dict(disk=[], gauss=[])
for split in ("dev", "test"):
    for grp in groups(split):
        img, cid, where = PT.layout(grp, 0)
        im = img.astype(np.float32)
        st = m18.stack(im, "disk")
        ints = {k: D.interior(cid, k) for k in PT.FREE}
        Bs = {k: M.stack_at(st, ints[k]) for k in PT.FREE}
        for r in (0, 6, 8, 12, 16, 20):
            y = m18.degrade(m18.blur(im, m18.disk(r)), rng)
            for k in PT.FREE:
                f = M.fit_fast(y[ints[k]], Bs[k])
                est["disk"].append(dict(patch=where[k], r_true=r, r_hat=f["v"], ok=abs(f["v"] - r) <= m18.VAL_TOL(r)))
        for s_ in (3, 5, 7, 9):
            y = m18.degrade(m18.blur(im, m18.gauss(s_)), rng)
            for k in PT.FREE:
                est["gauss"].append(dict(patch=where[k], sigma=s_, r_hat=M.fit_fast(y[ints[k]], Bs[k])["v"]))
gauss_cv = {s_: float(np.std(v := [e["r_hat"] for e in est["gauss"] if e["sigma"] == s_]) / np.mean(v)) for s_ in (3, 5, 7, 9)}
gate = dict(estimator_disk_all_patches=all(e["ok"] for e in est["disk"]), repeat=meta["repeat_max_abs_diff"] <= 1.0,
            outputs_ok=all(c["finite"] and not c["black"] for c in conds))
gate["passed"] = all(gate.values())
print("gate", gate, "| worst disk error %.2f px" % max(abs(e["r_hat"] - e["r_true"]) for e in est["disk"]),
      "| Gaussian disk-fit CV across patches", {k: round(v, 3) for k, v in gauss_cv.items()})

# ---- test measurements
rows = []
for gi, grp in enumerate(groups("test")):
    for rot in range(6):
        img, cid, where = PT.layout(grp, rot)
        im = img.astype(np.float32)
        st = m18.stack(im, "disk")
        ints = {k: D.interior(cid, k) for k in PT.FREE + [4]}
        Bs = {k: M.stack_at(st, ints[k]) for k in PT.FREE + [4]}
        del st
        for c in [c for c in conds if c.get("stage") == "test" and c["group"] == gi and c["rot"] == rot]:
            y = ld(c["file"])
            for k in PT.FREE:
                f = M.fit_fast(y[ints[k]], Bs[k])
                cc = c["cells"][str(k)]
                yv = y[ints[k]]
                rows.append(dict(method=c["method"], variant=c["variant"], target=c["target"], patch=cc["patch"], cell=k,
                                 source=pix[cc["patch"]]["source"], r_req=cc["r_req"], status=cc["status"], r_hat=f["v"],
                                 at_cap=f["at_cap"], fit_rel_resid=f["mse"] / float(yv.var(0).mean() + 1e-9)))
            f4 = float(np.abs(y - im)[ints[4]].mean())
            rows.append(dict(method=c["method"], variant=c["variant"], target=c["target"], patch="focus", cell=4,
                             mae_focus=f4, r_hat=M.fit_fast(y[ints[4]], Bs[4])["v"], at_cap=False))
    print("measured test group", gi, flush=True)
B = [r for r in rows if r["cell"] != 4]
agree = lambda r: (not r["at_cap"]) and abs(r["r_hat"] - r["target"]) <= M.TOL(r["target"])
acc = {m: float(np.mean([agree(r) for r in B if r["method"] == m])) for m in METHODS}
acc_t = {m: {f"{t:.1f}": float(np.mean([agree(r) for r in B if r["method"] == m and abs(r["target"] - t) < 1e-6]))
             for t in sorted({r["target"] for r in B})} for m in METHODS}
acc_src = {m: {s: float(np.mean([agree(r) for r in B if r["method"] == m and r["source"] == s])) for s in PT.TEST_SRC} for m in METHODS}

# consistency at a fixed request (raw): CV across patches of per-patch means over 6 positions; position share
cons = {}
for v in ("A0", "A1", "A2"):
    cons[v] = {}
    for t in sorted({r["target"] for r in B}):
        rr = [r for r in B if r["method"] == f"{v}-raw" and abs(r["target"] - t) < 1e-6 and not r["at_cap"]]
        pm = {p: np.mean([r["r_hat"] for r in rr if r["patch"] == p]) for p in {r["patch"] for r in rr}}
        qm = {k: np.mean([r["r_hat"] for r in rr if r["cell"] == k]) for k in PT.FREE}
        tot = np.var([r["r_hat"] for r in rr])
        cons[v][f"{t:.1f}"] = dict(mean=float(np.mean(list(pm.values()))), patch_cv=float(np.std(list(pm.values())) / np.mean(list(pm.values()))),
                                   position_cv=float(np.std(list(qm.values())) / np.mean(list(qm.values()))),
                                   patch_var_share=float(np.var(list(pm.values())) / tot), position_var_share=float(np.var(list(qm.values())) / tot),
                                   n=len(rr))
cv_red = {v: med([1 - cons[v][t]["patch_cv"] / cons["A0"][t]["patch_cv"] for t in cons["A0"]]) for v in ("A1", "A2")}

# quality
F = [r for r in rows if r["cell"] == 4]
focus = {m: med([r["mae_focus"] for r in F if r["method"] == m]) for m in METHODS}
resid = {m: med([r["fit_rel_resid"] for r in B if r["method"] == m and not r["at_cap"]]) for m in METHODS}
attn = {v: dict(leave=med([s_[c]["leave"] for cc in conds if cc.get("method") == f"{v}-raw" and cc["attn_stats"] for s_ in cc["attn_stats"] for c in s_]),
                entropy=med([s_[c]["entropy"] for cc in conds if cc.get("method") == f"{v}-raw" and cc["attn_stats"] for s_ in cc["attn_stats"] for c in s_]))
        for v in ("A0", "A1", "A2")}

# CP as a predictor on raw A0 test cells (prediction vs control, protection 3)
b = np.array(calib["CP"]["beta"])
zz = lambda pid: np.array([(pix[pid]["stats"][k] - calib["CP"]["zmu"][k]) / calib["CP"]["zsd"][k] for k in calib["CP"]["zkeys"]])
c1 = calib["C1"]["A0"]
pred = [r for r in B if r["method"] == "A0-raw" and not r["at_cap"]]
cp_err = med([abs(np.exp(b[0] + b[1] * np.log(r["r_req"]) + b[2:] @ zz(r["patch"])) - r["r_hat"]) / r["r_hat"] for r in pred])
c1_err = med([abs(np.interp(r["r_req"], c1["x"], c1["y"]) - r["r_hat"]) / r["r_hat"] for r in pred])

# ---- frozen decisions
measure_flag = med(list(gauss_cv.values())) >= 0.5 * med([cons["A0"][t]["patch_cv"] for t in cons["A0"]])
best_simple = max(acc["A0-C1"], acc["A0-CP"])
E1 = acc["A0-CP"] >= 0.80 and acc["A0-CP"] >= acc["A0-C1"] + 0.05
E2 = {v: bool(acc[f"{v}-C1"] >= best_simple + 0.05 and cv_red[v] >= 0.20 and focus[f"{v}-C1"] - focus["A0-C1"] <= 0.5
              and resid[f"{v}-C1"] <= 1.2 * resid["A0-C1"]) for v in ("A1", "A2")}
attn_changed = {v: bool(abs(attn[v]["entropy"] - attn["A0"]["entropy"]) / max(attn["A0"]["entropy"], 1e-9) >= 0.10 or
                        abs(attn[v]["leave"] - attn["A0"]["leave"]) / max(attn["A0"]["leave"], 1e-9) >= 0.10) for v in ("A1", "A2")}
E3 = {v: bool(attn_changed[v] and cv_red[v] < 0.20 and acc[f"{v}-C1"] < best_simple + 0.05) for v in ("A1", "A2")}
outcomes = []
if measure_flag:
    outcomes.append("M: known physical blur gives patch-dependent readings comparable to the model's spread; improve measurement first")
if E1:
    outcomes.append("E1: patch-statistics calibration reaches 80% and beats the radius curve by >= 5 points")
for v in ("A1", "A2"):
    if E2[v]:
        outcomes.append(f"E2 ({v}): attention intervention improves transfer beyond the best simple calibration, with quality kept")
    if E3[v]:
        outcomes.append(f"E3 ({v}): attention changes but consistency / transfer do not improve")
if not outcomes:
    outcomes.append("none of M / E1 / E2 / E3: reported as observed")

summary = dict(gate=gate, outcomes=outcomes, accuracy=acc, accuracy_by_target=acc_t, accuracy_by_source=acc_src,
               consistency=cons, patch_cv_reduction_vs_A0=cv_red, focus_mae_in=focus, fit_rel_resid=resid, attention=attn,
               attention_changed=attn_changed, estimator=dict(gauss_cv=gauss_cv, measure_flag=bool(measure_flag),
               worst_disk_err=float(max(abs(e["r_hat"] - e["r_true"]) for e in est["disk"]))),
               prediction_err_A0_raw=dict(CP=cp_err, C1=c1_err),
               req_status={m: {s: sum(1 for r in B if r["method"] == m and r["status"] == s) for s in ("ok", "raw", "clamped", "unreachable", "below_floor")} for m in METHODS},
               CP_beta=calib["CP"]["beta"], repeat_max_abs_diff=meta["repeat_max_abs_diff"], mean_seconds=meta["mean_seconds"])
json.dump(summary, open(f"{res}/attn_content.json", "w"), indent=1)
json.dump(rows, open(f"{res}/cells.json", "w"))
json.dump(est, open(f"{res}/estimator_check.json", "w"))
print("OUTCOMES", outcomes)
print("accuracy", {k: round(v, 3) for k, v in acc.items()})
for m in METHODS:
    print(" ", m, "by target", {k: round(v, 2) for k, v in acc_t[m].items()}, "by source", {k: round(v, 2) for k, v in acc_src[m].items()})
for v in cons:
    print("consistency", v, {t: {a: round(b_, 3) for a, b_ in d.items()} for t, d in cons[v].items()})
print("patch CV reduction vs A0", {k: round(v, 3) for k, v in cv_red.items()})
print("focus MAE", {k: round(v, 2) for k, v in focus.items()}, "fit resid", {k: round(v, 3) for k, v in resid.items()})
print("attention", attn, "changed", attn_changed)
print("prediction error on A0 raw test: CP %.3f, C1 %.3f" % (cp_err, c1_err), "CP beta", np.round(b, 3).tolist())
print("request status", summary["req_status"])

# sheet: two test layouts at target 11.6: input + 7 methods each (16 images)
T = 150
sheet = Image.new("RGB", (8 * (T + 4), 2 * (T + 16)), "white")
dr = ImageDraw.Draw(sheet)
for i, gi in enumerate((0, 1)):
    img, cid, where = PT.layout(groups("test")[gi], 0)
    sheet.paste(Image.fromarray(img).resize((T, T)), (0, i * (T + 16) + 14))
    dr.text((2, i * (T + 16) + 1), f"test g{gi} input", fill="black")
    for j, m in enumerate(METHODS):
        f = f"{out}/img/test_g{gi}_r0_{m}_t11.6.png" if "raw" not in m else f"{out}/img/test_g{gi}_r0_{m.split('-')[0]}_raw_t11.6.png"
        sheet.paste(Image.open(f).resize((T, T)), ((j + 1) * (T + 4), i * (T + 16) + 14))
        dr.text(((j + 1) * (T + 4) + 2, i * (T + 16) + 1), m, fill="black")
sheet.save(f"{res}/attn_content_sheet.png")
print("sheet written")
