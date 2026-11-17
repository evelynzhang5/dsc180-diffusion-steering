"""Round 23 analysis. Usage: python analyze23.py <R23_OUT> <results dir>"""
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

out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
conds = meta["conds"]
ld = lambda f: np.asarray(Image.open(os.path.join(out, f))).astype(np.float32)
CHANGE = lambda r0: max(1.0, 0.10 * r0)            # a cell "changes" if |delta r| exceeds this
VARIANTS = ["V2_flat", "V3_half_c0.5", "V4_half_c0.3", "V5_half_c0.7", "V6_max0.99"]

val = M.validate(D)
reps = [c for c in conds if c.get("kind") == "repeat"]
ref = {(c["scene"], c["run"]): c for c in conds if c["variant"] == "V1_true" and c["occ"] == "native" and c.get("kind") != "repeat"}
frozen_diff = {}
for c in conds:
    if c["occ"] == "frozen":
        frozen_diff[c["tag"]] = float(np.abs(ld(c["file"]) - ld(ref[(c["scene"], c["run"])]["file"])).max())
rep_same_inputs = all(r["input_disp_coc_hash"] == ref[(r["scene"], r["run"])]["input_disp_coc_hash"] and
                      [p["occ"] for p in r["pisa_calls"]] == [p["occ"] for p in ref[(r["scene"], r["run"])]["pisa_calls"]] for r in reps)
gate = dict(estimator=val["passed"], repeat_identical=all(r["repeat_max_abs_diff"] <= 1.0 for r in reps),
            repeat_inputs_identical=rep_same_inputs,
            frozen_equals_true=all(v <= 1.0 for v in frozen_diff.values()),
            outputs_ok=all(c["finite"] and not c["black"] for c in conds))
gate["passed"] = all(gate.values())

rows = []
for n in sorted({c["scene"] for c in conds}):
    s = D.make(n)
    img = s["image"].astype(np.float32)
    st = m18.stack(img, "disk")
    ints = {k: D.interior(s["cid"], k) for k in range(9)}
    Bs = {k: M.stack_at(st, ints[k]) for k in range(9)}
    for c in [c for c in conds if c["scene"] == n and c.get("kind") != "repeat"]:
        y = ld(c["file"])
        _, cl = D.cells(c["run"], c["perm"])
        gap = {x["cell"]: x["gap"] for x in cl}
        stats = {k: np.mean([s_["cells"][str(k)]["removed_by_occlusion"] if isinstance(next(iter(s_["cells"])), str) else s_["cells"][k]["removed_by_occlusion"] for s_ in c["stats"]]) for k in range(9)}
        leave = {k: np.mean([s_["cells"][str(k)]["leave"] if isinstance(next(iter(s_["cells"])), str) else s_["cells"][k]["leave"] for s_ in c["stats"]]) for k in range(9)}
        for k in range(9):
            f = M.fit_fast(y[ints[k]], Bs[k])
            rows.append(dict(scene=n, run=c["run"], variant=c["variant"], occ=c["occ"], cell=k, gap=gap[k],
                             r_nom=D.NOM * D.K * abs(gap[k]), r_hat=f["v"], at_cap=f["at_cap"],
                             removed_by_occlusion=float(stats[k]), leave=float(leave[k]),
                             mae_in=float(np.abs(y - img)[ints[k]].mean())))

idx = {(r["scene"], r["run"], r["variant"], r["occ"], r["cell"]): r for r in rows}
def effect(variant, occ):
    ch, d = [], []
    for r in rows:
        if r["variant"] != variant or r["occ"] != occ or r["cell"] == 4:
            continue
        r0 = idx[(r["scene"], r["run"], "V1_true", "native", r["cell"])]
        if r["at_cap"] or r0["at_cap"]:
            continue
        d.append((r["r_hat"] - r0["r_hat"]) / max(r0["r_hat"], 1))
        ch.append(abs(r["r_hat"] - r0["r_hat"]) > CHANGE(r0["r_hat"]))
    return dict(n=len(ch), changed_frac=float(np.mean(ch)), median_rel=float(np.median(d)),
                q10=float(np.percentile(d, 10)), q90=float(np.percentile(d, 90)), depth_effect=bool(np.mean(ch) >= 0.5))
eff = {v: effect(v, "native") for v in VARIANTS}
eff["V1_occlusion_zero"] = effect("V1_true", "zero")
outcome = ("Z2: the disparity map changes interior blur at a fixed blur map (through occlusion)" if any(eff[v]["depth_effect"] for v in VARIANTS)
           else "Z1: no depth effect on interior blur at a fixed blur map")
focus_eff = {v: float(np.median([idx[(s_, r_, v, "native", 4)]["mae_in"] - idx[(s_, r_, "V1_true", "native", 4)]["mae_in"]
                                 for s_ in ("S3", "S6") for r_ in ("R2", "R3")])) for v in VARIANTS}
occ_removed = {v: {o: float(np.median([r["removed_by_occlusion"] for r in rows if r["variant"] == v and r["occ"] == o and r["cell"] != 4]))
                   for o in ("native", "frozen", "zero") if any(r["variant"] == v and r["occ"] == o for r in rows)} for v in ["V1_true"] + VARIANTS}
anchor_front = [r for r in rows if r["variant"] in ("V1_true", "V6_max0.99") and r["occ"] == "native" and r["cell"] in (0, 8)]
nonfinite = sum(p["occ_nonfinite"] for c in conds for p in c["pisa_calls"])
summary = dict(gate=gate, frozen_max_diff=frozen_diff, outcome=outcome, effects=eff, focus_mae_in_change=focus_eff,
               occlusion_removed_median=occ_removed, occ_nonfinite_total=nonfinite,
               anchors_true_vs_max099=[(r["scene"], r["run"], r["cell"], r["variant"], round(r["r_hat"], 2), round(r["removed_by_occlusion"], 4)) for r in anchor_front],
               peak_mem_alloc_gb=meta["peak_mem_alloc_gb"])
json.dump(summary, open(f"{res}/isolation.json", "w"), indent=1)
json.dump(rows, open(f"{res}/cells.json", "w"))
print("gate", gate)
print("OUTCOME", outcome)
for k, v in eff.items():
    print(" ", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print("focus MAE change vs V1", {k: round(v, 2) for k, v in focus_eff.items()})
print("occlusion removed (median over cells)", {k: {a: round(b, 4) for a, b in v.items()} for k, v in occ_removed.items()})
print("nonfinite occlusion entries", nonfinite)
print("anchors", summary["anchors_true_vs_max099"])
for r in [r for r in rows if r["occ"] == "native" and r["scene"] == "S3" and r["run"] == "R2"]:
    print("S3 R2", r["variant"], r["cell"], "r_nom %.1f r_hat %.2f removed %.4f leave %.3f" % (r["r_nom"], r["r_hat"], r["removed_by_occlusion"], r["leave"]))

# sheet: S3 R2 and S6 R3: V1 true, V1 zero, V2..V6 native (7 each) + input (2) = 16
T = 160
cols = [("V1_true", "native"), ("V1_true", "zero")] + [(v, "native") for v in VARIANTS]
sheet = Image.new("RGB", ((len(cols) + 1) * (T + 4), 2 * (T + 16)), "white")
dr = ImageDraw.Draw(sheet)
for i, (n, r_) in enumerate([("S3", "R2"), ("S6", "R3")]):
    sheet.paste(Image.fromarray(D.make(n)["image"]).resize((T, T)), (0, i * (T + 16) + 14))
    dr.text((2, i * (T + 16) + 1), f"{n} input", fill="black")
    for j, (v, o) in enumerate(cols):
        sheet.paste(Image.open(f"{out}/img/{n}_{r_}P1_{v}_{o}.png").resize((T, T)), ((j + 1) * (T + 4), i * (T + 16) + 14))
        dr.text(((j + 1) * (T + 4) + 2, i * (T + 16) + 1), f"{r_} {v[:6]} {o}", fill="black")
sheet.save(f"{res}/isolation_sheet.png")
print("sheet written")
