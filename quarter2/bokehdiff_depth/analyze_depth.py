"""Round 18 analysis: estimator validation, gate, per-region blur measurements, frozen decision, contact sheet.
Usage: python analyze_depth.py <OUT dir with conds.json, convention.json and outputs> <results dir>"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure  # noqa: E402
import scenes  # noqa: E402

out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
conv = json.load(open(f"{out}/convention.json"))
conds = meta["conds"]

val = measure.validate(scenes)
json.dump(val, open(f"{res}/estimator_validation.json", "w"), indent=1)
print("estimator validation passed:", val["passed"], "worst disk error", round(val["worst_err"], 3), "px")

gate = dict(estimator=val["passed"], convention=conv["passed"], repeat=meta["repeat_max_abs_diff"] <= 1.0,
            outputs_ok=all(c["finite"] and not c["black"] for c in conds))
gate["passed"] = all(gate.values())
rows = measure.analyze(out, scenes, conds)
dec = measure.decide(rows) if gate["passed"] else dict(outcome="4: measurement or input conventions failed; repair the test")
# seed noise floor: seed 1 vs seed 0 at K 20, near focus
s0 = {(r["scene"], r["label"]): r for r in rows if r["K"] == 20 and r["focus"] == "near" and r["seed"] == 0}
seed_d = [abs(r["r_hat"] - s0[(r["scene"], r["label"])]["r_hat"]) for r in rows if r["seed"] == 1]
summary = dict(gate=gate, convention=conv, repeat_max_abs_diff=meta["repeat_max_abs_diff"],
               peak_mem_alloc_gb=meta["peak_mem_alloc_gb"], mean_seconds=meta["mean_seconds"],
               seed_r_diff_max=float(max(seed_d)), seed_r_diff_mean=float(np.mean(seed_d)), decision=dec, rows=rows)
json.dump(summary, open(f"{res}/depth.json", "w"), indent=1)

print("gate", gate)
print("decision", json.dumps(dec, indent=1))
print("seed 0 vs 1 radius difference: max %.2f mean %.2f px" % (max(seed_d), np.mean(seed_d)))
print("%-5s %-4s %-3s %-3s %-7s %-6s %5s %6s %6s %6s %6s %6s %6s" % (
    "scene", "foc", "K", "lab", "tex", "side", "z", "r_nom", "r_hat", "sigma", "r_vsK0", "shift", "maeK0"))
for r in rows:
    if r["seed"] != 0:
        continue
    print("%-5s %-4s %-3d %-3d %-7s %-6s %5.1f %6.1f %6.1f %6.1f %6.1f %6.2f %6.2f%s" % (
        r["scene"], r["focus"] or "-", r["K"], r["label"], r["texture"], r["side"], r["z"], r["r_nom"], r["r_hat"],
        r["sigma_hat"], r["r_hat_vs_k0"], max(abs(r["shift_dy"]), abs(r["shift_dx"])), r["mae_vs_k0"],
        " CAP" if r["at_cap"] else ""))

# contact sheet: rows = scenes; columns = input, K0, near K10/20/40, mid K10/20/40 (24 images)
T = 192
cols = [("input", None)] + [("K0", "sceneX_K0_none_s0.png")] + \
       [(f"{f} K{k}", f"sceneX_K{k}_{f}_s0.png") for f in ("near", "mid") for k in (10, 20, 40)]
sheet = Image.new("RGB", (len(cols) * (T + 4), 3 * (T + 18)), "white")
d = ImageDraw.Draw(sheet)
for i, n in enumerate(scenes.LAYOUTS):
    for j, (lab, f) in enumerate(cols):
        im = Image.fromarray(scenes.make(n)["image"]) if f is None else Image.open(os.path.join(out, f.replace("X", n)))
        sheet.paste(im.resize((T, T), Image.LANCZOS), (j * (T + 4), i * (T + 18) + 16))
        d.text((j * (T + 4) + 2, i * (T + 18) + 2), f"{n}: {lab}", fill="black")
sheet.save(f"{res}/depth_sheet.png")
print("sheet written")
