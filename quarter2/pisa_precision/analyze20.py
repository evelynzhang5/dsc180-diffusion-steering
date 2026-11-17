"""Round 20 analysis. Usage: python analyze20.py <R20_OUT> <results dir>"""
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
SC = meta["scene"]
conds = [c for c in meta["conds"] if c["kind"] in ("K0", "sweep")]
ld = lambda f: np.asarray(Image.open(os.path.join(out, f))).astype(np.float32)

val = M.validate(D)
gate = dict(estimator=val["passed"],
            wrapper_fp16_matches_round19=meta["gate_max_abs_diff"] is not None and meta["gate_max_abs_diff"] <= 1.0,
            repeat=meta["repeat_max_abs_diff"] <= 1.0,
            outputs_ok=all(c["finite"] and not c["black"] for c in meta["conds"]))
gate["passed"] = all(gate.values())

s = D.make(SC)
img = s["image"].astype(np.float32)
st = m18.stack(img, "disk")
ints = {k: D.interior(s["cid"], k) for k in range(9)}
Bs = {k: M.stack_at(st, ints[k]) for k in range(9)}
FREE = [3 * rc[0] + rc[1] for p in (1, 2, 3) for rc in D.PAIRS[p]]
ANCH = [3 * rc[0] + rc[1] for rc in D.PAIRS[0]]
k0img = {(c["mode"], c["seed"]): ld(c["file"]) for c in conds if c["kind"] == "K0"}
rows = []
for c in conds:
    y = ld(c["file"])
    for k in range(9):
        f = M.fit_fast(y[ints[k]], Bs[k])
        rows.append(dict(mode=c["mode"], seed=c["seed"], gap=c["gap"], r_nom=D.NOM * D.K * c["gap"] if k in FREE and c["kind"] == "sweep" else None,
                         cell=k, role="focus" if k == 4 else "anchor" if k in ANCH else "free", kind=c["kind"],
                         r_hat=f["v"], at_cap=f["at_cap"], mae_in=float(np.abs(y - img)[ints[k]].mean()),
                         mae_k0=float(np.abs(y - k0img[(c["mode"], c["seed"])])[ints[k]].mean())))

med = lambda xs: float(np.median(xs))
def free_med(mode, gap, seed=None):
    return med([r["r_hat"] for r in rows if r["mode"] == mode and r["kind"] == "sweep" and abs(r["gap"] - gap) < 1e-9
                and r["role"] == "free" and (seed is None or r["seed"] == seed)])

floor = {m: med([r["r_hat"] for r in rows if r["mode"] == m and r["kind"] == "K0" and r["role"] == "free"]) for m in ("fp16", "fp32")}
curve = {m: [(round(D.NOM * D.K * g, 2), free_med(m, g), free_med(m, g, 0), free_med(m, g, 1)) for g in meta["gaps"]] for m in ("fp16", "fp32")}
KEY = [0.075, 0.10, 0.125]                     # r_nom 2.9, 3.9, 4.8 px
raised = [all(free_med("fp32", g, sd) - free_med("fp16", g, sd) >= 1.0 for sd in (0, 1)) for g in KEY]
def covers(mode):
    vals = [free_med(mode, g) for g in meta["gaps"]]
    return {f"{a}-{a+1}": any(a <= v < a + 1 for v in vals) for a in (2, 3, 4)}
cov = {m: covers(m) for m in ("fp16", "fp32")}
A = sum(raised) >= 2
B = all(cov["fp32"].values())
outcome = ("P1: precision causes the gap (fp32 raises mild blur and covers 2-5 px)" if A and B else
           "P2: precision contributes (fp32 raises mild blur), but 2-5 px is not fully covered" if A else
           "P3: gap persists with fp32 PISA attention; look at the coarse representation / gate next")

focus = {m: {g: med([r["mae_in"] for r in rows if r["mode"] == m and r["role"] == "focus" and r["kind"] == "sweep" and abs(r["gap"] - g) < 1e-9])
             for g in meta["gaps"]} for m in ("fp16", "fp32")}
focus_diff = med([focus["fp32"][g] - focus["fp16"][g] for g in meta["gaps"]])
focus_r = {m: med([r["r_hat"] for r in rows if r["mode"] == m and r["role"] == "focus" and r["kind"] == "sweep"]) for m in ("fp16", "fp32")}
anchors = {m: med([r["r_hat"] for r in rows if r["mode"] == m and r["role"] == "anchor" and r["kind"] == "sweep"]) for m in ("fp16", "fp32")}
seed_noise = med([abs(free_med(m, g, 0) - free_med(m, g, 1)) for m in ("fp16", "fp32") for g in meta["gaps"]])
rt = {m: dict(seconds_per_image=med([c["seconds"] for c in conds if c["mode"] == m]),
              pisa_seconds_per_image=med([c["pisa_seconds"] for c in conds if c["mode"] == m]),
              pisa_calls_per_image=int(med([c["pisa_calls"] for c in conds if c["mode"] == m]))) for m in ("fp16", "fp32")}
shapes = sorted({(str(c["q_shape"]), c["q_dtype"], c["w_dtype"], c["coc_dtype"], c["out_dtype"], c["mode"], c["autocast"]) for c in meta["pisa_log"]})
spread = {}
for c in meta["pisa_log"]:
    for v, d in (c.get("spread") or {}).items():
        spread.setdefault(c["mode"], {}).setdefault(v, []).append((d["leave"], d["nonzero_targets"]))
spread = {m: {v: dict(leave=float(np.mean([x[0] for x in xs])), nonzero_targets=float(np.mean([x[1] for x in xs])))
              for v, xs in sorted(d.items(), key=lambda t: float(t[0]))} for m, d in spread.items()}

summary = dict(gate=gate, outcome=outcome, raised_by_1px_both_seeds=dict(zip(["2.9", "3.9", "4.8"], raised)),
               covers_bins=cov, floor=floor, curve_rnom_median_seed0_seed1=curve, focus_mae_vs_input=focus,
               focus_fp32_minus_fp16_median=focus_diff, focus_r_hat=focus_r, anchor_r_hat=anchors,
               seed_noise_median=seed_noise, runtime=rt, pisa_shapes_dtypes=shapes, attention_spread=spread,
               peak_mem_alloc_gb=meta["peak_mem_alloc_gb"], gate_max_abs_diff=meta["gate_max_abs_diff"],
               repeat_max_abs_diff=meta["repeat_max_abs_diff"], validation_worst=val["worst_err"])
json.dump(summary, open(f"{res}/precision.json", "w"), indent=1)
json.dump(rows, open(f"{res}/cells.json", "w"))
print("gate", gate)
print("OUTCOME", outcome)
print("raised", summary["raised_by_1px_both_seeds"], "covers", cov, "floor", floor, "seed noise %.2f" % seed_noise)
for m in ("fp16", "fp32"):
    print(m, "r_nom -> median r_hat (seed0, seed1):", [(a, round(b, 1), round(c_, 1), round(d, 1)) for a, b, c_, d in curve[m]])
print("focus MAE vs input fp16", {g: round(v, 2) for g, v in focus["fp16"].items()})
print("focus MAE vs input fp32", {g: round(v, 2) for g, v in focus["fp32"].items()}, "median diff %.2f" % focus_diff)
print("focus r_hat", focus_r, "anchors", anchors)
print("runtime", rt)
print("shapes/dtypes", shapes)
for m, d in spread.items():
    print("spread", m, {v: (round(x["leave"], 4), round(x["nonzero_targets"], 1)) for v, x in d.items()})

# sheet: input + 11 per mode (seed 0) = 23 images
T = 140
labels = [("K0", "K0")] + [(f"{D.NOM * D.K * g:.1f}px", f"g{g:.4f}") for g in meta["gaps"]]
sheet = Image.new("RGB", ((len(labels) + 1) * (T + 4), 2 * (T + 16)), "white")
d = ImageDraw.Draw(sheet)
for i, m in enumerate(("fp16", "fp32")):
    if i == 0:
        sheet.paste(Image.fromarray(s["image"]).resize((T, T), Image.LANCZOS), (0, 14))
        d.text((2, 1), "input", fill="black")
    for j, (lab, tag) in enumerate(labels):
        im = Image.open(f"{out}/img/{SC}_{tag}_{m}_s0.png")
        sheet.paste(im.resize((T, T), Image.LANCZOS), ((j + 1) * (T + 4), i * (T + 16) + 14))
        d.text(((j + 1) * (T + 4) + 2, i * (T + 16) + 1), f"{m} {lab}", fill="black")
sheet.save(f"{res}/precision_sheet.png")
print("sheet written")
