"""Round 19 analysis: gate, per-cell measurements, frozen decisions (see README), contact sheet.
Usage: python analyze19.py <R19_OUT> <results dir>"""
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
calib = json.load(open(f"{out}/calib.json"))
conds = [c for c in meta["conds"] if c["kind"] not in ("repeat", "gate")]
ld = lambda f: np.asarray(Image.open(os.path.join(out, f))).astype(np.float32)

val = M.validate(D)
json.dump(val, open(f"{res}/estimator_validation.json", "w"), indent=1)
gate = dict(estimator=val["passed"], repeat=meta["repeat_max_abs_diff"] <= 1.0,
            fp32_wrapper_matches_round21=meta["gate_max_abs_diff"] is not None and meta["gate_max_abs_diff"] <= 1.0,
            outputs_ok=all(c["finite"] and not c["black"] for c in meta["conds"]), calibration_frozen=True)
gate["passed"] = all(gate.values())
print("estimator validation", val["passed"], "worst %.2f px" % val["worst_err"], "closed form matches", val["closed_form_matches"])
print("gate", gate)

# ---- per-cell measurements
rows = []
for n in D.SCENES:
    s = D.make(n)
    img = s["image"].astype(np.float32)
    st = m18.stack(img, "disk")
    ints = {k: D.interior(s["cid"], k) for k in range(9)}
    Bs = {k: M.stack_at(st, ints[k]) for k in range(9)}
    y0 = ld(f"img/{n}_K0.png")
    for k in range(9):
        f = M.fit_fast(y0[ints[k]], Bs[k])
        rows.append(dict(scene=n, run="K0", perm="P1", method="none", seed=0, cell=k, side="none", gap=0.0, r_nom=0.0,
                         r_hat=f["v"], at_cap=f["at_cap"], mae_in=float(np.abs(y0 - img)[ints[k]].mean()), mae_k0=0.0,
                         texture=s["texture"][k]))
    for c in [c for c in conds if c["scene"] == n and c["kind"] != "K0"]:
        y = ld(c["file"])
        for cl in c["cells"]:
            k = cl["cell"]
            f = M.fit_fast(y[ints[k]], Bs[k])
            rows.append(dict(scene=n, run=c["run"], perm=c["perm"], method=c["method"], seed=c["seed"], cell=k,
                             pair=cl["pair"], side=cl["side"], gap=cl["gap"], d_f=c["d_f"], r_nom=cl["r_nom"],
                             r_req=cl.get("r_req", cl["r_nom"]), req_status=cl.get("req_status", "none"),
                             r_hat=f["v"], at_cap=f["at_cap"], texture=s["texture"][k],
                             mae_in=float(np.abs(y - img)[ints[k]].mean()), mae_k0=float(np.abs(y - y0)[ints[k]].mean())))
    print("measured", n, flush=True)

agree = lambda r: (not r["at_cap"]) and abs(r["r_hat"] - r["r_nom"]) <= M.TOL(r["r_nom"])
sel = lambda **kw: [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
frac = lambda rs: float(np.mean([agree(r) for r in rs])) if rs else float("nan")

# ---- 1. calibration on held-out test scenes
E = {m: [r for r in rows if r["scene"] in D.TEST and r["run"] in D.GRID_RUNS and r["seed"] == 0
         and r["method"] == m and r["side"] != "focus"] for m in ("none", "C0", "C1", "C2")}
cal = {m: dict(n=len(v), agree=frac(v), above_range=int(sum(r["at_cap"] for r in v))) for m, v in E.items()}
bins = [(0, 3), (3, 6), (6, 12), (12, 20), (20, 40)]
for m, v in E.items():
    cal[m]["by_target"] = {f"{a}-{b}": frac([r for r in v if a <= r["r_nom"] < b]) for a, b in bins}
    cal[m]["by_side"] = {s: frac([r for r in v if r["side"] == s]) for s in ("front", "behind")}
    cal[m]["by_run"] = {g: frac([r for r in v if r["run"] == g]) for g in D.GRID_RUNS}
    cal[m]["by_texture"] = {t: frac([r for r in v if r["texture"].split("_")[0] == t]) for t in ("grass", "gravel", "brick")}
    cal[m]["median_rel_residual_by_side"] = {
        s: float(np.median([(r["r_hat"] - r["r_nom"]) / max(r["r_nom"], 1) for r in v if r["side"] == s and not r["at_cap"]]))
        for s in ("front", "behind")}
    cal[m]["by_group"] = {g: frac([r for r in v if a <= r["r_nom"] < b]) for g, (a, b) in
                          {"tiny <2": (0, 2), "mild 2-6": (2, 6), "larger >=6": (6, 99)}.items()}
    cal[m]["n_by_group"] = {g: len([r for r in v if a <= r["r_nom"] < b]) for g, (a, b) in
                            {"tiny <2": (0, 2), "mild 2-6": (2, 6), "larger >=6": (6, 99)}.items()}
    cal[m]["by_scene"] = {sc: frac([r for r in v if r["scene"] == sc]) for sc in D.TEST}
    cal[m]["req_status"] = {k: int(sum(r["req_status"] == k for r in v)) for k in ("ok", "below_floor", "unreachable", "none")}
if cal["C1"]["agree"] >= 0.8:
    cal_outcome = "O1: a monotone curve of requested radius alone (C1) corrects held-out scenes"
elif cal["C2"]["agree"] >= 0.8:
    cal_outcome = "O2: curves by requested radius and side of focus (C2) correct held-out scenes"
else:
    cal_outcome = "O3: neither C1 nor C2 reaches 80% on held-out scenes; the same request has different effects"

# failure concentration (frozen): a factor "dominates" if one of its levels holds >= 60 % of C1 failures
fails = [r for r in E["C1"] if not agree(r)]
def share(key):
    if not fails:
        return {}
    vals = [key(r) for r in fails]
    return {str(k): vals.count(k) / len(vals) for k in sorted(set(vals), key=str)}
binof = lambda r: next(f"{a}-{b}" for a, b in bins if a <= r["r_nom"] < b) if r["r_nom"] < 40 else "40+"
conc = dict(n_fail=len(fails), by_run=share(lambda r: r["run"]), by_scene=share(lambda r: r["scene"]),
            by_target=share(binof), by_side=share(lambda r: r["side"]), by_texture=share(lambda r: r["texture"].split("_")[0]))
conc["dominant"] = {k: max(v.items(), key=lambda t: t[1]) for k, v in conc.items() if isinstance(v, dict) and v and max(v.values()) >= 0.6}

# ---- 2. symmetry: identical patches, equal gaps in front / behind, positions swapped (P1 vs P2)
U = [r for r in rows if r["method"] == "none" and r["seed"] == 0 and r["run"] in D.RUNS]
idx = {(r["scene"], r["run"], r["perm"], r["cell"]): r for r in U}
sym = []
for n in D.SCENES:
    for run in ("R2", "R3", "R4"):
        _, gaps = D.RUNS[run]
        for p, ((a, b), (ga, gb)) in enumerate(zip(D.PAIRS, gaps)):
            if (p == 0 and run != "R2") or abs(ga) < 0.175:
                continue
            ca, cb = 3 * a[0] + a[1], 3 * b[0] + b[1]
            lr, cap = [], False
            for perm in ("P1", "P2"):
                ra, rb = idx[(n, run, perm, ca)], idx[(n, run, perm, cb)]
                front, behind = (ra, rb) if ra["side"] == "front" else (rb, ra)
                cap |= front["at_cap"] or behind["at_cap"]
                lr.append(np.log(behind["r_hat"] / front["r_hat"]))
            sym.append(dict(scene=n, run=run, pair=p, gap=abs(ga), r_nom=D.NOM * D.K * abs(ga), above_range=cap,
                            ratio_behind_front=float(np.exp(np.mean(lr))), ratio_P1=float(np.exp(lr[0])), ratio_P2=float(np.exp(lr[1]))))
sv = [np.log(x["ratio_behind_front"]) for x in sym if not x["above_range"]]
med = float(np.median(sv))
share = float(np.mean(np.sign(sv) == np.sign(med)))
symmetry = dict(n=len(sv), n_above_range=sum(x["above_range"] for x in sym), median_ratio=float(np.exp(med)),
                same_sign_share=share, asymmetric=bool(abs(med) >= np.log(1.15) and share >= 0.8), instances=sym)

# ---- 3. context: same anchor patch and request, calm (R0) vs busy (R3); focus cell detail
ctx, foc = [], []
for n in D.SCENES:
    for perm in ("P1", "P2"):
        for k in (0, 8):
            c0_, c3_ = idx[(n, "R0", perm, k)], idx[(n, "R3", perm, k)]
            ctx.append(dict(scene=n, perm=perm, cell=k, side=c0_["side"], r_calm=c0_["r_hat"], r_busy=c3_["r_hat"],
                            r_R1=idx[(n, "R1", perm, k)]["r_hat"], r_R2=idx[(n, "R2", perm, k)]["r_hat"],
                            above_range=c0_["at_cap"] or c3_["at_cap"],
                            changed=bool(abs(c3_["r_hat"] - c0_["r_hat"]) > M.TOL(c0_["r_hat"]))))
        f0, f3 = idx[(n, "R0", perm, 4)], idx[(n, "R3", perm, 4)]
        foc.append(dict(scene=n, perm=perm, mae_k0_calm=f0["mae_k0"], mae_k0_busy=f3["mae_k0"],
                        mae_in_calm=f0["mae_in"], mae_in_busy=f3["mae_in"],
                        mae_in_k0=[r for r in rows if r["scene"] == n and r["run"] == "K0" and r["cell"] == 4][0]["mae_in"],
                        r_calm=f0["r_hat"], r_busy=f3["r_hat"]))
s1 = {(r["scene"], r["cell"]): r for r in rows if r["seed"] == 1}
seed_noise = [abs(s1[(n, k)]["r_hat"] - idx[(n, "R3", "P1", k)]["r_hat"]) for n in D.SCENES for k in range(9)
              if (n, k) in s1 and not s1[(n, k)]["at_cap"]]
context = dict(
    anchor_changed_frac=float(np.mean([c["changed"] for c in ctx if not c["above_range"]])),
    anchor_effect=bool(np.mean([c["changed"] for c in ctx if not c["above_range"]]) >= 0.5),
    anchor_median_busy_minus_calm=float(np.median([c["r_busy"] - c["r_calm"] for c in ctx if not c["above_range"]])),
    focus_vs_k0_up_frac=float(np.mean([f["mae_k0_busy"] - f["mae_k0_calm"] >= 1.0 for f in foc])),
    focus_vs_input_up_frac=float(np.mean([f["mae_in_busy"] - f["mae_in_calm"] >= 1.0 for f in foc])),
    focus_vs_input_median_increase=float(np.median([f["mae_in_busy"] - f["mae_in_calm"] for f in foc])),
    seed_noise_max=float(max(seed_noise)), seed_noise_median=float(np.median(seed_noise)), anchors=ctx, focus=foc)
context["focus_detail_effect_vs_k0"] = context["focus_vs_k0_up_frac"] >= 0.8
context["focus_extra_damage_vs_input"] = context["focus_vs_input_up_frac"] >= 0.8

# ---- 4. small blur: can C1-corrected requests produce 3-6 px?
small = [r for r in E["C1"] if 3 <= r["r_nom"] < 6]
plateau = dict(n=len(small), within_1px=float(np.mean([abs(r["r_hat"] - r["r_nom"]) <= 1 for r in small])) if small else None,
               uncorrected_within_1px=float(np.mean([abs(r["r_hat"] - r["r_nom"]) <= 1 for r in E["none"] if 3 <= r["r_nom"] < 6])),
               values=[(round(r["r_nom"], 2), round(r["r_req"], 2), round(r["r_hat"], 2)) for r in small])
plateau["attainable"] = bool(plateau["within_1px"] is not None and plateau["within_1px"] >= 0.8)

summary = dict(gate=gate, calibration=dict(outcome=cal_outcome, methods=cal, C0_gain=calib["C0"], floor=calib["floor"]),
               failure_concentration=conc, pisa_calls=meta.get("pisa_calls"), pisa_shapes=meta.get("pisa_shapes"),
               symmetry=symmetry, context=context, plateau=plateau, repeat_max_abs_diff=meta["repeat_max_abs_diff"],
               peak_mem_alloc_gb=meta["peak_mem_alloc_gb"], mean_seconds=meta["mean_seconds"], n_outputs=meta["n_outputs"])
json.dump(summary, open(f"{res}/confirm.json", "w"), indent=1)
json.dump(rows, open(f"{res}/cells.json", "w"))
json.dump(calib, open(f"{res}/calib.json", "w"), indent=1)

pr = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k not in ("instances", "anchors", "focus", "values", "methods")}
print("CALIBRATION", cal_outcome)
for m, v in cal.items():
    print(" ", m, "agree %.2f (n %d, above range %d)" % (v["agree"], v["n"], v["above_range"]), "by target", {k: round(x, 2) for k, x in v["by_target"].items()},
          "by side", {k: round(x, 2) for k, x in v["by_side"].items()}, "rel resid", {k: round(x, 2) for k, x in v["median_rel_residual_by_side"].items()}, v["req_status"])
    print("    groups", {k: round(x, 2) for k, x in v["by_group"].items()}, v["n_by_group"], "scenes", {k: round(x, 2) for k, x in v["by_scene"].items()})
print("FAILURES (C1)", {k: ({a: round(b, 2) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in conc.items()})
print("SYMMETRY", pr(symmetry))
print("CONTEXT", pr(context))
print("PLATEAU", pr(plateau))
print("C1 curve", [(round(a, 1), round(b, 1)) for a, b in zip(calib["C1"]["x"], calib["C1"]["y"])])

# ---- sheet (24 images): rows S3, S4, S5
T = 192
cols = [("input", None), ("K0", "K0"), ("calm R0 P1", "R0_P1"), ("busy R3 P1", "R3_P1"), ("R3 P2 (swap)", "R3_P2"),
        ("R3 P1 C1", "R3_P1_C1"), ("R3 P1 C2", "R3_P1_C2"), ("R5 P1 C1", "R5_P1_C1")]
sheet = Image.new("RGB", (len(cols) * (T + 4), 3 * (T + 18)), "white")
d = ImageDraw.Draw(sheet)
for i, n in enumerate(["S3", "S4", "S5"]):
    for j, (lab, tag) in enumerate(cols):
        im = Image.fromarray(D.make(n)["image"]) if tag is None else Image.open(f"{out}/img/{n}_{tag}.png")
        sheet.paste(im.resize((T, T), Image.LANCZOS), (j * (T + 4), i * (T + 18) + 16))
        d.text((j * (T + 4) + 2, i * (T + 18) + 2), f"{n}: {lab}", fill="black")
sheet.save(f"{res}/confirm_sheet.png")
print("sheet written")
