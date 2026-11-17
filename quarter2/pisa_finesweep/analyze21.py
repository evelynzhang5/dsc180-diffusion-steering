"""Round 21 analysis. Usage: python analyze21.py <R21_OUT> <results dir>"""
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
SC, KS = meta["scene"], meta["Ks"]
conds = [c for c in meta["conds"] if c["kind"] in ("K0", "sweep")]
ld = lambda f: np.asarray(Image.open(os.path.join(out, f))).astype(np.float32)
MODES = ("fp16", "fp32")
TARGETS, REACH, JUMP = [2, 3, 4, 5], 0.5, 1.0

val = M.validate(D)
gate = dict(estimator=val["passed"],
            wrapper_fp16_matches_round20=meta["gate_max_abs_diff"] is not None and meta["gate_max_abs_diff"] <= 1.0,
            repeat=meta["repeat_max_abs_diff"] <= 1.0, outputs_ok=all(c["finite"] and not c["black"] for c in meta["conds"]))
gate["passed"] = all(gate.values())

s = D.make(SC)
img = s["image"].astype(np.float32)
st = m18.stack(img, "disk")
ints = {k: D.interior(s["cid"], k) for k in range(9)}
Bs = {k: M.stack_at(st, ints[k]) for k in range(9)}
FREE = [3 * rc[0] + rc[1] for p in (1, 2, 3) for rc in D.PAIRS[p]]
rows = []
for c in conds:
    y = ld(c["file"])
    for k in FREE + [4]:
        f = M.fit_fast(y[ints[k]], Bs[k])
        rows.append(dict(mode=c["mode"], seed=c["seed"], K=c["K"], r_nom=D.NOM * c["K"] * meta["gap"], cell=k,
                         role="focus" if k == 4 else "free", kind=c["kind"], r_hat=f["v"], at_cap=f["at_cap"],
                         mae_in=float(np.abs(y - img)[ints[k]].mean())))
med = lambda xs: float(np.median(xs))
def curve(mode, seed=None, cell=None):
    return [med([r["r_hat"] for r in rows if r["mode"] == mode and r["kind"] == "sweep" and r["K"] == K and r["role"] == "free"
                 and (seed is None or r["seed"] == seed) and (cell is None or r["cell"] == cell)]) for K in KS]

R = [D.NOM * K * meta["gap"] for K in KS]
summary = dict(gate=gate, r_nom=R, Ks=KS, modes={})
for m in MODES:
    cv = curve(m)
    jumps = np.diff(cv)
    reach = {t: any(abs(v - t) <= REACH for v in cv) for t in TARGETS}
    skipped = {t: bool(any(cv[i] < t - REACH and cv[i + 1] > t + REACH and jumps[i] > JUMP for i in range(len(cv) - 1))) for t in TARGETS}
    if all(reach.values()) and jumps.max() <= JUMP:
        outcome = "F1: continuous but narrow - every target reachable, no step > 1 px between 0.1 px request steps"
    elif any(skipped.values()):
        outcome = "F2: skips - a target is jumped over (step > 1 px between neighbouring requests 0.1 px apart)"
    else:
        outcome = "F3: neither (targets unreachable without a straddling jump, or jumps away from targets)"
    # per-cell: request at which each cell first reaches each target (linear interpolation on its own curve)
    need = {}
    for k in FREE:
        ck = curve(m, cell=k)
        need[k] = {}
        for t in TARGETS:
            j = next((i for i in range(len(ck)) if ck[i] >= t), None)
            need[k][t] = None if j is None else (R[0] if j == 0 else float(np.interp(t, [ck[j - 1], ck[j]], [R[j - 1], R[j]])))
    spread = {t: [need[k][t] for k in FREE if need[k][t] is not None] for t in TARGETS}
    rise = [i for i in range(len(cv)) if cv[i] > cv[0] + 0.5]
    slope = None
    if rise:
        a = max(rise[0] - 1, 0)
        b = next((i for i in range(len(cv)) if cv[i] >= 5.5), len(cv) - 1)
        slope = float((cv[b] - cv[a]) / (R[b] - R[a])) if b > a else None
    floor = med([r["r_hat"] for r in rows if r["mode"] == m and r["kind"] == "K0" and r["role"] == "free"])
    summary["modes"][m] = dict(outcome=outcome, curve=cv, curve_seed0=curve(m, 0), curve_seed1=curve(m, 1), max_step=float(jumps.max()),
                               max_step_at_r_nom=[R[int(np.argmax(jumps))], R[int(np.argmax(jumps)) + 1]], reachable=reach, skipped=skipped,
                               floor=floor, slope_px_per_px=slope,
                               request_needed_per_cell={str(k): v for k, v in need.items()},
                               request_spread_across_cells={t: (float(min(v)), float(max(v)), len(v)) if v else None for t, v in spread.items()},
                               seed_diff_median=med(np.abs(np.array(curve(m, 0)) - np.array(curve(m, 1)))),
                               focus_mae_in_median=med([r["mae_in"] for r in rows if r["mode"] == m and r["role"] == "focus" and r["kind"] == "sweep"]))
json.dump(summary, open(f"{res}/finesweep.json", "w"), indent=1)
json.dump(rows, open(f"{res}/cells.json", "w"))
print("gate", gate)
for m in MODES:
    d = summary["modes"][m]
    print(m, "OUTCOME", d["outcome"])
    print("  curve", [(round(r, 2), round(v, 2)) for r, v in zip(R, d["curve"])])
    print("  max step %.2f px between r_nom %s; reachable %s; skipped %s; floor %.2f; slope %s; seed diff %.2f; focus mae %.2f" % (
        d["max_step"], [round(x, 2) for x in d["max_step_at_r_nom"]], d["reachable"], d["skipped"], d["floor"],
        None if d["slope_px_per_px"] is None else round(d["slope_px_per_px"], 1), d["seed_diff_median"], d["focus_mae_in_median"]))
    print("  request needed across 6 cells (min, max, n):", {t: (None if v is None else (round(v[0], 2), round(v[1], 2), v[2])) for t, v in d["request_spread_across_cells"].items()})

# sheet: 128 px crop of swept cell 1 (top middle); input + every other level per mode (24 images)
y0, x0 = 40, 192
crop = lambda a: Image.fromarray(np.asarray(a)[y0:y0 + 128, x0:x0 + 128])
idx = list(range(0, len(KS), 2))
T = 128
sheet = Image.new("RGB", ((len(idx) + 1) * (T + 4), 2 * (T + 16)), "white")
d = ImageDraw.Draw(sheet)
sheet.paste(crop(s["image"]), (0, 14))
d.text((2, 1), "input", fill="black")
for i, m in enumerate(MODES):
    for j, q in enumerate(idx):
        im = Image.open(f"{out}/img/{SC}_K{KS[q]:.1f}_{m}_s0.png")
        sheet.paste(crop(im), ((j + 1) * (T + 4), i * (T + 16) + 14))
        d.text(((j + 1) * (T + 4) + 2, i * (T + 16) + 1), f"{m} {R[q]:.1f}px", fill="black")
sheet.save(f"{res}/finesweep_sheet.png")
print("sheet written")
