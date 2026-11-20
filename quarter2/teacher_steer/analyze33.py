"""Round 33 analysis: frozen decision for teacher activation difference injection (privileged feasibility).

valid = |r_hat - t| <= max(2 px, 20 %) and focus guard <= 0.5/255.
Step 1, pass-through: the teacher keeps the target if, in >= 6 of 8 cases, its output is valid and its primary
error is <= 0.5 x the request-only error (round 31's best request-only output, regenerated). Otherwise STOP:
the teacher's activations do not encode the target better than the request, and step 2 is not interpreted.
Step 2, per candidate location (post_pisa, down2, mid, up0, up1; up2 is the ceiling): per case the best valid alpha
(lowest primary error; privileged, like the request-only search). Unattainable = failure (ratio inf; request
unattainable and injection valid -> 0). A location PASSES if median(injection / request-only) <= 0.80, wins >= 6/8,
and its unedited-cell change (rocket_0, text_0, MAE vs baseline, max over targets) <= the largest request-only
other-cell change + 0.5/255.
Round: PASS-THROUGH FAIL / PASS (some location passes) / FAIL.
usage: python analyze33.py <R33_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import metrics29 as Q  # noqa: E402

RATIO_MAX, WINS_MIN, PRES_MARGIN, PT_RATIO, PT_MIN = 0.80, 6, 0.5, 0.5, 6
out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
rows = meta["rows"]
valid = lambda d: abs(d["r_hat"] - d["t"]) <= m18.TOL(d["t"]) and d["focus_add"] <= Q.FOCUS_ADD_MAX
one = lambda kind, k, t: next(r for r in rows if r["kind"] == kind and r["cell"] == k and r["t"] == t)
keys = [(k, p, t) for t in Q.TARGETS for k, p in Q.CASES]

pt, pt_ok = {}, 0
for k, p, t in keys:
    a, b, z = one("teacher", k, t), one("request", k, t), one("base", k, t)
    ok = valid(a) and a["err"] <= PT_RATIO * b["err"]
    pt_ok += ok
    pt[f"{p}@{t:g}"] = dict(teacher_err=a["err"], teacher_r=a["r_hat"], request_err=b["err"], base_err=z["err"], ok=ok)
passthrough = pt_ok >= PT_MIN
req_others = max(one("request", k, t)["others_vs_base"] for k, p, t in keys)

locs = {}
for loc in meta["locs"]:
    ratios, wins, cases, others = [], 0, {}, 0.0
    for k, p, t in keys:
        cand = [r for r in rows if r["kind"] == "inject" and r["loc"] == loc and r["cell"] == k and r["t"] == t]
        others = max(others, max(r["others_vs_base"] for r in cand))
        ok = [r for r in cand if valid(r)]
        b = one("request", k, t)
        x = min(ok, key=lambda r: r["err"]) if ok else None
        ratio = (x["err"] / b["err"] if valid(b) else 0.0) if x else float("inf")
        ratios.append(ratio)
        wins += ratio < 1
        cases[f"{p}@{t:g}"] = dict(ratio=ratio, alpha=x and x["alpha"], err=x and x["err"], r_hat=x and x["r_hat"],
                                   request_err=b["err"], by_alpha={r["alpha"]: (r["err"], r["r_hat"]) for r in cand})
    med = float(np.median(ratios))
    core = med <= RATIO_MAX and wins >= WINS_MIN
    locs[loc] = dict(median_ratio=med, wins=int(wins), others_max=others, ceiling=(loc == "up2"),
                     passed=core and others <= req_others + PRES_MARGIN and loc != "up2", cases=cases)

passing = [l for l, v in locs.items() if v["passed"]]
decision = ("GATE FAIL" if not meta["gate_ok"] else "PASS-THROUGH FAIL" if not passthrough else
            "PASS" if passing else "FAIL")
summary = dict(decision=decision, passing=passing, gate=meta["gate"], gate_ok=meta["gate_ok"],
               n_generations=meta["n_generations"], passthrough=dict(ok_cases=int(pt_ok), passed=passthrough, cases=pt),
               request_others_max=req_others,
               locations={l: {k: v for k, v in d.items() if k != "cases"} for l, d in locs.items()})
json.dump(summary, open(f"{res}/teacher.json", "w"), indent=1)
json.dump({l: d["cases"] for l, d in locs.items()}, open(f"{res}/teacher_cases.json", "w"), indent=1)
print("gate ok", meta["gate_ok"], "| DECISION", decision, passing)
print("pass-through:", pt_ok, "/ 8")
for key, d in pt.items():
    print(f"  {key:18s} base {d['base_err']:.2f} | request {d['request_err']:.2f} | teacher {d['teacher_err']:.2f} "
          f"r {d['teacher_r']:.1f} {'ok' if d['ok'] else 'NO'}")
for l, d in locs.items():
    print(f"{l:9s} median {d['median_ratio']:.3f} wins {d['wins']}/8 others {d['others_max']:.2f} "
          f"(request {req_others:.2f}) {'CEILING' if d['ceiling'] else ''} pass {d['passed']}")
    print("   " + " ".join(f"{c}:{v['ratio']:.2f}(a{v['alpha']})" for c, v in d["cases"].items()))

# crop sheet at 11.625: input | target | request-only | teacher | best location (alpha 1.0)
best = min([l for l in locs if l != "up2"], key=lambda l: locs[l]["median_ratio"])
img, cid, _ = Q.scene()
im = img.astype(np.float32)
t = 11.625
ld = lambda tag: np.asarray(Image.open(f"{out}/img/{tag}.png")).astype(np.float32)
T = 160
sheet = Image.new("RGB", (5 * T, 4 * (T + 16)), "white")
dr = ImageDraw.Draw(sheet)
for i, (k, p) in enumerate(Q.CASES):
    ys, xs = np.where(D.interior(cid, k))
    cy, cx = int(ys.mean()), int(xs.mean())
    sl = (slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))
    c = locs[best]["cases"][f"{p}@{t:g}"]
    a = c["alpha"] if c["alpha"] is not None else 1.0
    tiles = [(im, "input"), (Q.target(img, t), "target"), (ld(f"req_{p}_t{t:.1f}"), "request"),
             (ld(f"teacher_t{t:.1f}"), "teacher"), (ld(f"inj_{best}_a{a:.2f}_t{t:.1f}"), f"{best} a{a}")]
    for j, (x, lab) in enumerate(tiles):
        sheet.paste(Image.fromarray(np.clip(x[sl], 0, 255).astype(np.uint8)).resize((T, T), Image.NEAREST), (j * T, i * (T + 16) + 16))
        dr.text((j * T + 3, i * (T + 16) + 2), f"{p} {lab}", fill="black")
sheet.save(f"{res}/crops_t11.6.png")
print("crop sheet location:", best)
