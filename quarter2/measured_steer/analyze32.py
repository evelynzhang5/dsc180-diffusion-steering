"""Round 32 analysis: frozen decision, measured-error adaptive steering vs equally adaptive request adjustment.

Compared outputs are each method's final combined generation (one image per method and target, all 4 case cells
corrected at once). Per case (4 cells x 2 targets):
  steering = S+R final of an arm (R1 requests + one adaptive steering round, 7 generations)
  request  = R+R final (R1 requests + one more adaptive request round, 7 generations)
  valid    = |r_hat - t| <= max(2 px, 20 %) and focus guard <= 0.5/255; invalid counts as a failure
             (steering invalid -> ratio inf; request invalid and steering valid -> ratio 0).
An arm PASSES if median(steering err / request err) <= 0.80, steering wins >= 6 of 8, and its unedited cells
(rocket_0, text_0) change no more than the request method's + 0.5/255 (MAE vs baseline, mean over the 2 targets).
Round PASS: some size/shape arm passes and the random arm at that layer does not meet the ratio-and-wins part.
Reported: steering alone (S final) vs request round 1 (R1 final), same rules; probe responses.
usage: python analyze32.py <R32_OUT> <results dir>
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

RATIO_MAX, WINS_MIN, PRES_MARGIN = 0.80, 6, 0.5
out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
rows = meta["rows"]
valid = lambda d: abs(d["r_hat"] - d["t"]) <= m18.TOL(d["t"]) and d["focus_add"] <= Q.FOCUS_ADD_MAX
final = lambda stage, arm, k, t: next(r for r in rows if r["stage"] == stage and r["arm"] == arm and r["kind"] == "final"
                                      and r["cell"] == k and r["t"] == t)


def compare(stage_a, arm_a, stage_b, arm_b):
    cases, ratios, wins, pres = {}, [], 0, {}
    for t in Q.TARGETS:
        pres[f"{t:g}"] = (final(stage_a, arm_a, Q.CASES[0][0], t)["others_vs_base"],
                          final(stage_b, arm_b, Q.CASES[0][0], t)["others_vs_base"])
        for k, p in Q.CASES:
            a, b = final(stage_a, arm_a, k, t), final(stage_b, arm_b, k, t)
            va, vb = valid(a), valid(b)
            ratio = a["err"] / b["err"] if va and vb else (0.0 if va else float("inf"))
            ratios.append(ratio)
            wins += ratio < 1
            cases[f"{p}@{t:g}"] = dict(ratio=ratio, a=dict(err=a["err"], r_hat=a["r_hat"], valid=va, setting=a["setting"]),
                                       b=dict(err=b["err"], r_hat=b["r_hat"], valid=vb, setting=b["setting"]))
    med = float(np.median(ratios))
    oa = float(np.mean([v[0] for v in pres.values()]))
    ob = float(np.mean([v[1] for v in pres.values()]))
    core = med <= RATIO_MAX and wins >= WINS_MIN
    return dict(median_ratio=med, wins=int(wins), core=core, others_steer=oa, others_request=ob,
                preserved=oa <= ob + PRES_MARGIN, passed=core and oa <= ob + PRES_MARGIN, cases=cases)


arms = [f"{L}_{d}" for L in meta["layers"] for d in meta["dirs"]]
main = {a: compare("SR", a, "RR", "request") for a in arms}
alone = {a: compare("S", a, "R1", "request") for a in arms}
real = [a for a in arms if not a.endswith("random")]
passing = [a for a in real if main[a]["passed"] and not main[a.split("_")[0] + "_random"]["core"]]
decision = "PASS" if passing else "FAIL"

# probe responses (exploratory): per arm, target, cell -- error and radius across alphas on the R1 requests
probe = {}
for a in arms:
    for t in Q.TARGETS:
        for k, p in Q.CASES:
            rs = sorted([r for r in rows if r["stage"] == "SR" and r["arm"] == a and r["kind"] == "probe"
                         and r["cell"] == k and r["t"] == t], key=lambda r: r["setting"])
            st = final("R1", "request", k, t)
            probe[f"{a}|{p}@{t:g}"] = [(0.0, st["err"], st["r_hat"])] + [(r["setting"], r["err"], r["r_hat"]) for r in rs]

summary = dict(decision=decision, passing_arms=passing, gate=meta["gate"], n_generations=meta["n_generations"],
               rule=dict(ratio_max=RATIO_MAX, wins_min=WINS_MIN, preservation_margin=PRES_MARGIN),
               main={a: {k: v for k, v in m.items() if k != "cases"} for a, m in main.items()},
               steering_alone_vs_R1={a: {k: v for k, v in m.items() if k != "cases"} for a, m in alone.items()})
json.dump(summary, open(f"{res}/measured_steer.json", "w"), indent=1)
json.dump(dict(main={a: m["cases"] for a, m in main.items()}, alone={a: m["cases"] for a, m in alone.items()},
               probe=probe), open(f"{res}/measured_steer_cases.json", "w"), indent=1)
print("gate", meta["gate"])
print("DECISION", decision, passing)
print(f"{'arm':12s} | S+R vs R+R: median  wins  others(S/R)  pass | S vs R1: median wins")
for a in arms:
    m, s = main[a], alone[a]
    print(f"{a:12s} | {m['median_ratio']:6.3f} {m['wins']:3d}/8  {m['others_steer']:.2f}/{m['others_request']:.2f}  "
          f"{str(m['passed']):5s} | {s['median_ratio']:6.3f} {s['wins']:3d}/8")
for t in Q.TARGETS:
    for k, p in Q.CASES:
        b, r1, rr = final("base", "base", k, t), final("R1", "request", k, t), final("RR", "request", k, t)
        print(f"{p}@{t:g}: base {b['err']:.2f} r{b['r_hat']:.1f} | R1 {r1['err']:.2f} r{r1['r_hat']:.1f} | "
              f"R+R {rr['err']:.2f} r{rr['r_hat']:.1f} | " +
              " ".join(f"{a} {final('SR', a, k, t)['err']:.2f}(a{final('SR', a, k, t)['setting']:+.1f})" for a in real))

# crop sheet at 11.625 for the arm with the lowest S+R median ratio: input | target | R+R | S+R
best = min(real, key=lambda a: main[a]["median_ratio"])
img, cid, _ = Q.scene()
im = img.astype(np.float32)
t = 11.625
tgt = Q.target(img, t)
ld = lambda tag: np.asarray(Image.open(f"{out}/img/{tag}.png")).astype(np.float32)
T = 160
sheet = Image.new("RGB", (4 * T, 4 * (T + 16)), "white")
dr = ImageDraw.Draw(sheet)
for i, (k, p) in enumerate(Q.CASES):
    ys, xs = np.where(D.interior(cid, k))
    cy, cx = int(ys.mean()), int(xs.mean())
    sl = (slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))
    tiles = [(im, "input"), (tgt, "target"), (ld(final("RR", "request", k, t)["tag"]), "request R+R"),
             (ld(final("SR", best, k, t)["tag"]), f"steer {best}")]
    for j, (a, lab) in enumerate(tiles):
        sheet.paste(Image.fromarray(np.clip(a[sl], 0, 255).astype(np.uint8)).resize((T, T), Image.NEAREST), (j * T, i * (T + 16) + 16))
        dr.text((j * T + 3, i * (T + 16) + 2), f"{p} {lab}", fill="black")
sheet.save(f"{res}/crops_t11.6.png")
print("crop sheet arm:", best)
