"""Round 45 (CPU, on saved outputs): score the fresh-scene confirmation. FROZEN before any round 45 image exists.

Scenes: the 24 fresh generated scenes that pass (1) automatic eligibility (plan45.json: >= 4 behind tiles) and (2) the
input-only review (review45.json: mask failures and measurement flags, decided on the INPUT sheets before any output is
viewed). Coverage counts every loss: generated, prepared (depth + mask), eligible, reviewed in, scored.
Per cell (scene x target x arm): blur on the EVALUATION tiles (odd-indexed behind tiles; primary) and on all behind tiles
(secondary); ok = |blur - t| <= max(2, 0.2 t), unmeasurable = miss; |error|; subject hf_ratio and colour_shift vs the
input (measures40.preservation, colour in 0-255 units); render seconds.
Scene level first (mean over the 3 targets, or per target), then paired scene bootstrap (10,000, seed 0).
PRIMARY comparison: monotone - linear (both fp32, same scenes). Secondary: monotone - fixed, linear - fixed.
Decision (frozen):
  A  attainment: monotone - linear scene-averaged within-tolerance >= +0.15 and its 95% CI lower bound > 0;
  B  no target deteriorates: for each of 4, 8, 12 px the paired point estimate >= -0.10;
  C  preservation: paired subject hf_ratio CI lower bound >= -0.05 and paired colour_shift CI upper bound <= +1.0.
  CONFIRMED = A and B and C; PARTIAL = A but not (B and C); NOT CONFIRMED = not A.
usage (repo root): python quarter2/confirm45/score45.py <data> <out> <results dir> <review45.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compare40"))
import blur40 as BL  # noqa: E402
import common40 as C  # noqa: E402
import measures40 as MS  # noqa: E402

DATA, OUT, RES, REVIEW = sys.argv[1:5]
os.makedirs(RES, exist_ok=True)
P = json.load(open(f"{DATA}/plan45.json"))
run = json.load(open(f"{OUT}/bd45.json"))
review = json.load(open(REVIEW))
rng = np.random.default_rng(0)
ARMS = ["fixed", "linear", "monotone"]
gen = json.load(open(f"{DATA}/gen45.json"))["rows"]
prepared = [s["id"] for s in gen if os.path.exists(f"{DATA}/depth/{s['id']}_pred.npy")
            and os.path.exists(f"{DATA}/mask/{s['id']}.png")]
excl = set(review["exclude"])
ids = [i for i in sorted(P["plan"]) if i not in excl and len(P["plan"][i]["eval_tiles"]) >= 2]
coverage = dict(generated=len(gen), prepared=len(prepared), eligible=len(P["plan"]),
                too_few_tiles=P["coverage"]["too_few_tiles"], review_excluded=sorted(excl & set(P["plan"])),
                scored=len(ids))

scored = []
for i in ids:
    x = C.img(f"{DATA}/input/{i}.png")
    H, W = x.shape[:2]
    subj = C.subject(DATA, i, (H, W))
    st_e, st_a = BL.TileStack(x, P["plan"][i]["eval_tiles"]), BL.TileStack(x, P["plan"][i]["select_tiles"]
                                                                           + P["plan"][i]["eval_tiles"])
    for r in [r for r in run["rows"] if r["id"] == i]:
        y = C.img(f"{OUT}/{r['file']}")
        b, ba = st_e.measure(y)["r"], st_a.measure(y)["r"]
        pv = MS.preservation(x, y, subj)
        t = r["target"]
        scored.append(dict(r, blur=b, blur_all=ba, ok=bool(np.isfinite(b) and abs(b - t) <= C.TOL(t)),
                           ok_all_tiles=bool(np.isfinite(ba) and abs(ba - t) <= C.TOL(t)),
                           abs_err=abs(b - t) if np.isfinite(b) else float("nan"),
                           hf_ratio=pv["hf_ratio"], colour_shift=pv["colour_shift"]))
    print(i, "scored", flush=True)


def sm(arm, key, ts=(4.0, 8.0, 12.0)):
    return np.array([np.nanmean([float(r[key]) for r in scored if r["id"] == i and r["arm"] == arm
                                 and r["target"] in ts]) for i in ids])


def boot(v, stat=np.mean):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, est=float("nan"), lo=float("nan"), hi=float("nan"))
    b = [stat(rng.choice(v, len(v))) for _ in range(10000)]
    return dict(n=int(len(v)), est=float(stat(v)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))


summary = {a: dict(ok=boot(sm(a, "ok")), ok_4=boot(sm(a, "ok", (4.0,))), ok_8=boot(sm(a, "ok", (8.0,))),
                   ok_12=boot(sm(a, "ok", (12.0,))), ok_all_tiles=boot(sm(a, "ok_all_tiles")),
                   abs_err=boot(sm(a, "abs_err"), np.median), hf_ratio=boot(sm(a, "hf_ratio"), np.median),
                   colour_shift=boot(sm(a, "colour_shift"), np.median), render_s=boot(sm(a, "render_s")))
           for a in ARMS}
paired = {}
for a, b in [("monotone", "linear"), ("monotone", "fixed"), ("linear", "fixed")]:
    paired[f"{a}-{b}"] = dict(
        ok=boot(sm(a, "ok") - sm(b, "ok")), ok_4=boot(sm(a, "ok", (4.0,)) - sm(b, "ok", (4.0,))),
        ok_8=boot(sm(a, "ok", (8.0,)) - sm(b, "ok", (8.0,))), ok_12=boot(sm(a, "ok", (12.0,)) - sm(b, "ok", (12.0,))),
        ok_all_tiles=boot(sm(a, "ok_all_tiles") - sm(b, "ok_all_tiles")),
        hf_ratio=boot(sm(a, "hf_ratio") - sm(b, "hf_ratio"), np.median),
        colour_shift=boot(sm(a, "colour_shift") - sm(b, "colour_shift"), np.median),
        render_s=boot(sm(a, "render_s") - sm(b, "render_s")))
p = paired["monotone-linear"]
A = p["ok"]["est"] >= 0.15 and p["ok"]["lo"] > 0
B = all(p[k]["est"] >= -0.10 for k in ("ok_4", "ok_8", "ok_12"))
Cc = p["hf_ratio"]["lo"] >= -0.05 and p["colour_shift"]["hi"] <= 1.0
decision = dict(verdict="CONFIRMED" if A and B and Cc else "PARTIAL" if A else "NOT CONFIRMED",
                A_attainment=bool(A), B_no_target_worse=bool(B), C_preservation=bool(Cc), pisa_calls=run["calls"])
json.dump(dict(decision=decision, coverage=coverage, summary=summary, paired=paired, rows=scored),
          open(f"{RES}/score45.json", "w"), indent=1, default=float)
print("\ncoverage:", coverage)
for a in ARMS:
    s = summary[a]
    print(f"{a:9s} ok {s['ok']['est']:.2f} [{s['ok']['lo']:.2f},{s['ok']['hi']:.2f}]  4px {s['ok_4']['est']:.2f}  "
          f"8px {s['ok_8']['est']:.2f}  12px {s['ok_12']['est']:.2f}  all-tiles {s['ok_all_tiles']['est']:.2f}  "
          f"|err| {s['abs_err']['est']:.2f}  hf {s['hf_ratio']['est']:.3f}  colour {s['colour_shift']['est']:.2f}  "
          f"{s['render_s']['est']:.1f}s")
for k, v in paired.items():
    print(k, " ".join(f"{m} {q['est']:+.3f} [{q['lo']:+.3f},{q['hi']:+.3f}]" for m, q in v.items()))
print("decision:", json.dumps(decision, indent=1))
