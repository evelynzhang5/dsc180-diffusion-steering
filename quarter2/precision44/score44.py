"""Round 44 (CPU, on saved outputs): fp16 vs fp32 PISA at identical inputs and settings. FROZEN before any output.

Round 40 estimator and measures throughout (blur40.TileStack on the frozen behind tiles; measures40.preservation).
Part A, development curves (8 round 40 mapping scenes x K_DEV, both modes):
  r(K)        delivered additional blur, all behind tiles.
  continuity  largest jump |r(K_j+1) - r(K_j)| between consecutive grid K in the weak range (both values <= 8 px);
              a scene counts as JUMPING if that jump is > 3 px (round 21's fp16 staircase skipped ~3 px).
  reach_4     some grid K delivers 4 +- TOL(4) = 2 px.
  determinism fp16 renders at round 40's grid K vs round 40's bd_dev outputs (max |diff| <= 2/255).
Part B, test diagnostic (24 round 42 test scenes x 4/8/12 px at round 42's depth-rule K; already used scenes):
  per cell and mode: blur, within tolerance, |error|, subject hf_ratio and colour_shift vs the input.
  fp16 = round 42's depth-arm outputs (3 scenes re-rendered as a check, max |diff| <= 2/255).
  Paired scene-level differences fp32 - fp16 (bootstrap over scenes, 10,000, seed 0): within tolerance per target and
  overall, and subject hf_ratio.
Reading (descriptive, frozen; no controller is chosen here):
  4 px precision effect at identical settings: LARGE if fp32 - fp16 within-tolerance at 4 px >= +0.25 with CI lower
  bound > 0; PRESENT if only the lower bound > 0; NOT SHOWN otherwise. Because fp32 can shift the whole response, NOT
  SHOWN does not mean fp32 is ineffective: recalibration on fp32 development curves is the next round's job either way.
  Continuity: JUMPING scenes in fp16 vs fp32. Subject: paired hf_ratio difference with CI (additional subject change).
usage (repo root): python quarter2/precision44/score44.py <R44 out> <R42 run> <results dir>
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

DATA = os.environ.get("R40_DATA", os.path.join(HERE, "..", "..", "outputs", "q2_compare40", "data"))
OUT, R42, RES = sys.argv[1:4]
os.makedirs(RES, exist_ok=True)
scenes, regions, review = C.load(DATA)
run = json.load(open(f"{OUT}/bd44.json"))
rng = np.random.default_rng(0)

rows = [dict(r, root=OUT) for r in run["rows"] if not (r["part"] == "test" and r["mode"] == "fp16")]
for r in run["rows"]:                                       # fp16 test cells = round 42 depth outputs
    if r["part"] == "test" and r["mode"] == "fp32":
        rows.append(dict(part="test", id=r["id"], target=r["target"], K=r["K"], mode="fp16",
                         file=f"depth/{r['id']}_t{r['target']:g}.png", root=R42))
checks16 = [r for r in run["rows"] if r["part"] == "test" and r["mode"] == "fp16"]

scored = []
for i in sorted({r["id"] for r in rows}):
    x = C.img(f"{DATA}/input/{i}.png")
    H, W = x.shape[:2]
    st = BL.TileStack(x, regions[i]["tiles"])
    subj = C.subject(DATA, i, (H, W))
    for r in [r for r in rows if r["id"] == i]:
        y = C.img(f"{r['root']}/{r['file']}")
        b = st.measure(y)["r"]
        pv = MS.preservation(x, y, subj)
        s = {k: v for k, v in r.items() if k != "root"}
        s.update(blur=b, hf_ratio=pv["hf_ratio"], colour_shift=pv["colour_shift"])
        if r["part"] == "test":
            t = r["target"]
            s.update(ok=bool(np.isfinite(b) and abs(b - t) <= C.TOL(t)),
                     abs_err=abs(b - t) if np.isfinite(b) else float("nan"))
        scored.append(s)
    print(i, "scored", flush=True)


def boot(v, stat=np.mean):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, est=float("nan"), lo=float("nan"), hi=float("nan"))
    b = [stat(rng.choice(v, len(v))) for _ in range(10000)]
    return dict(n=int(len(v)), est=float(stat(v)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))


# part A
dev = [s for s in scored if s["part"] == "dev"]
curves = {}
for i in sorted({s["id"] for s in dev}):
    curves[i] = {}
    for mode in ["fp16", "fp32"]:
        pts = sorted((s["K"], s["blur"]) for s in dev if s["id"] == i and s["mode"] == mode)
        Ks, rs = [p[0] for p in pts], [p[1] for p in pts]
        jumps = [abs(b - a) for a, b in zip(rs, rs[1:]) if np.isfinite(a) and np.isfinite(b) and max(a, b) <= 8]
        curves[i][mode] = dict(K=Ks, r=rs, max_weak_jump=max(jumps) if jumps else float("nan"),
                               jumping=bool(jumps and max(jumps) > 3),
                               reach_4=bool(any(np.isfinite(v) and abs(v - 4) <= C.TOL(4) for v in rs)))
det_dev = [s["max_abs_diff_vs_r40"] for s in dev if "max_abs_diff_vs_r40" in s]
dev_diff = {}
for K in sorted({s["K"] for s in dev}):
    d = [next(s["blur"] for s in dev if s["id"] == i and s["K"] == K and s["mode"] == "fp32")
         - next(s["blur"] for s in dev if s["id"] == i and s["K"] == K and s["mode"] == "fp16") for i in curves]
    dev_diff[f"{K:g}"] = dict(median_fp32_minus_fp16=float(np.nanmedian(d)), n=int(np.isfinite(d).sum()))

# part B
test = [s for s in scored if s["part"] == "test"]
ids = sorted({s["id"] for s in test})


def cell(i, t, mode, key):
    v = [s[key] for s in test if s["id"] == i and s["target"] == t and s["mode"] == mode]
    return float(v[0]) if v else float("nan")


def scene_mean(mode, key, ts):
    return np.array([np.nanmean([cell(i, t, mode, key) for t in ts]) for i in ids])


T = [4.0, 8.0, 12.0]
summary, paired = {}, {}
for mode in ["fp16", "fp32"]:
    summary[mode] = {f"ok_{t:g}": boot(scene_mean(mode, "ok", [t])) for t in T}
    summary[mode]["ok_all"] = boot(scene_mean(mode, "ok", T))
    summary[mode]["abs_err"] = boot(scene_mean(mode, "abs_err", T), np.median)
    summary[mode]["hf_ratio"] = boot(scene_mean(mode, "hf_ratio", T), np.median)
for t in T:
    paired[f"ok_{t:g}"] = boot(scene_mean("fp32", "ok", [t]) - scene_mean("fp16", "ok", [t]))
paired["ok_all"] = boot(scene_mean("fp32", "ok", T) - scene_mean("fp16", "ok", T))
paired["hf_ratio"] = boot(scene_mean("fp32", "hf_ratio", T) - scene_mean("fp16", "hf_ratio", T), np.median)
paired["colour_shift"] = boot(scene_mean("fp32", "colour_shift", T) - scene_mean("fp16", "colour_shift", T), np.median)
p4 = paired["ok_4"]
reading = dict(
    precision_effect_4px="LARGE" if p4["est"] >= 0.25 and p4["lo"] > 0 else "PRESENT" if p4["lo"] > 0 else "NOT SHOWN",
    jumping_scenes_fp16=sum(c["fp16"]["jumping"] for c in curves.values()),
    jumping_scenes_fp32=sum(c["fp32"]["jumping"] for c in curves.values()), n_dev=len(curves),
    reach_4_fp16=sum(c["fp16"]["reach_4"] for c in curves.values()),
    reach_4_fp32=sum(c["fp32"]["reach_4"] for c in curves.values()),
    determinism_dev_max=max(det_dev) if det_dev else None,
    determinism_test_max=max((r["max_abs_diff_vs_r42"] for r in checks16), default=None),
    determinism_ok=bool(det_dev and max(det_dev) <= 2 and (not checks16 or max(r["max_abs_diff_vs_r42"]
                                                                               for r in checks16) <= 2)),
    pisa_calls=run["calls"])
json.dump(dict(reading=reading, curves=curves, dev_fp32_minus_fp16=dev_diff, summary=summary, paired=paired,
               rows=scored), open(f"{RES}/score44.json", "w"), indent=1, default=float)
print("\nPart A: weak-range curves (delivered px at K)")
for i, c in curves.items():
    for mode in ["fp16", "fp32"]:
        print(f"{i:13s} {mode} " + " ".join(f"{k:g}:{v:.1f}" for k, v in zip(c[mode]["K"], c[mode]["r"]))
              + f"   max weak jump {c[mode]['max_weak_jump']:.1f} reach4 {c[mode]['reach_4']}")
print("median fp32 - fp16 by K:", {k: round(v["median_fp32_minus_fp16"], 2) for k, v in dev_diff.items()})
print("\nPart B (diagnostic, used scenes)")
for mode in ["fp16", "fp32"]:
    s = summary[mode]
    print(mode, " ".join(f"{k} {v['est']:.2f} [{v['lo']:.2f},{v['hi']:.2f}]" for k, v in s.items()))
for k, v in paired.items():
    print(f"fp32-fp16 {k}: {v['est']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}]")
print("reading:", json.dumps(reading, indent=1))
