"""Round 42 (CPU, on saved outputs): score the four control arms. FROZEN before any round 42 output exists.

Per held-out test scene x target x arm (fixed = round 40 outputs; depth; correct; oracle):
  blur       delivered additional blur, round 40 estimator, ALL behind tiles (primary).
  blur_odd   same on the odd-indexed tiles only (never used by the correct arm's feedback; secondary check).
  ok         |blur - target| <= max(2, 0.2 target); unmeasurable counts as a miss.
  subject    hf_ratio (subject-interior detail kept vs the input) and colour_shift (measures40.preservation);
             focus check = hf_ratio > far behind-tile retention (round 40 definition).
  runtime    render seconds (and total including measurement); n_renders.
Scene level first (mean over the 3 targets), then a percentile bootstrap over scenes (10,000, seed 0) for 95% CIs.

Decision rules:
  D1  depth vs fixed, target attainment (scene within-tolerance fraction, paired):
      SUBSTANTIAL if estimate >= +0.25 and CI lower bound > 0; IMPROVED if CI lower bound > 0 only; else NOT SHOWN.
  D2  subject not damaged (depth vs fixed, and correct vs fixed): hf_ratio paired CI lower bound >= -0.05.
  D3  correct vs depth: same thresholds as D1 (does one measured correction add?); also reported on odd tiles.
  D4  oracle = ceiling of what choosing K per image can give; reported, not a controller.
  D5  learned predictor (e.g. CNN) considered justified only if the depth arm misses >= 1/3 of cells AND >= 50% of
      the variance of its log(delivered / target) is between scenes (consistent per-scene error = predictable in
      principle). A necessary condition, not proof that a CNN would work.
  Determinism: fixed_check renders must match round 40 (max abs diff <= 2/255), otherwise the fixed arm is re-rendered
  in a follow-up and the comparison is reported as provisional.
usage (repo root): python quarter2/control42/score42.py <R42 out> <results dir>
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
S3 = os.environ.get("R40_S3", os.path.join(HERE, "..", "..", "outputs", "q2_compare40", "s3"))
OUT, RES = sys.argv[1], sys.argv[2]
os.makedirs(RES, exist_ok=True)
scenes, regions, review = C.load(DATA)
run = json.load(open(f"{OUT}/bd42.json"))
r40 = {(r["id"], r["file"].split("_t")[-1][:-4]): r for r in json.load(open(f"{S3}/bd_test.json"))["rows"]}
rng = np.random.default_rng(0)
ARMS = ["fixed", "depth", "correct", "oracle"]

det = [r for r in run["rows"] if r["arm"] == "fixed_check"]
det_ok = bool(det) and max(r["max_abs_diff_vs_r40"] for r in det) <= 2
items = [dict(r, root=OUT) for r in run["rows"] if r["arm"] in ARMS]
for i, t in sorted({(r["id"], r["target"]) for r in items}):
    f = r40[(i, f"{t:g}")]
    items.append(dict(id=i, target=t, arm="fixed", K=f["K"], file=f["file"], root=S3, render_s=f["seconds"],
                      total_s=f["seconds"], n_renders=1))

scored = []
for i in sorted({r["id"] for r in items}):
    x = C.img(f"{DATA}/input/{i}.png")
    H, W = x.shape[:2]
    rg = regions[i]
    subj, dep = C.subject(DATA, i, (H, W)), C.depth(DATA, i, (H, W))
    st_all, st_odd = BL.TileStack(x, rg["tiles"]), (BL.TileStack(x, rg["tiles"][1::2])
                                                     if len(rg["tiles"][1::2]) >= C.MIN_TILES else None)
    for r in items:
        if r["id"] != i:
            continue
        y = C.img(f"{r['root']}/{r['file']}")
        b = st_all.measure(y)["r"]
        bo = st_odd.measure(y)["r"] if st_odd else float("nan")
        pv = MS.preservation(x, y, subj)
        far = MS.boundary(x, y, subj, rg["tiles"], dep, rg["focus_disp"])["far_hf_retention"]
        t = r["target"]
        scored.append(dict({k: v for k, v in r.items() if k not in ("root", "grid_r")}, blur=b, blur_odd=bo,
                           ok=bool(np.isfinite(b) and abs(b - t) <= C.TOL(t)),
                           ok_odd=bool(np.isfinite(bo) and abs(bo - t) <= C.TOL(t)),
                           abs_err=abs(b - t) if np.isfinite(b) else float("nan"),
                           hf_ratio=pv["hf_ratio"], colour_shift=pv["colour_shift"],
                           focus_ok=bool(np.isfinite(far) and pv["hf_ratio"] > far)))
    print(i, "scored", flush=True)

ids = sorted({r["id"] for r in scored})


def per_scene(arm, key):
    return np.array([np.nanmean([float(r[key]) for r in scored if r["id"] == i and r["arm"] == arm]) for i in ids])


def boot(v, stat=np.mean):
    v = v[np.isfinite(v)]
    b = [stat(rng.choice(v, len(v))) for _ in range(10000)]
    return dict(n=int(len(v)), est=float(stat(v)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))


summ = {arm: {k: boot(per_scene(arm, k), np.median if k in ("abs_err", "hf_ratio", "colour_shift") else np.mean)
              for k in ["ok", "ok_odd", "abs_err", "hf_ratio", "colour_shift", "focus_ok", "render_s", "total_s",
                        "n_renders"]} for arm in ARMS}
pair = {}
for a, b in [("depth", "fixed"), ("correct", "fixed"), ("oracle", "fixed"), ("correct", "depth")]:
    pair[f"{a}-{b}"] = {k: boot(per_scene(a, k) - per_scene(b, k), np.mean if k.startswith("ok") else np.median)
                        for k in ["ok", "ok_odd", "abs_err", "hf_ratio"]}


def verdict(p):
    return ("SUBSTANTIAL" if p["est"] >= 0.25 and p["lo"] > 0 else "IMPROVED" if p["lo"] > 0 else "NOT SHOWN")


dep = [r for r in scored if r["arm"] == "depth"]
lr = np.array([np.log(r["blur"] / r["target"]) if np.isfinite(r["blur"]) and r["blur"] > 0 else np.nan for r in dep])
sid = np.array([r["id"] for r in dep])
fin = np.isfinite(lr)
means = {i: np.mean(lr[fin & (sid == i)]) for i in set(sid[fin])}
between = np.var([means[i] for i in sid[fin]])
share = float(between / np.var(lr[fin])) if fin.sum() > 1 else float("nan")
miss = 1 - np.mean([r["ok"] for r in dep])
decisions = dict(
    D1_depth_vs_fixed=verdict(pair["depth-fixed"]["ok"]),
    D2_subject_ok_depth=bool(pair["depth-fixed"]["hf_ratio"]["lo"] >= -0.05),
    D2_subject_ok_correct=bool(pair["correct-fixed"]["hf_ratio"]["lo"] >= -0.05),
    D3_correct_vs_depth=verdict(pair["correct-depth"]["ok"]),
    D3_correct_vs_depth_odd_tiles=verdict(pair["correct-depth"]["ok_odd"]),
    D5_learned_predictor_justified=bool(miss >= 1 / 3 and share >= 0.5),
    D5_depth_miss_rate=float(miss), D5_between_scene_share=share,
    determinism_ok=det_ok, determinism_max_abs_diff=max((r["max_abs_diff_vs_r40"] for r in det), default=None))
json.dump(dict(decisions=decisions, summary=summ, paired=pair, rows=scored), open(f"{RES}/score42.json", "w"),
          indent=1, default=float)
print(f"\n{len(ids)} scenes")
for arm in ARMS:
    s = summ[arm]
    print(f"{arm:8s} within tol {s['ok']['est']:.2f} [{s['ok']['lo']:.2f},{s['ok']['hi']:.2f}]  odd tiles "
          f"{s['ok_odd']['est']:.2f}  median |err| {s['abs_err']['est']:.2f}px  subject hf {s['hf_ratio']['est']:.2f}  "
          f"focus {s['focus_ok']['est']:.2f}  renders {s['n_renders']['est']:.1f}  total {s['total_s']['est']:.1f}s")
for k, p in pair.items():
    print(f"{k:16s} d(within tol) {p['ok']['est']:+.2f} [{p['ok']['lo']:+.2f},{p['ok']['hi']:+.2f}]  "
          f"d(hf) {p['hf_ratio']['est']:+.3f} [{p['hf_ratio']['lo']:+.3f},{p['hf_ratio']['hi']:+.3f}]")
print("decisions:", json.dumps(decisions, indent=1))
