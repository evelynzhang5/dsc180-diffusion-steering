"""Round 42 (CPU): fit and FREEZE the depth-based K rule on DEVELOPMENT scenes only, then write the test plan.

Depth gap of a scene (inputs only, before any rendering): BokehDiff's own preprocessing of the stored depth map
(dataset.py min-max + 3x3 erode/dilate clip), focus = median inside the mask (> 0.5), gap = median over the frozen
BEHIND tiles of each tile's median |depth - focus| (768-px scale). Same as round 40 post40_gap.py.

Data for the fit: round 40's development responses (mapping.json "responses": delivered blur of the 8 mapping scenes at
K = 4 ... 64, all behind tiles). Only points inside the working range 2 <= r <= 20 px are used (targets are 4-12 px;
below 2 px the estimator sits at its floor, above 20 px it nears its 24 px cap).
Two candidate forms, chosen by leave-one-scene-out median |prediction error| on the development points:
  P  r = a * K * gap                (inverse K = t / (a gap))
  L  r = b * (K * gap - c)          (inverse K = (t / b + c) / gap)
Frozen K = inverse, clipped to [4, 64] (the range BokehDiff was run at in round 40; clipped plans are flagged).
The one-step correction (arm 3) uses the same form's slope: K2 = K1 + (t - r1) / (slope * gap), clipped likewise.

DISCLOSURE: the proportional form P was looked at post hoc on round 40's TEST outputs (post40_gap.py: correlation 0.88).
The form choice here is made on development points only, and the test scenes are the same 24 held-out scenes.
usage (repo root): python quarter2/control42/fit42.py   -> quarter2/control42/results/rule42.json, plan42.json
"""
import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sw
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compare40"))
import common40 as C  # noqa: E402

DATA = os.environ.get("R40_DATA", os.path.join(HERE, "..", "..", "outputs", "q2_compare40", "data"))
MAP40 = os.path.join(HERE, "..", "compare40", "results", "stage3", "mapping.json")
RES = os.path.join(HERE, "results")
K_MIN, K_MAX, R_LO, R_HI = 4.0, 64.0, 2.0, 20.0


def _mm(d, fn):
    return fn(sw(np.pad(d, 1, mode="edge"), (3, 3)), axis=(-2, -1))


def gap(i, regions):
    d = np.load(f"{DATA}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = _mm(_mm(d, np.min), np.max)
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.open(f"{DATA}/mask/{i}.png").resize((d.shape[1], d.shape[0]), resample=Image.LANCZOS)
                 .convert("L")) / 255.0
    f = np.median(d[m > 0.5])
    H, W = regions[i]["size"]
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W, H), resample=Image.BILINEAR))
    return float(np.median([np.median(np.abs(d[t["y"]:t["y"] + 64, t["x"]:t["x"] + 64] - f))
                            for t in regions[i]["tiles"]]))


def fit(form, pts):
    x = np.array([k * g for k, g, _ in pts])
    r = np.array([v for _, _, v in pts])
    if form == "P":
        return dict(a=float(x @ r / (x @ x)))
    A = np.stack([x, -np.ones_like(x)], 1)
    b, bc = np.linalg.lstsq(A, r, rcond=None)[0]
    return dict(b=float(b), c=float(bc / b))


def predict(form, p, K, g):
    return p["a"] * K * g if form == "P" else p["b"] * (K * g - p["c"])


def inverse(form, p, t, g):
    return t / (p["a"] * g) if form == "P" else (t / p["b"] + p["c"]) / g


def slope(form, p):
    return p["a"] if form == "P" else p["b"]


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    scenes, regions, review = C.load(DATA)
    mp = json.load(open(MAP40))["bokehdiff"]
    dev = mp_scenes = sorted(mp["responses"])
    assert all(scenes[i]["split"] == "dev" for i in dev)
    gaps = {i: gap(i, regions) for i in dev}
    pts = {i: [(float(K), gaps[i], v["r"]) for K, v in mp["responses"][i].items()
               if np.isfinite(v["r"]) and R_LO <= v["r"] <= R_HI] for i in dev}
    cv = {}
    for form in ["P", "L"]:
        errs = []
        for i in dev:
            p = fit(form, [q for j in dev if j != i for q in pts[j]])
            errs += [abs(predict(form, p, K, g) - r) for K, g, r in pts[i]]
        cv[form] = dict(loso_median_abs_err=float(np.median(errs)), n_points=len(errs))
    form = min(cv, key=lambda f: cv[f]["loso_median_abs_err"])
    params = fit(form, [q for i in dev for q in pts[i]])
    rule = dict(form=form, params=params, cv=cv, k_min=K_MIN, k_max=K_MAX, r_range=[R_LO, R_HI],
                dev_scenes=dev, dev_gaps=gaps, n_points=sum(len(v) for v in pts.values()))
    print("dev gaps:", {i: round(g, 3) for i, g in gaps.items()})
    print("LOSO:", cv, "-> form", form, params)

    test = [i for i, s in sorted(scenes.items()) if s["kind"] == "gen" and s["split"] == "test"
            and s["category"] != "existing_blur" and i not in review["mask_failure"]
            and i not in review["measurement_flag"]]
    fixed = {f"{t:g}": mp["gen"][f"{t:g}"]["setting"] for t in C.TARGETS}
    plan = {}
    for i in test:
        g = gap(i, regions)
        plan[i] = dict(gap=g, targets={})
        for t in C.TARGETS:
            k = inverse(form, params, t, g)
            kc = float(np.clip(k, K_MIN, K_MAX))
            plan[i]["targets"][f"{t:g}"] = dict(fixed_K=fixed[f"{t:g}"], depth_K=kc, depth_K_unclipped=float(k),
                                                clipped=bool(kc != k))
    json.dump(rule, open(f"{RES}/rule42.json", "w"), indent=1)
    json.dump(dict(test_scenes=test, plan=plan), open(f"{RES}/plan42.json", "w"), indent=1)
    ks = [v["depth_K"] for p in plan.values() for v in p["targets"].values()]
    print(f"test plan: {len(test)} scenes; depth K range {min(ks):.1f}-{max(ks):.1f}; clipped "
          f"{sum(v['clipped'] for p in plan.values() for v in p['targets'].values())}/{len(ks)}")
