"""Round 45 (CPU, on the pod after preparation, before any rendering): eligibility, tile split and the K plan.

Regions: round 40's regions40.py run unchanged on the fresh data (behind tiles: >= 24 px outside the subject, >= 90% of
pixels >= 0.15 from the focus disparity, textured, behind the focus; up to 8, most textured first).
Tile split (frozen): SELECTION tiles = even-indexed behind tiles, EVALUATION tiles = odd-indexed. No arm in this round
measures its own output to choose K, so the selection tiles are unused (reserved); every arm is scored on the evaluation
tiles. No fallback: a scene with < 2 evaluation tiles is not scored (reported in coverage).
Eligible (automatic): >= 4 behind tiles (so >= 2 per subset). The input-only review (mask failures) is applied at scoring.
K per scene and target (4, 8, 12 px):
  fixed     round 40's single setting per target (mapping.json), secondary baseline;
  linear    round 42's frozen offset rule (rule42.json): K = (t / b + c) / gap, clipped to [4, 64];
  monotone  round 45's frozen isotonic curve (rule45.json, fit45.plan_K), clipped to [4, 64].
gap = fit42.gap definition (BokehDiff preprocessing of the stored depth, median over ALL behind tiles: inputs only).
usage: R40_DATA=<fresh data> python plan45.py <fresh data>   -> <data>/plan45.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1]
os.environ["R40_DATA"] = DATA
sys.path.insert(0, os.path.join(HERE, "..", "control42"))
sys.path.insert(0, HERE)
import fit42  # noqa: E402
import fit45  # noqa: E402

regions = json.load(open(f"{DATA}/regions40.json"))
r42 = json.load(open(os.path.join(HERE, "..", "control42", "results", "rule42.json")))
r45 = json.load(open(os.path.join(HERE, "results", "rule45.json")))
mp = json.load(open(os.path.join(HERE, "..", "compare40", "results", "stage3", "mapping.json")))["bokehdiff"]["gen"]
b, c = r42["params"]["b"], r42["params"]["c"]
plan, cov = {}, dict(n_inputs=len(regions), eligible=[], too_few_tiles=[])
for i in sorted(regions):
    rg = regions[i]
    if len(rg["tiles"]) < 4 or not rg["blur_measurable"]:
        cov["too_few_tiles"].append(i)
        continue
    cov["eligible"].append(i)
    g = fit42.gap(i, regions)
    p = dict(gap=g, select_tiles=rg["tiles"][0::2], eval_tiles=rg["tiles"][1::2], targets={})
    for t in [4.0, 8.0, 12.0]:
        kl = (t / b + c) / g
        km, info = fit45.plan_K(r45, t, g)
        p["targets"][f"{t:g}"] = dict(fixed=float(mp[f"{t:g}"]["setting"]), linear=float(np.clip(kl, 4, 64)),
                                      linear_clipped=bool(not 4 <= kl <= 64), monotone=km, monotone_info=info)
    plan[i] = p
    print(i, f"gap {g:.3f}", {t: {a: round(v[a], 1) for a in ("fixed", "linear", "monotone")}
                               for t, v in p["targets"].items()}, flush=True)
json.dump(dict(coverage=cov, plan=plan), open(f"{DATA}/plan45.json", "w"), indent=1)
print(f"plan: {len(cov['eligible'])} eligible of {cov['n_inputs']}; too few tiles: {cov['too_few_tiles']}", flush=True)
