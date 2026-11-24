"""Round 41 step 1 (CPU, on saved outputs): score the three depth conditions and apply the decision rule.
FROZEN before any round 41 output exists.

Per output (round 40 measures, unchanged):
  halo_sharp  measures40.boundary, band 2-12 px outside the subject, depth-gated on the STORED depth map and the round
              40 focus (the same band pixels in every condition), minus the far behind-tile retention.
  blur        background blur on the frozen behind tiles (blur40).
  hf_ratio    subject-interior high-frequency retention vs the input (measures40.preservation).
Scene value = mean over the 3 targets; summaries are medians over the 8 scenes, with paired sign counts.

Decision rule (two_layer B vs original A; hard_edge C is reported only):
  0. Is there a failure to explain? If median halo(A) < 0.10, report "no halo on the development scenes" and stop.
  1. INPUT: median halo(B) <= 0.5 * median halo(A) AND halo(B) < halo(A) in >= 6 of 8 scenes
     -> the halo follows the depth input. That is EITHER correct rendering of background that really is near the
        focus depth (visual check before launch: hair_6's wall, highlights_6's crowd) OR depth-map smoothing at the
        edge (hard_edge C separates these: C close to B = smoothing, C close to A = genuine depth). Either way it
        is not a model failure, and round 41 needs a different failure for an in-model change.
  2. MODEL: median halo(B) >= 0.75 * median halo(A) OR halo(B) < halo(A) in <= 4 of 8 scenes
     -> the halo survives a perfectly flat background request; it is made inside the model; go to an in-model change.
  3. otherwise MIXED: both contribute; report both.
  If B also changes background blur by more than TOL at the tiles, that is reported next to the verdict.
usage (repo root): python quarter2/halo41/measure41.py <R41 out dir> <results dir>
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
OUT, RES = sys.argv[1], sys.argv[2]
os.makedirs(RES, exist_ok=True)
scenes, regions, review = C.load(DATA)
rows = json.load(open(f"{OUT}/bd41.json"))["rows"]
CONDS = ["original", "two_layer", "hard_edge"]

scored = []
for i in sorted({r["id"] for r in rows}):
    x = C.img(f"{DATA}/input/{i}.png")
    H, W = x.shape[:2]
    subj, dep, rg = C.subject(DATA, i, (H, W)), C.depth(DATA, i, (H, W)), regions[i]
    st = BL.TileStack(x, rg["tiles"])
    for r in rows:
        if r["id"] != i:
            continue
        y = C.img(f"{OUT}/{r['file']}")
        bd = MS.boundary(x, y, subj, rg["tiles"], dep, rg["focus_disp"])
        scored.append(dict(r, halo_sharp=bd["halo_sharp"], halo_px=bd["halo_px"], far=bd["far_hf_retention"],
                           blur=st.measure(y)["r"], hf_ratio=MS.preservation(x, y, subj)["hf_ratio"]))
    print(i, "scored", flush=True)

ids = sorted({r["id"] for r in scored})


def scene(metric, cond):
    return {i: float(np.nanmean([r[metric] for r in scored if r["id"] == i and r["cond"] == cond])) for i in ids}


summ = {}
for m in ["halo_sharp", "blur", "hf_ratio"]:
    summ[m] = {c: dict(median=float(np.nanmedian(list(scene(m, c).values()))), per_scene=scene(m, c)) for c in CONDS}
hA, hB, hC = (scene("halo_sharp", c) for c in CONDS)
fin = [i for i in ids if np.isfinite(hA[i]) and np.isfinite(hB[i])]
mA, mB = np.median([hA[i] for i in fin]), np.median([hB[i] for i in fin])
lower = sum(hB[i] < hA[i] for i in fin)
n = len(fin)
if mA < 0.10:
    verdict = "NO_FAILURE"
elif mB <= 0.5 * mA and lower >= 6 * n / 8:
    verdict = "INPUT"
elif mB >= 0.75 * mA or lower <= 4 * n / 8:
    verdict = "MODEL"
else:
    verdict = "MIXED"
bA, bB = summ["blur"]["original"]["per_scene"], summ["blur"]["two_layer"]["per_scene"]
blur_shift = {i: bB[i] - bA[i] for i in ids}
res = dict(verdict=verdict, n_scenes=n, median_halo=dict(original=float(mA), two_layer=float(mB),
           hard_edge=float(np.median([hC[i] for i in fin if np.isfinite(hC[i])]))),
           two_layer_lower_in=lower, blur_shift_two_layer_minus_original=blur_shift,
           blur_shift_beyond_tol=[i for i in ids if np.isfinite(blur_shift[i]) and abs(blur_shift[i]) > C.TOL(bA[i])],
           summary=summ, rows=scored)
json.dump(res, open(f"{RES}/measure41.json", "w"), indent=1, default=float)
print("\nscene         halo A   halo B   halo C   blur A  blur B   hf A   hf B")
for i in ids:
    print(f"{i:13s} {hA[i]:7.3f}  {hB[i]:7.3f}  {hC[i]:7.3f}   {bA[i]:5.1f}  {bB[i]:5.1f}   "
          f"{summ['hf_ratio']['original']['per_scene'][i]:.2f}  {summ['hf_ratio']['two_layer']['per_scene'][i]:.2f}")
print(f"median halo: original {mA:.3f}  two_layer {mB:.3f}  hard_edge {res['median_halo']['hard_edge']:.3f}; "
      f"two_layer lower in {lower}/{n}")
print("blur shift beyond tolerance (two_layer vs original):", res["blur_shift_beyond_tol"])
print("VERDICT:", verdict)
