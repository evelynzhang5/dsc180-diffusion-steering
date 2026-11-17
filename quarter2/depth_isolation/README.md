# Q2 round 23: does the depth map change BokehDiff's blur when the full blur map is held fixed?

**Question.** BokehDiff uses two inputs:
- the **blur map**, which sets how far features may spread;
- the **disparity (depth) map**, which in the released code enters only PISA's occlusion test.

Holding the whole blur map, the image and the random state fixed, does changing the depth map change the delivered blur inside large flat regions, away from boundaries? If so, does that change travel through occlusion?

**Background.**
- Round 22's "disparity map" lead was retracted: it compared different patches, and the same-cell comparison shows 0 %.
- What is still unexplained from rounds 19 and 22: at the same request, blur is lower when the focus sits near the end of the disparity range (d_f 0.2 / 0.8) than at 0.5.
- One code detail to check: PISA's occlusion test divides by (1 − disparity). After BokehDiff's normalization the nearest surface always has disparity exactly 1.

**Status.** Run Nov 17 (pod `q2-isolation`, 52 images in 2 min). All checks passed, including the key one: **every frozen-occlusion variant reproduces the true-geometry output exactly** (max difference 0), so in this code depth reaches the output only through the occlusion mask. Plan and rules were committed (93e42e4) before the run.

**Outcome Z1: at a fixed blur map, changing the depth map does not change interior blur.** Every variant changes 0–12.5 % of cells; the threshold was 50 %.

## Test (`run23.py`)

**Setup:**
- PISA attention in fp32 (as in round 22); seed 0 for everything;
- scenes S3 and S6 (round 22's two weakest held-out scenes);
- blur maps of R2 P1 (requests 7.8 / 9.7 / 11.6 px) and R3 P1 (13.6 / 15.5 / 17.4 px), both with d_f = 0.5 and anchors at 19.4 px.

**Depth variants, blur map unchanged:**

| variant | disparity map |
|---|---|
| V1 | true geometry (spans 0–1) |
| V2 | flat 0.5 |
| V3 | gaps halved, centred 0.5 (0.25–0.75) |
| V4 | gaps halved, centred 0.3 (0.05–0.55) |
| V5 | gaps halved, centred 0.7 (0.45–0.95) |
| V6 | true, but capped at 0.99 (only the near anchor changes) |

V3–V5 share the same relative structure and differ only in absolute disparity.

**Occlusion handling:**
- **native:** every variant, with the mask PISA computes;
- **frozen:** V2–V6 with V1's recorded mask substituted;
- **zero:** V1 with the mask set to 0;
- plus a V1 repeat.

That makes 52 images.

**Logged:**
- hashes of the input image, the full disparity + blur tensor, and each PISA call's occlusion mask;
- non-finite entries in the occlusion mask;
- per cell (dropout-free recomputation): attention mass leaving self, and attention mass removed by occlusion.

## Rules (fixed now)

**Gate:**
- estimator validation passes;
- the V1 repeat is identical, with identical logged inputs and masks;
- no NaN or black outputs;
- **every "frozen" variant equals V1** (≤ 1/255). This checks that the disparity map has no path to the output other than the occlusion mask.

**Measure:** r̂ of the 8 non-focus cells (interiors, ≥ 48 px from edges). A cell "changes" if |r̂ − r̂_V1| > max(1 px, 10 %); round 22's seed noise was a median of 0.4 px.

| outcome | rule | next |
|---|---|---|
| Z1 | no variant changes ≥ 50 % of cells | no depth effect on interior blur at a fixed blur map. The focus-position difference in rounds 19 and 22 is then not a depth effect, and points to the different blur maps / patches in those runs |
| Z2 | some variant changes ≥ 50 % of cells | depth changes interior blur, and by the gate it does so through occlusion. Interior occlusion of same-plane contributions is a candidate physically incorrect response, to be checked against a renderer at boundaries next. Occlusion removal is **not** normalized away |

**Also reported:**
- V1 with occlusion zeroed: how much the true-geometry occlusion contributes at all;
- the absolute-disparity contrast V4 vs V5;
- the near-anchor (disparity 1) cell, true vs capped at 0.99;
- the in-focus MAE change against the input.

**Not done here:** boundaries against a renderer (the reviewer's step 4). That is the next step only if Z2.

## Results

Files:
- `results/isolation.json`;
- `results/cells.json`;
- `results/isolation_sheet.png`: S3 R2 and S6 R3; input, V1, V1 with occlusion zeroed, V2–V6.

Images are in `outputs/q2_depth_isolation/`.

### Gate: passed

- estimator validation passed;
- the V1 repeat is identical, with identical input-tensor and occlusion-mask hashes;
- 52 outputs, none bad;
- **all 20 frozen variants equal V1 exactly.**

### Depth variants against true geometry (32 non-focus cells each; native occlusion)

| variant | cells changed > max(1 px, 10 %) | median change in r̂ | 10th–90th percentile |
|---|---|---|---|
| V2 flat 0.5 | 4 / 32 (12.5 %) | +6.1 % | −1.4 … +10.6 % |
| V3 half, centred 0.5 | 0 / 32 | +3.1 % | −0.8 … +6.0 % |
| V4 half, centred 0.3 | 0 / 32 | +3.9 % | 0.0 … +7.3 % |
| V5 half, centred 0.7 | 0 / 32 | +1.3 % | 0.0 … +2.3 % |
| V6 true, capped at 0.99 | 0 / 32 | 0.0 % | 0.0 … +0.1 % |
| V1 with occlusion **zeroed** | 4 / 32 (12.5 %) | +6.1 % | −1.4 … +10.6 % |

**What the variants show:**
- **Absolute disparity barely matters.** V4 (centred 0.3) and V5 (centred 0.7) share the same structure and differ from true geometry by +3.9 % and +1.3 %. That is far too small to explain the focus-position differences of rounds 19 and 22 (e.g. 6.3 vs 8.4 px at the same request). **Those differences therefore do not come from depth at a fixed blur map.** They point to the different blur maps and patches in those runs; this round did not test that directly.
- **Occlusion's whole contribution is small.** With true geometry, occlusion removes a median of 4.0 % of each non-focus key's attention mass: 0.7 % for cells in front of the focus, 6.1 % behind, and up to 8.8 % for the far anchor. Removing occlusion entirely raises blur by a median of 6 %. Flat depth (no occlusion) gives exactly the same numbers, as it should.
- **The disparity = 1 edge case is harmless here.** No occlusion entry was non-finite. The near anchor at disparity 1 has 0 occlusion removal, and capping it at 0.99 changes nothing (identical r̂).
- **In-focus detail:** MAE against the input changes by at most 0.03/255 across the variants.

### Interpretation

- **Depth is not the mechanism behind the remaining larger-blur error.** With the blur map fixed, depth reaches the output only through occlusion. Occlusion moves interior blur by about 6 % at most, and absolute depth by about 1–4 %.
- **Whether occlusion's removal is physically right is a separate question.** It removes about 6 % of behind-focus interior attention, even inside large flat regions where nothing should block that plane's own light. The effect on blur is small (~6 %), and the reviewer's boundary-vs-renderer check would be needed to call it wrong. It is not pursued now, because it does not explain the transfer error.
- **What remains:** round 22's shortfall on larger blur is a transfer problem. The same blur request produces different blur on different patches and scenes (held-out about 10 % lower). It is not explained by depth, focus side, or surrounding blur. Patch content is the remaining candidate; rounds 19 and 21 already showed patch-to-patch spread at a fixed request.

**Not shown:**
- boundaries (the renderer comparison);
- other scenes;
- K other than 40.
