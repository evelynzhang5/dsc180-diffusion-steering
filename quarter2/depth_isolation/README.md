# Q2 round 23: does the depth map change BokehDiff's blur when the full blur map is held fixed?

**Question.** BokehDiff uses two inputs:
- the **blur map**, which sets how far features may spread;
- the **disparity (depth) map**, which in the released code enters only PISA's occlusion test.

Holding the whole blur map, the image and the random state fixed, does changing the depth map change the delivered blur inside large flat regions, away from boundaries? If so, does that change travel through occlusion?

**Background.**
- Round 22's "disparity map" lead was retracted: it compared different patches, and the same-cell comparison shows 0 %.
- What is still unexplained from rounds 19 and 22: at the same request, blur is lower when the focus sits near the end of the disparity range (d_f 0.2 / 0.8) than at 0.5.
- One code detail to check: PISA's occlusion test divides by (1 − disparity). After BokehDiff's normalization the nearest surface always has disparity exactly 1.

**Status.** Plan and rules fixed before running.

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

Pending.
