# Q2 round 24: does content-dependent attention make the same blur request produce different blur, and can changing it make control more consistent?

**Question.** Round 22 left larger blur (≥ 6 px) transferring imperfectly to new patches (67 %). Round 23 ruled out depth.
- Is the remaining patch-to-patch variation tied to PISA's content-similarity (query–key) weights?
- Does weakening them make blur control more consistent?
- How does that compare with a simple calibration that uses patch statistics?

**Status.** Plan and rules fixed before running.

## Design (`patches.py`, `run24.py`)

**Patches.** Two 171 × 171 crops per source, chosen as the two non-overlapping crops with the most image gradient, from 12 offline scikit-image sources. **Whole sources are held out.**

| split | sources |
|---|---|
| development | grass, brick, camera, coffee, immunohistochemistry, page |
| test | gravel, astronaut, chelsea, coins, rocket, text |

Patch statistics: log luminance std, log mean gradient, high-frequency energy share.

**Layout.** Round 19's 3 × 3 grid.
- Centre: the focus cell (a fixed brick crop).
- Corners: anchors (a fixed filler).
- The other 6 cells hold 6 patches, one from each source in the split.
- **Content vs position:** every free cell gets the same request. The 6 patches rotate through the 6 positions, all 6 positions on test and 3 on development, so a patch effect can be separated from a position effect.
- Geometry: d_f = 0.5, front/behind pairs at ±gap, and the map spans [0, 1].
- Round 23 showed that depth hardly matters at a fixed blur map, so corrections change only the blur map.

**Attention variants.** PISA in fp32 throughout. Only the query–key logits are scaled by τ; the optical gate, normalization, occlusion and dropout are untouched.

| variant | τ | meaning |
|---|---|---|
| A0 | 1 | unchanged |
| A1 | 0.5 | content similarity weakened |
| A2 | 0 | removed: uniform within the optical support |

**Stages:**
1. **Development.** Every variant gets the same budget: 12 dev patches × 3 positions × 8 requests (5.8–19.4 px).
2. **Calibrations**, frozen to `calib.json` before any test run:
   - **C1**, per variant: monotone request → radius curve;
   - **CP**, for A0: log r = a + b log r_req + c · z(patch statistics).
3. **Test.** 12 held-out-source patches × 6 positions × targets 7.8 / 11.6 / 15.5 px.
   - Raw requests for A0, A1, A2.
   - Corrected requests, actually rerun: A0-C1, A0-CP, A1-C1, A2-C1.
   - 216 cells per method.

**Protection 2 (measurement).** Known blur is applied to these exact patches in these layouts:
- disk r ∈ {0, 6, 8, 12, 16, 20}: must be recovered within max(1 px, 10 %) on every patch; this is part of the gate;
- Gaussian σ ∈ {3, 5, 7, 9}: the across-patch spread (CV) of the disk-fit reading is reported.

**Recorded:**
- per raw test run (position 0): attention mass leaving self and key-distribution entropy, per cell;
- focus-cell MAE against the input;
- the normalized residual of the disk fit, as a blur-quality check.

## Decision rules (fixed now)

**Gate:**
- the disk check passes on all 24 patches;
- the repeat is identical;
- no NaN or black outputs.

**Agreement:** |r̂ − t| ≤ max(2 px, 20 %); above-range never agrees.

| outcome | rule | meaning |
|---|---|---|
| M | median across-patch CV of known-Gaussian readings ≥ ½ × the median across-patch CV of A0's raw outputs | measurement contributes; improve it before model claims |
| E1 | A0-CP ≥ 80 % and ≥ A0-C1 + 5 points | a simple patch-statistics correction suffices for this problem |
| E2 (per A1, A2) | all four hold: (a) its own C1 agreement ≥ max(A0-C1, A0-CP) + 5 points; (b) across-patch CV at fixed raw request reduced ≥ 20 % relative to A0 (median over targets); (c) focus MAE not worse by > 0.5/255; (d) disk-fit residual not worse by > 20 % | evidence to develop the attention-steering method |
| E3 (per A1, A2) | attention changed (entropy or leave mass ≥ 10 % relative), but CV reduction < 20 % and agreement < best simple + 5 points | the proposed mechanism lacks support; look at another component before building a controller |

Several outcomes can hold at once; all are reported. Also reported:
- agreement by target and by held-out source;
- the patch vs position share of variance;
- how well CP and C1 *predict* raw outputs, separately from how well they *control*;
- requests that were clamped or unreachable.

**Not claimed:**
- the τ intervention is a probe, not the proposed method;
- one model, K = 40, 512², planar patches, interiors only.

## Results

Pending.
