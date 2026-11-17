# Q2 round 24: does content-dependent attention make the same blur request produce different blur, and can changing it make control more consistent?

**Question.** Round 22 left larger blur (≥ 6 px) transferring imperfectly to new patches (67 %). Round 23 ruled out depth.
- Is the remaining patch-to-patch variation tied to PISA's content-similarity (query–key) weights?
- Does weakening them make blur control more consistent?
- How does that compare with a simple calibration that uses patch statistics?

**Status.** Run Nov 17 (pod `q2-attn`, 397 images in 12 min). The test's own checks all passed. Plan and rules were committed (5c826c2) before the run. Pre-set outcomes:
- **M:** measurement contributes to the apparent patch dependence.
- **E3 for both attention variants:** weakening content similarity changes attention a lot but does not make blur more consistent.
- **Not E1:** the patch-statistics calibration is worse than the plain curve.

**The plain radius curve (A0-C1), fitted on 12 diverse development patches, controls larger blur on held-out sources 88 % of the time.** The remaining misses are concentrated on three high-contrast patches.

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

Files:
- `results/attn_content.json`: gate, outcomes, all breakdowns;
- `results/cells.json`;
- `results/estimator_check.json`;
- `results/attn_content_sheet.png`: two test layouts at target 11.6 px; input then the 7 methods.

Images and `calib.json` are in `outputs/q2_attn_content/`.

### Gate: passed

- known disk blur recovered on all 24 patches (worst error 0.25 px);
- the repeat is identical;
- 397 outputs, none bad;
- 9.6 GB peak GPU.

### Control on held-out sources (216 cells per method; targets 7.8 / 11.6 / 15.5 px)

| method | agree | 7.8 | 11.6 | 15.5 |
|---|---|---|---|---|
| A0 raw request | 38 % | 97 % | 17 % | 0 % |
| **A0-C1** (fp32, radius curve) | **88 %** | 92 % | 85 % | 88 % |
| A0-CP (radius + patch statistics) | 61 % | 82 % | 46 % | 56 % |
| A1 raw (τ 0.5) | 8 % | 0 % | 1 % | 24 % |
| A1-C1 | 87 % | 90 % | 85 % | 86 % |
| A2 raw (τ 0) | 0 % | 0 % | 0 % | 0 % |
| A2-C1 | 0 % (all 216 requests "unreachable") | 0 % | 0 % | 0 % |

**Per patch, A0-C1** (delivered / target, median):

| | gravel | astronaut | chelsea | text | rocket | coins |
|---|---|---|---|---|---|---|
| patch 0 | 1.07 | 0.94 | 0.89 | 1.07 | 1.02 | **1.18** |
| patch 1 | 1.09 | 1.06 | 0.90 | 1.08 | **1.21** | **1.20** |

- Overall: median ratio 1.06 (interquartile range 0.98–1.15).
- **All failures sit on coins_0 (61 % agree), coins_1 (56 %) and rocket_1 (50 %); every other patch agrees ≥ 94 %.**
- These three are high-contrast isolated objects: bright coins on a plain ground, and lights at night. All three are over-blurred by about 20 %.

### Patch statistics (E1: no)

- The fitted coefficients on the three statistics are tiny: 0.019, −0.031, 0.023 per standard deviation, against 1.69 for log request.
- The log-linear form fits the curved response worse than the monotone curve: it predicts raw outputs with a median error of 10 %, against 8 % for C1.
- Inverting it as a control gives 61 %. **These statistics do not predict delivered blur, and the simpler curve controls better.**

### Content-similarity intervention (E3 for A1 and A2)

**Effect on attention** (attention mass leaving its own token / entropy): A0 0.195 / 0.216, A1 0.097 / 0.166, A2 0.003 / 0.022.

**Raw blur at a fixed request** (mean r̂ for 7.8 / 11.6 / 15.5 px):
- A0: 7.4 / 16.1 / 23.3;
- A1: 1.3 / 6.8 / 20.0;
- **A2: 1.4 / 1.4 / 1.5. With content similarity removed, there is no blur at all at these requests.**

**Mechanism this reveals.** These requests are all below one token spacing (16 px). The optical gate gives a neighbouring token only a tiny weight (e.g. ~10⁻³ at 11.6 px).

Blur appears anyway because PISA normalizes each source token's weights over targets after exp(query·key − max). A neighbour that is much more similar than the token's own query can outweigh the gate's ~10⁻³. **So the content-similarity term is what produces sub-token blur in PISA, not a disturbance on top of it.** This also fits rounds 20–21: blur appears only once the gate leak, amplified by similarity, reaches a few percent.

**Consistency:**

| | A0 | A1 | A2 |
|---|---|---|---|
| across-patch CV at a fixed raw request | 0.096 / 0.104 / 0.086 | 0.206 / 0.102 / 0.092 | ≈ 0.19 |

A1 is 8 % *worse* than A0 (median), so the ≥ 20 % reduction rule fails. With its own calibration (same budget) A1 matches A0 (87 % vs 88 %), with the same per-patch pattern (coins and rocket_1 over-blurred by about 20 % in both). **Weakening content similarity moves the blur but leaves the patch-to-patch pattern unchanged.**

**Quality:**
- focus MAE against the input: 2.85 (A0-C1), 2.84 (A1-C1) and 2.09 (A2-C1) /255;
- disk-fit residual: 0.040, 0.040 and 0.043.

Neither is worse.

### Patch vs position, and measurement (M)

- **Patch vs position:** at a fixed request, patch identity explains 65–87 % of the variance of r̂ (A0) and position 1–3 %. The variation is tied to the patch, not where it sits.
- **Measurement (M fires):** the same patches under *known* Gaussian blur give disk-fit readings with across-patch CV 3.2 / 4.6 / 6.5 / 10.9 % (σ 3 / 5 / 7 / 9). The median, 5.6 %, is more than half of A0's patch CV (9.6 %). So a substantial part of the apparent patch dependence may come from how a non-disk kernel is read on different content, not from the model.

## Interpretation

> On interior regions in our configuration, fp32 PISA attention plus an ordinary monotone radius curve, fitted on 12 diverse development patches, controls larger blur (7.8–15.5 px) on patches from held-out source images 88 % of the time. Patch statistics did not help. Weakening PISA's content-similarity weights did not make blur more consistent: those weights are what produce sub-token blur at all. The remaining misses are on three high-contrast patches (coins, lights at night), over-blurred by about 20 %. They may be partly measurement.

- **Compared with round 22** (larger blur 67 %): calibration coverage looks like the main difference. Round 22 fitted on 2 scenes; this round uses 12 patches from 6 sources, and every free cell carries the same request. This is not a controlled comparison.
- **For the attention-steering proposal:** this probe gives **no support** for "steer content-similarity to make control more consistent". The patch-to-patch pattern survives a halving of attention spread. Under the decision table, look at another component before building a controller, or at a different failure.
- **Open, small and specific:** the high-contrast / highlight patches. Two explanations remain: (a) BokehDiff renders bright isolated features with more spread, which may even be physically right for highlights (bokeh discs); (b) the disk-equivalent reading is biased on such content. Telling them apart needs known-kernel renders of those patches with a realistic bokeh kernel.

**Not shown:**
- occlusion boundaries (where BokehDiff's physics matters most);
- real photographs;
- other K or resolutions;
- whether τ applied only to some layers or heads behaves differently. τ here is a global probe, not a proposed method.
