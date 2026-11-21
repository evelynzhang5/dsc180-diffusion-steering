# Round 38 (CPU): is depth readable at boundaries, and what is the calibrated request's remaining error made of?

These are diagnostics on round 36's saved outputs: up_blocks.0 activations and the unedited calibrated-request images of 12 development and 12 test scenes. No new generations. They inform the boundary investigation; they do not test steering.

## D1. Depth at boundaries (`edge_probe.py`)

**Design.**
- **Tokens:**
  - *interior* tokens lie inside a free cell's interior;
  - *boundary* tokens are covered by exactly two cells, each with ≥ 25 % of the pixels.
- A boundary token mixes two depths, so it gets no single depth label. Instead the targets are:
  - the disparity of each side;
  - their difference;
  - which side is closer;
  - controls: the requested blur on each side, and its difference.
- **Probes:** ridge regression, with λ by leave-one-scene-out cross-validation on development scenes, evaluated on the 144 test edges.
- **Baselines for the closer side:**
  - *blur-only*: a logistic model on the two requested blurs, fitted on dev;
  - *position-only* (post hoc): the dev majority per cell pair.
- **Rule (fixed before running):** depth is readable at boundaries if (i) the probe beats the blur-only baseline by ≥ 10 points, and (ii) the same-blur edges are ordered correctly in ≥ 11/12.

**Results (test).**

| Measure | Value |
|---|---|
| Interior probe, disparity: r / MAE | 0.09 / 0.25 (as round 36: nothing) |
| Interior probe, requested blur: r | 0.96 |
| Boundary probe, disparity difference across the edge: r (token / edge) | 0.73 / 0.79 |
| Boundary probe, requested-blur difference: r | 0.94 |
| Closer side, per edge: probe / blur-only / position-only | **89 % / 83 % / 83 %** |
| Same-blur edges ordered correctly | 8/8 (p = 0.004), but see below |
| *Post hoc:* edge partial r (predicted vs true disparity difference), given blur | 0.50 |
| *Post hoc:* the same, given blur **and** cell-pair position | **0.34** (144 edges) |
| *Post hoc:* free–free edges only, partial r given blur | 0.51 (48 edges) |

**Decision: the rule fails.**
- (i) The probe beats blur-only by only +5.6 points.
- (ii) Only **8** same-blur edges exist, not 12. I had counted them with the nominal blur; the calibrated requests separate some pairs.
- The 8/8 is uninformative. The 8 edges are two configurations repeated four times each (cells 6–7 and cells 1–2, both with gaps ±0.15), always with the same side closer, and the position-only baseline also gets 8/8.

**Reading.**
- **Interiors carry no depth.** Boundaries carry *some* depth-order information that requested blur and position do not explain (partial r 0.34–0.51). It is consistent with disparity entering only through PISA's occlusion, which acts at edges.
- **The signal is modest.** Most of what a boundary probe reads is the blur difference across the edge.
- A boundary steering test would therefore have a real, if limited, depth signal to work with. Whether steering is useful there is a separate question.

## D2. The calibrated request's remaining error (`residual_diag.py`)

**Measures.** On 66/72 test cells within tolerance; development cells give similar results. These are complementary diagnostics, not an additive decomposition.

| Measure | Median (10–90 %) |
|---|---|
| Best-fitting disk radius − requested radius | +0.2 px (−1.2 to +1.8) |
| Error at r\* (round 36's score) | 3.17 (1.78–8.31) |
| Error at the best-fitting radius | 2.99 (1.72–8.08) |
| Share of the error removed by the best radius | **2.4 %** (0.1–24 %) |
| Contrast ratio at the best radius | 0.975 (0.92–1.10) |
| Gain on disk(r_best) | 0.95 (0.90–1.07) |
| Gain/offset-invariant error at r_best | 2.97, about equal to the error itself |
| Per-channel offset (removed by the score) | 2.5/255 (0.9–9.8) |
| Colour cast (spread of the offsets across channels) | 1.0/255 (0.6–4.4) |
| Low-frequency share of the residual (σ 4 px) | 0.61 (0.41–0.75) |
| Retained-source-detail slope | 0.016 (−0.01 to 0.04): small |

**Reading.**
- Within tolerance, radius explains little of the remaining error: a median of 2.4 %, up to about 25 % for some gravel and text cells.
- Contrast and gain are close to correct.
- The residual is mostly broad, low-frequency structure, not retained fine texture.
- The error is dominated by a few textures: coins 9.6 and astronaut 7.6, against 2.3–3.5 for the others.

## D2b (post hoc). The sRGB reference itself is a large part of that error (`linear_check.py`)

**Motivation.** Every reference since round 18 is a disk blur computed in **sRGB**. A lens averages light in **linear** light. This check re-scores the same outputs against a linear-light disk blur (sRGB → linear → blur → sRGB), each at its own best-fitting radius.

| | Median best error, sRGB reference | Median best error, linear reference | Linear better |
|---|---|---|---|
| Test (72 cells) | 2.99 | **2.02** | **72/72** |
| Dev (72 cells) | 2.54 | 1.75 | 68/72 |

Per test texture (sRGB → linear):

| Texture | sRGB | Linear |
|---|---|---|
| coins | 9.23 | **2.50** |
| astronaut | 7.46 | 3.95 |
| rocket | 3.01 | 1.42 |
| chelsea | 2.72 | 1.78 |
| gravel | 2.28 | 1.79 |
| text | 1.84 | 1.22 |

The improvement does not track near-saturated highlight share (r −0.04 on test). It is a general tone effect of blurring high-contrast texture in linear light.

**Implication (needs a proper rescoring before any claim is revised).**
- Much of the "texture-dependent error" that rounds 18–37 measured, coins in particular, may be a mismatch between BokehDiff's (apparently linear-light-like) blur and our sRGB reference, not a BokehDiff failure.
- The teacher targets of rounds 33–37 were also sRGB disk blurs.
- The best-radius shifts are small (≤ 0.9 px), so radius-based results (calibration, the round 36 size direction) are largely unaffected. Error-based comparisons are not.
- **Next:** rescore the key earlier comparisons (round 29/31 request-only and baselines, round 32) against linear-light targets, and build every future reference, including the boundary renderer, in linear light.

Files: `edge_probe.py`, `residual_diag.py`, `linear_check.py`, and `results/` (`edge_probe.json`, `residual_cells.json`, `residual_summary.json`, `linear_check.json`).
