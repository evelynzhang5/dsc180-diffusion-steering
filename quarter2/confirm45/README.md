# Round 45: fresh-scene confirmation of the monotone blur controller (final round of the controller workstream)

Branch `q2-confirm45` (from `q2-precision44`). Round 44 showed that fp32 attention alone does not fix round 42's 4 px
misses at the tested settings (it does not rule out precision effects in general). Every development scene has a
scene-specific dead zone before blur rises. A monotone curve fitted on the development curves reached 4 px in 7/8
held-out development scenes, but only by interpolation between rendered settings, not by corrected renders. This round
renders the selected settings on fresh scenes. If it succeeds, the controller is written up; if not, its limits are
reported and no further controller search follows.

## Design (frozen before any round 45 image exists)

**Scenes.** 24 fresh generated scenes (`prompts45.py`): the same 4 initially-sharp categories, generator and settings as
round 40, with new subjects, backgrounds and seeds (4500+). Masks and depth from BokehDiff's own preparation. Regions
from round 40's `regions40.py`, unchanged.

**Coverage.** Every loss is counted: generated, prepared, eligible (≥ 4 behind tiles), excluded by the input-only review,
scored.

**Input-only review.** Mask failures and measurement flags are decided on the input contact sheets and committed as
`review45.json` before any output is pulled or viewed.

**Measurement tiles.**
- The behind tiles are split into selection tiles (even-indexed) and evaluation tiles (odd-indexed).
- No arm in this round measures its own output, so the selection tiles are reserved and unused.
- Every arm is scored on the same evaluation tiles. There is no fallback, so a scene with fewer than 2 evaluation tiles
  is not scored.
- All-tile results are secondary.

**Arms** (all fp32 PISA; K planned from inputs only, `plan45.py`; one render per cell):

| arm | K | role |
|---|---|---|
| monotone | isotonic curve of blur against K × gap, fitted on round 44's 8 fp32 development curves (`fit45.py`, `results/rule45.json`). Linear interpolation between knots; first crossing of the target; out of range → end knot (flagged); K clipped to [4, 64] | tested controller |
| linear | round 42's frozen offset rule (r = 0.950 (K·gap − 0.739)), unchanged | primary comparison |
| fixed | round 40's one setting per target | secondary baseline |

The monotone form was fixed in advance. Only one curve is fitted, with no selection among curves.

**Precision.** The fp32 wrapper (rounds 20-22, 44) now asserts, inside the attention call, that PISA's attention result
is float32, and logs input and compute dtypes for every call.

**Outcomes.**
- Primary: scene-averaged target attainment over 4, 8 and 12 px on the evaluation tiles, monotone − linear, paired
  scene bootstrap.
- Each target is also reported separately.
- Also: |error|, additional subject-detail loss (hf_ratio), colour change and render time, relative to linear and to fixed.

**Decision (frozen).**
- A: monotone − linear attainment ≥ +0.15, with CI lower bound > 0.
- B: no target deteriorates (paired point estimate ≥ −0.10 at each of 4, 8 and 12 px).
- C: paired hf_ratio CI lower bound ≥ −0.05 and paired colour-shift CI upper bound ≤ +1.0 (0-255 units).
- **CONFIRMED** = A, B and C. **PARTIAL** = A only. **NOT CONFIRMED** = not A.

**Dry run** (CPU; round 40 test scenes as stand-in data): the plan reproduces round 42's linear K exactly, and the scorer
and decision run end to end.

Renders: 24 × 3 targets × 3 arms = 216, about 45 min, plus about 1.5 h of generation.

## Result (`results/run/score45.json`; run 2026-11-25 on an NVIDIA A30)

**Coverage.** 24 generated → 24 prepared → 21 eligible (f_highlights_5, f_thin_2 and f_thin_5 had fewer than 4 behind
tiles) → 0 excluded by the input-only review (`review45.json`, committed before any output was pulled) → **21 scored**.

**Precision.** All 3,402 PISA calls received float16 inputs and computed in float32, checked by an assertion inside the
attention call.

Within tolerance on the reserved evaluation tiles, with scene bootstrap 95% CIs:

| arm | all targets | 4 px | 8 px | 12 px | all tiles (secondary) | median abs error | subject hf_ratio | render time |
|---|---|---|---|---|---|---|---|---|
| fixed | 0.33 [0.17, 0.51] | 0.43 | 0.29 | 0.29 | 0.27 | 3.67 px | 0.611 | 2.9 s |
| linear (round 42) | 0.41 [0.32, 0.49] | 0.19 | 0.71 | 0.33 | 0.51 | 2.06 px | 0.573 | 2.9 s |
| **monotone** | **0.73 [0.57, 0.87]** | **0.86** | 0.67 | **0.67** | 0.90 | 0.80 px | 0.578 | 2.9 s |

Paired monotone − linear:

| measure | difference [95% CI] |
|---|---|
| all targets | **+0.32 [+0.17, +0.46]** |
| 4 px | +0.67 [+0.38, +0.90] |
| 8 px | −0.05 [−0.14, 0.00] |
| 12 px | +0.33 [+0.05, +0.57] |
| subject hf_ratio | +0.003 [0.000, +0.005] |
| colour shift | 0.000 |
| render time | 0.000 s |

Monotone − fixed: +0.40 [+0.13, +0.65].

**Pre-set decision: CONFIRMED.**
- A: +0.32 ≥ +0.15 and the CI excludes 0.
- B: no target worse by more than 0.10; 8 px changes by −0.05.
- C: no additional subject or colour change.

**Reading.**
- On fresh scenes, the monotone curve fitted on fp32 development curves roughly doubles the share of requests delivered
  within tolerance compared with round 42's offset line: 0.41 → 0.73, and 0.86 at 4 px, where the line reached 0.19.
- It costs nothing extra: one render, same time, no additional subject change.
- The gain comes from modelling the dead zone at small K × gap; at 8 px the two controllers ask for nearly the same K.

**Limits.**
- 21 generated scenes from one generator; no real photographs in this test.
- Blur is measured by our own estimator on 2-4 evaluation tiles per scene.
- The curve is pooled over scenes: scenes whose dead zone differs strongly from the pool still miss.
- The run used an A30, not the 1080 Ti of earlier rounds. All arms ran on the same GPU, so the comparison is unaffected,
  but absolute times are not comparable with earlier rounds.

This closes the blur-controller workstream (review #7): the controller is consolidated and written up from here.
