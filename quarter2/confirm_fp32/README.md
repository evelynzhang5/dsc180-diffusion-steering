# Q2 round 22: does ordinary calibration give reliable control once PISA attention runs in full precision?

**Question.** Round 21 showed that, in our configuration, BokehDiff's released fp16 inference path turns the mild-blur response into a staircase, by fp16 rounding of PISA's neighbour gate. Full-precision (fp32) PISA attention removes the steps, with no measured runtime increase. Round 19's calibration test ran on the fp16 path. With fp32 attention, does an ordinary calibration now control blur reliably across held-out scenes?

**Status.** Run Nov 17 (pod `q2-confirm32`, 242 images in 7.5 min). The test's own checks all passed; the fp32 wrapper reproduces round 21 exactly. Plan and rules were committed (7ffb94b) before the run.
- **Outcome O3:** C1 and C2 reach 75 % on held-out cells (round 19, fp16: 70 %), short of 80 %.
- **Mild targets (2–6 px) are now handled well:** C1 reaches 90 %.
- **What remains is larger blur (≥ 6 px: 67 %).** No single condition dominates the failures.

## What is the same as round 19 (`bokehdiff_confirm/`)

- **Scenes and split:** six 3 × 3-cell scenes; S1, S2 for development, S3–S6 held out.
- **Conditions:** R0–R6 in both orientations; K = 40.
- **Calibration families:** C0 global gain, C1 monotone radius curve, C2 radius curve per side of focus.
- **Inversion:** request the smallest radius whose curve value reaches the target.
- **Measures:** the estimator, the ≥ 80 % agreement bar, the tolerance max(2 px, 20 %), and above-range handling.
- **Checks:** symmetry, context and small-blur, with round 19's rules.

## What changes

1. **Precision.** PISA's attention function runs in float32 for every call (round 20/21 wrapper); the rest of the model stays fp16.
2. **Refitting.** The calibrations are **refitted on fp32 development outputs.** Round 19's curves are not reused, since they compensate for the fp16 staircase.
3. **Gate addition.** The wrapper must reproduce round 21's fp32 output for the identical condition (S3, gap 0.15, K 40, seed 0) exactly. That proves it is active.
4. **Extra reporting (decided now):**
   - agreement for **tiny (< 2 px), mild (2–6 px) and larger (≥ 6 px)** targets;
   - agreement **per held-out scene**;
   - where C1's failures concentrate (run / focus setting, scene, target bin, side, texture).

## Decision rules (fixed now)

**Gate:** as in round 19 (estimator validation, repeat ≤ 1/255, no NaN or black outputs), plus the fp32 wrapper check above.

| outcome | rule | meaning |
|---|---|---|
| O1 | C1 ≥ 80 % on held-out cells | fp32 attention + an ordinary radius calibration addresses the tested failure. Record this before proposing anything more complex |
| O2 | not O1, C2 ≥ 80 % | the same, with a radius + side calibration |
| O3 | neither | calibration still falls short. Next: the richer focus-dependent calibration, tested on fresh scenes |

**Failure concentration** is reported with any outcome. A factor "dominates" if one of its levels holds ≥ 60 % of C1's failures, e.g. one run (focus setting) or one target bin. If one dominates, the next smoke test targets that condition instead of expanding everything.

**Comparison with round 19 (fp16, same design):**

| | uncorrected | C0 | C1 | C2 |
|---|---|---|---|---|
| round 19 agreement | 45 % | 59 % | 70 % | 70 % |

**Not claimed:**
- that every BokehDiff configuration behaves this way (one model variant, 512², K = 40, our scenes);
- anything about real photos or occlusion boundaries.

## Results

Files:
- `results/confirm.json`: gate, decisions, breakdowns, failure concentration;
- `results/cells.json`;
- `results/calib.json`: the refitted calibrations;
- `results/estimator_validation.json`;
- `results/confirm_sheet.png`, as in round 19.

Images are in `outputs/q2_confirm_fp32/`.

**Gate passed:**
- the fp32 wrapper matches round 21 (max difference 0);
- the repeat is identical;
- 242 outputs, none bad;
- estimator validation passed.

**Refitted calibration:**
- K = 0 softness: 1.46 px;
- C0 gain: 1.355;
- the C1 curve now rises 4.8 → 1.3 and 5.8 → 5.2 px (round 19: 1.2 → 5.8), with the rest almost unchanged.

The development grid has no requests between 4.8 and 5.8 px, so the curve interpolates linearly through that steep stretch.

### 1. Calibration on held-out scenes (O3)

| method | all (384) | tiny < 2 px (32) | **mild 2–6 px (96)** | larger ≥ 6 px (256) | S3 | S4 | S5 | S6 | round 19 (fp16), all |
|---|---|---|---|---|---|---|---|---|---|
| uncorrected | 43 % | 100 % | 44 % | 36 % | 40 % | 44 % | 46 % | 43 % | 45 % |
| C0 gain | 58 % | 100 % | 23 % | 66 % | 56 % | 60 % | 58 % | 56 % | 59 % |
| C1 radius curve | **75 %** | 100 % | **90 %** | 67 % | 71 % | 81 % | 81 % | 68 % | 70 % |
| C2 radius + side | **75 %** | 100 % | 91 % | 66 % | 73 % | 81 % | 79 % | 68 % | 70 % |

- **What fp32 changed:**
  - C1 on 3–6 px targets: 88 % (round 19: 70 %);
  - C1 on targets under 3 px: 100 % (round 19: 92 %);
  - overall: +5 points.
- **What it did not change:** larger targets.
  - 6–12 px: 56 %;
  - 12–20 px: 68 %;
  - 20–40 px: 78 %.
- **C2 again adds nothing over C1.**
- **Small-blur rule (targets 3–6 px within ±1 px):** 39 % (round 19: 26 %), so still "not attainable" under that stricter rule. Most of these cells are within the ordinary tolerance (88 %), but not within ±1 px.

**Failure concentration (C1, 95 failures).** No factor reaches the 60 % "dominates" threshold.

| factor | largest share of failures |
|---|---|
| target bin | 12–20 px 38 %, 6–12 px 37 % (75 % of failures are 6–20 px targets) |
| run | R2 29 %, R6 23 % |
| scene | S6 33 %, S3 29 % |
| side | behind 54 % |
| texture | gravel 44 % |

By focus position, C1's failure rate is 16 % at d_f = 0.2, 25 % at 0.5 and 34 % at 0.8.

### 2–4. Other checks (as in round 19; unchanged)

- **Symmetry:** median behind/front ratio 0.99 (71 % same sign). Not asymmetric.
- **Context:**
  - anchors are unchanged from calm to busy surroundings (0 / 24);
  - in-focus detail differs from the K = 0 output (12 / 12) but **no additional in-focus error against the input was measured** (4 / 12 rise ≥ 1/255; median +0.5/255).
- **Seed noise:** median 0.40 px.

### Post-hoc diagnostic (after seeing the data): why corrected requests fall short

As in round 19, the calibration curve predicts the *uncorrected* held-out outputs well (median residual −2 %). Yet *corrected* cells come out 16 % below the curve's prediction for the request they were given. Two explanations were checked:

1. **Interpolation between calibration points.** If this were the cause, the shortfall would shrink near the measured points. It does not:

   | corrected requests | median relative to the curve |
   |---|---|
   | within 0.15 px of a calibration point (n = 152) | −27 % |
   | ≥ 0.5 px from any calibration point (n = 232) | −17 % |

   **Not the cause.**
2. **The same request in a different setting.** Corrected cells were compared with *uncorrected* cells at the same request (within 0.15 px), the same focus position and the same scene. **Corrected cells blur 11 % less** (median, n = 96).
   - The difference: a correction changes only the blur map. The disparity map keeps the true geometry, so a corrected cell has the same blur request as the uncorrected one but a larger disparity gap from the focus. Other cells' requests also change.
   - Round 19 found that other cells' blur does not move an anchor's blur. That leaves the **disparity map as the likely factor**: it enters PISA through its occlusion test.
   - This would also fit round 19's focus-position effect, where the same request gave less blur when the focus sat near the end of the disparity range.

   This is a lead, not a result. The next test should change one thing at a time:
   - (a) the same request with the disparity map changed and the blur map fixed;
   - (b) PISA's occlusion mask recorded per cell.

## Interpretation

> In our tested configuration, fp16 attention quantization produces predictable steps in delivered blur (round 21). Full-precision attention removes those steps with no measured runtime increase. **With that fix, an ordinary radius calibration fitted on two scenes controls mild blur (2–6 px) on four held-out scenes within tolerance 90 % of the time, but larger blur only 67 %; overall 75 %, below the 80 % bar.**

- **The mild-blur control problem raised in rounds 19–21 is largely addressed** by precision plus an ordinary calibration. It needs no new method.
- **What remains is larger blur (6–20 px) on new scenes.**
  - Not side of focus (C2 = C1).
  - Not surrounding blur (round 19).
  - Not one dominant condition.
  - The post-hoc lead: blur depends on the disparity map as well as the blur map, so a calibration that only changes the blur map undershoots.

  Whether a richer calibration that uses disparity / focus position fixes this (O3's planned next step) is untested.

**Not claimed:**
- other BokehDiff configurations;
- other K or resolutions;
- real photos;
- occlusion boundaries.
