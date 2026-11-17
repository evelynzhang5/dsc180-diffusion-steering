# Q2 round 22: does ordinary calibration give reliable control once PISA attention runs in full precision?

**Question.** Round 21 showed that, in our configuration, BokehDiff's released fp16 inference path turns the mild-blur response into a staircase, by fp16 rounding of PISA's neighbour gate. Full-precision (fp32) PISA attention removes the steps, with no measured runtime increase. Round 19's calibration test ran on the fp16 path. With fp32 attention, does an ordinary calibration now control blur reliably across held-out scenes?

**Status.** Plan and rules fixed before running.

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

Pending.
