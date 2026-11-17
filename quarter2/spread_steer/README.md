# Q2 round 26: attention-spread steering. Can steering PISA's attention make blur control consistent where calibration is not?

**Question.** Rounds 20–25 showed how PISA produces blur.
- Delivered blur tracks the share of each token's attention that leaves the token.
- That share is a gate leak growing across many orders of magnitude, amplified by content similarity exp(q·k − max).
- The amplification varies by patch. Calibration therefore gets 94 % (mild, within ±1 px) / 88 % (larger) on held-out patches, but leaves about 10 % patch-to-patch spread, and high-contrast patches (coins, night lights) over-blurred by about 20 %.

Global levers (similarity temperature, round 24; gate shape, round 25) failed.

**This round steers each token's spread directly.** Does that beat calibration where calibration falls short?

**Status.** Plan and rules fixed before running.

## Method (`run26.py`, inference only, training-free)

For source token j (per head) PISA has weights w_ij = exp(q_i·k_j − max) · gate_ij, normalized over targets i. The share leaving j is L_j = O_j / (w_jj + O_j), where O_j = Σ_{i≠j} w_ij.

**Spread steering** sets L_j to a target m for every blurred token, in closed form:
- w_ij ← s_j · w_ij for i ≠ j;
- s_j = m · w_jj / ((1 − m) · O_j).

**Content similarity still decides which neighbours receive the spread; steering fixes how much.**

**What is held fixed:**
- the blur map of steered cells is a fixed reference radius of 12 px (it shapes the neighbourhood);
- the disparity map keeps the geometry of the requested blur, as in the baseline;
- occlusion, dropout and the rest of the model are unchanged;
- the focus cell and anchors are not steered;
- PISA in fp32.

**The single knob.** m is mapped to delivered radius once, on development patches: a monotone curve m → r̂, frozen. Targets are then reached with m*(t).

**Baseline:** round 25's fp32 + original gate + its frozen radius curve, rerun here. It must reproduce round 25's images exactly.

## Design (round 24–25 patches; whole sources held out)

| part | setup | images |
|---|---|---|
| development | 12 patches × positions 0 and 3 × m ∈ {0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40} | 44 |
| test, raw | the same m sweep on 12 held-out patches (consistency at a fixed knob) | 44 |
| test, corrected | targets 3, 4, 5, 6, 8, 11.6, 15.5 px, positions 0, 2, 4, for steering and for the baseline | 2 × 42 |

That is 175 images, plus checks.

## Decision rules (fixed now)

**Gate:**
- estimator validation passes;
- the baseline reproduces round 25 (≤ 1/255);
- the repeat is identical;
- no NaN or black outputs;
- **steering hits its target share:** median |L − m| ≤ 0.005 over steered tokens, before occlusion and dropout.

**Pass conditions:**

| | condition |
|---|---|
| (a) accuracy not worse | mild (3–6 px) within ±1 px ≥ baseline − 5 points, and larger (8–15.5 px) within max(2 px, 20 %) ≥ baseline − 5 points |
| (b) consistency | across-patch CV at equal targets (median over the 7 targets) ≤ 0.8 × baseline |
| (c) high-contrast patches | median \|delivered/target − 1\| on coins_0, coins_1 and rocket_1 ≤ 0.5 × baseline |
| (d) quality | focus MAE not worse by > 0.5/255, and disk-fit residual not worse by > 20 % |

| outcome | rule | next |
|---|---|---|
| S | all of (a)–(d) | spread steering beats calibration where calibration falls short. Develop it (fresh scenes, boundaries, real photos) |
| Partial | (b) or (c), but not all | report which condition fails |
| F | neither (b) nor (c) | attention-level steering of this mechanism is closed; next steering target would be boundary behaviour |

**Not claimed:**
- one model at 512², K = 40, planar interiors, synthetic layouts;
- a pass is a smoke-test result, not a method paper.

## Results

Pending.
