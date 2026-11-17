# Q2 round 26: attention-spread steering. Can steering PISA's attention make blur control consistent where calibration is not?

**Question.** Rounds 20–25 showed how PISA produces blur.
- Delivered blur tracks the share of each token's attention that leaves the token.
- That share is a gate leak growing across many orders of magnitude, amplified by content similarity exp(q·k − max).
- The amplification varies by patch. Calibration therefore gets 94 % (mild, within ±1 px) / 88 % (larger) on held-out patches, but leaves about 10 % patch-to-patch spread, and high-contrast patches (coins, night lights) over-blurred by about 20 %.

Global levers (similarity temperature, round 24; gate shape, round 25) failed.

**This round steers each token's spread directly.** Does that beat calibration where calibration falls short?

**Status.** Run Nov 17 (pod `q2-steer`, 173 images in 6 min). The test's own checks all passed: the baseline reproduces round 25 exactly, and steering hits its target share (median error 2·10⁻⁷). Plan and rules were committed (e9467c5) before the run.

**Outcome F.** Steering every token to exactly the same attention share does **not** make delivered blur more consistent across patches, and does not halve the high-contrast error. So the remaining patch dependence is not set by how much attention PISA spreads.

**Design flaw on my side:** the m range (≤ 0.4) reaches only about 4.8 px, so every target ≥ 6 px was unreachable. The mild-only comparison below does not depend on that flaw and leads to the same conclusion.

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

Files:
- `results/steer.json`;
- `results/cells.json`;
- `results/steer_sheet.png`: two test layouts; baseline vs steering at 4, 8 and 15.5 px.

Images and `calib.json` are in `outputs/q2_spread_steer/`.

**Gate passed:**
- the baseline equals round 25 (max difference 0);
- the repeat is identical;
- 173 outputs, none bad;
- estimator validation passed;
- the steered leave share hit its target (median |L − m| = 2·10⁻⁷).

The one large error (0.98) is from a source token whose off-self weight underflowed (O_j = 0), so it could not be steered.

**Frozen steering curve** (m → delivered px, development):

| m | 0.06 | 0.10 | 0.13 | 0.16 | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---|---|---|---|---|---|---|---|
| delivered (px) | 1.6 | 1.7 | 1.9 | 2.2 | 2.6 | 3.0 | 3.4 | 4.5 |

This is a smooth, gentle knob: about 1 px per 0.1 of share, compared with the baseline's 2 → 5 px in 1.5 px of request. But it stops at about 4.8 px.

### Frozen comparison (held-out patches, corrected reruns)

| | mild 3–6 px within ±1 px | larger 8–15.5 px within tolerance | across-patch CV (median, 7 targets) | high-contrast \|ratio − 1\| | focus MAE | residual |
|---|---|---|---|---|---|---|
| baseline (fp32 + curve) | **94 %** | **88 %** | 0.103 | 0.144 | 2.55 | 0.037 |
| spread steering | 77 % | 0 % (unreachable) | 0.116 | 0.222 | 2.28 | 0.035 |

Conditions (a), (b) and (c) fail; (d) holds. **Outcome F.**

### Mild targets only (3–5 px; both methods reach them; post-hoc, not part of the frozen rule)

| | high-contrast patches, median \|ratio − 1\| | other patches | across-patch CV at 3 / 4 / 5 px |
|---|---|---|---|
| baseline | 0.112 | 0.073 | 0.117 / 0.097 / 0.103 |
| steering | 0.085 | 0.089 | 0.108 / 0.106 / 0.115 |

**Per-patch delivered/target ratio at 3–5 px:**

| method | astronaut_1 | chelsea_0/1 | coins_0 | coins_1 | gravel_0 | rocket_1 | text_0/1 |
|---|---|---|---|---|---|---|---|
| baseline | 0.82 | 0.90 / 0.85 | 1.13 | 1.09 | 1.12 | 1.08 | 1.04 / 1.03 |
| steering | 0.80 | 0.85 / 0.82 | 1.13 | 0.97 | 1.05 | 1.03 | 0.93 / 0.95 |

- The high-contrast error falls by about a quarter, not half. The other patches get slightly worse. The CV is unchanged.
- **The per-patch pattern survives:** astronaut and chelsea under-blur, and coins_0 over-blurs by 13 %. That holds even though every steered token now sends exactly the same share of its attention to its neighbours.
- Raw sweeps agree. At a fixed m, the across-patch CV is 0.13–0.20, no lower than the baseline's at a fixed request (round 25: 0.08–0.21).

## Interpretation

- **The patch dependence is not in PISA's attention amount.** With the attention spread fixed per token, patches still blur differently, in the same order. The dependence must arise after PISA:
  - in how the rest of the UNet and the decoder render the same attention spread on different content;
  - and partly in the measurement (round 24: known blur already reads 3–11 % differently across these patches).
- **Steering's one benefit:** a smooth, gentle mild-blur knob (about 1 px per 0.1 share). Calibration already reaches mild targets just as accurately, so this is a usability property, not an accuracy gain.
- **For the steering goal:** three attention-level levers are now closed on the patch-consistency problem.

  | round | lever | result |
  |---|---|---|
  | 24 | similarity temperature | moves blur, same patch pattern |
  | 25 | gate shape | breaks control |
  | 26 | per-token spread | same patch pattern even at exactly equal spread |

  If steering is to fix patch dependence, it has to act **downstream of PISA**, on the features that turn attention spread into pixels. Or it targets a different failure (boundaries, highlights).

**Not shown:**
- steering with a wider m range or a larger R_REF (needed for ≥ 6 px; a design flaw here);
- downstream steering;
- boundaries.
