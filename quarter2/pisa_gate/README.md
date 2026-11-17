# Q2 round 25: can a redesigned PISA gate restore graded control of sub-token blur?

**Question.** Rounds 20–24 traced how BokehDiff makes small blur.
- PISA's optical gate is a sharp step on the 16 px token grid. For requests under about 16 px it gives neighbouring tokens only 10⁻¹¹–10⁻³.
- Blur below one token therefore appears only when content similarity, exp(q·k − max), amplifies that leak. The result is a flat-then-steep response (rounds 20–21) that depends on the patch (rounds 22, 24).

**Test.** At inference only, replace the gate with one that represents sub-token blur directly. Does that give graded, consistent control of mild blur without hurting larger blur or the in-focus content?

**Status.** Run Nov 17 (pod `q2-gate`, 263 images in 8 min). The test's own checks all passed; the re-implementation with G0 equals the original exactly. Plan and rules were committed (86356ad) before the run.
- **Outcome N: the cell-transfer gate (G1) does not restore graded mild blur; it makes control much worse.** Any request above about 0.7 px jumps to 21–42 px of blur.
- **The original gate (G0) with fp32 and a curve fitted on 12 diverse patches controls mild blur well:** 94 % of held-out mild targets within ±1 px, and 88 % of larger targets within tolerance.

## The two gates (`run25.py`)

| | gate from source token j to target token i |
|---|---|
| G0 (original) | sigmoid(ln 10⁷ · (coc_j − 51.2 · dist_ij)) |
| G1 (cell transfer) | the fraction of a uniform 16 × 16 source cell's light that a disc of radius r_j = 9.6875 · coc_j carries into target cell i (the disc convolved with both cells) |

G1 sums to 1 over targets. It equals G0's behaviour at r = 0 (self only), and is close to an anti-aliased G0 for r ≳ 16 px.

| request (px) | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| G1, self | 0.90 | 0.80 | 0.71 | 0.62 | 0.45 | 0.31 |
| G1, direct neighbour | 0.025 | 0.048 | 0.068 | 0.086 | 0.114 | 0.133 |
| G0, direct neighbour | 8·10⁻¹¹ | 2·10⁻⁹ | 6·10⁻⁸ | 2·10⁻⁶ | 10⁻³ | 0.5 |

**What stays the same:** content similarity, per-source normalization, the occlusion mask, dropout, and the rest of the model. PISA runs in fp32, and the attention function is re-implemented so that only the gate differs.

**Check:** the re-implementation run with G0 must reproduce the untouched original exactly.

**Caveat, stated now:** the released model was fine-tuned with the step-like G0. Feeding it graded weights is off-distribution, and quality checks guard against that.

## Design (round 24's patches; whole sources held out)

**Development sweep:**
- requests 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 19 px (K = 40; 19 px keeps the depth gap ≤ 0.5);
- 12 development patches × 2 positions, for each gate;
- this gives a monotone request → radius curve per gate. Both gates get the same calibration budget.

**Test, raw:** the same sweep on 12 held-out patches × 2 positions.

**Test, corrected:** each gate's own curve is inverted for targets 3, 4, 5, 6, 8, 11.6 and 15.5 px, and the editor is actually rerun. 12 patches × 3 positions.

That is 262 images in all.

## Decision rules (fixed now)

**Gate:**
- estimator validation passes;
- the re-implementation with G0 equals the original (≤ 1/255);
- the repeat is identical;
- no NaN or black outputs.

**Measures:**
- **Mild control:** corrected held-out cells with targets 3–6 px landing within **±1 px** (stricter than the usual tolerance, which would let the 1.5 px floor pass for small targets).
- **Larger control:** targets 8–15.5 px within max(2 px, 20 %).
- **Consistency at equal achieved blur:** the across-patch CV of corrected outputs at mild targets (median over 3–6 px).
- **Quality:** focus-cell MAE against the input, and the disk-fit residual.

| outcome | rule |
|---|---|
| W | G1 mild ±1 px ≥ G0 + 10 points, **and** G1 larger ≥ G0 − 5 points, **and** G1 mild patch CV ≤ 0.8 × G0's, **and** focus MAE not worse by > 0.5/255, **and** residual not worse by > 20 % |
| P | mild improves by ≥ 10 points, but another W condition fails |
| N | mild improves by < 10 points |

**Also reported:**
- the raw response curves;
- the request window that takes the output from 2 to 5 px;
- the largest step between neighbouring requests up to 8 px;
- raw across-patch CV per request;
- runtime.

**Reference** (round 24, different targets): fp32 + G0 + radius curve reached 88 % at 7.8–15.5 px.

**Not claimed:**
- a W would be a training-free inference change on one released model at 512², planar interiors only;
- it would still need fresh scenes and boundaries before counting as a method.

## Results

Files:
- `results/gate.json`;
- `results/cells.json`;
- `results/gate_sheet.png`: crops of one swept cell at raw requests 1–12 px, G0 row and G1 row.

Images and `calib.json` are in `outputs/q2_pisa_gate/`.

**Gate passed:**
- the re-implementation with G0 equals the original (max difference 0);
- the repeat is identical;
- 263 outputs, none bad;
- estimator validation passed;
- 1.42 s per image for both gates.

### Raw response on held-out patches (median r̂, px)

| request (px) | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 1.2 | 1.2 | 1.2 | 1.2 | 2.0 | 3.9 | 8.0 | 12.6 | 16.5 | 23.7 | 25.5 |
| G1 | **23.2** | 24.7 | 25.7 | 26.5 | 27.1 | 27.8 | 29.0 | 30.3 | 31.4 | 33.3 | 42.2 |

- **G0:** the output goes from 2 to 5 px over requests 5.0–6.5 px.
- **G1:** never in the 2–5 px range at these requests. Even 1 px gives 23 px.

### Control after each gate's own calibration (held-out patches, corrected reruns)

| | mild 3–6 px within ±1 px | larger 8–15.5 px within tolerance | mild across-patch CV | focus MAE (/255) | disk-fit residual |
|---|---|---|---|---|---|
| G0 | **94 %** | **88 %** | 0.106 | 2.52 | 0.037 |
| G1 | 1 % | 8 % | 0.201 | 2.79 | 0.053 |

**G0 per target** (within ±1 px / within tolerance):

| target (px) | 3 | 4 | 5 | 6 | 8 | 11.6 | 15.5 |
|---|---|---|---|---|---|---|---|
| G0 | 100 / 100 % | 100 / 100 % | 92 / 100 % | 83 / 97 % | 61 / 86 % | 33 / 86 % | 36 / 92 % |

**G1** asks for 0.07–0.7 px to hit 3–15.5 px targets. The output stays at about 1.2–1.7 px up to a 0.5 px request, then jumps to about 21 px at 0.69 px. Its curve has an even sharper step than G0's, only at a smaller request.

### Mechanism this reveals

- **Content similarity amplifies the gate enormously.** PISA normalizes each source token's weights over targets after exp(q·k − max). With G1, a neighbour weight of only about 1 % (a 0.5–1 px request) is already enough for similar neighbours to take most of a token's attention mass, which gives heavy blur.
- **The original gate's design is what makes the usable window possible.** Its sigmoid moves the neighbour weight through many orders of magnitude (10⁻¹¹ → 10⁻³ → 0.5) as the request rises. Combined with the amplification, that spreads the transition over requests of about 5–19 px. G1 moves the weight only linearly (0 → 0.13), so the amplified response has nowhere to go but straight up.
- Together with round 24 (removing content similarity removes blur), this means delivered blur in PISA is governed by *the product of a log-scale gate leak and content-similarity amplification*. A redesigned gate would need to act on that log scale, which the original sigmoid already does.

## Interpretation

- **Outcome N.** A physically motivated, linear sub-token gate is not a usable inference-time intervention for this model; it breaks control. This candidate steering option is closed.
- **The baseline is stronger than expected.** fp32 PISA plus a monotone radius curve, fitted on 12 diverse development patches, now controls:
  - mild blur (3–6 px) within ±1 px on 94 % of held-out patches;
  - larger blur (8–15.5 px) within tolerance on 88 %.

  Rounds 19–22 needed the fp16 fix and better calibration coverage to get here.
- **What remains open:**
  - occlusion boundaries (round 24's option 2);
  - highlight / high-contrast patches (round 24);
  - real photographs.

  In flat interiors, after the fp32 fix, an ordinary calibration is sufficient on this evidence.

**Not shown:**
- other gate shapes, e.g. a log-scaled graded gate or a G1 with tempered similarity;
- other models, K or resolutions.
