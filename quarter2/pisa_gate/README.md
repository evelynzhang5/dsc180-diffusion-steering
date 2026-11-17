# Q2 round 25: can a redesigned PISA gate restore graded control of sub-token blur?

**Question.** Rounds 20–24 traced how BokehDiff makes small blur.
- PISA's optical gate is a sharp step on the 16 px token grid. For requests under about 16 px it gives neighbouring tokens only 10⁻¹¹–10⁻³.
- Blur below one token therefore appears only when content similarity, exp(q·k − max), amplifies that leak. The result is a flat-then-steep response (rounds 20–21) that depends on the patch (rounds 22, 24).

**Test.** At inference only, replace the gate with one that represents sub-token blur directly. Does that give graded, consistent control of mild blur without hurting larger blur or the in-focus content?

**Status.** Plan and rules fixed before running.

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

Pending.
