# Q2 round 20: does half precision in BokehDiff's blur attention cause the 2–5 px gap?

**Question.** In round 19, requests up to 4.8 px gave only the baseline softness (about 1.5 px), and 5.8 px jumped to about 6 px. Is that gap caused by half-precision rounding in PISA's attention gate?

**Status.** Plan and rules fixed before running.

## Why this is a candidate (numerical check, not a generation)

PISA lets a source token j reach a target i with gate weight sigmoid(ln 10⁷ · (coc_j − 51.2 · dist(i, j))), in `PISA_attn_processor.py`. At 512² the attention grid is 32 × 32, so neighbouring tokens are 51.2/31 = 1.65 gate units apart. The table gives the neighbour gate for each requested radius (`r_nom = 0.969 · K · |Δd|`):

| r_nom (px) | 1.9 | 2.9 | 3.9 | 4.8 | 5.3 | **5.8** | 6.8 | 7.8 | 9.7 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| gate, float64 | 6·10⁻¹¹ | 3·10⁻¹⁰ | 2·10⁻⁹ | 8·10⁻⁹ | 2·10⁻⁸ | 4·10⁻⁸ | 2·10⁻⁷ | 1·10⁻⁶ | 3·10⁻⁵ | 0.5 |
| gate, float16 | 0 | 0 | 0 | 0 | 0 | **6·10⁻⁸** | 2·10⁻⁷ | 1·10⁻⁶ | 3·10⁻⁵ | 0.5 |

**The fp16 gate is exactly 0 up to 5.3 px and first becomes non-zero at 5.8 px**, which is where the output jumps.

The gate is then multiplied by exp(qk − max over targets) and normalized per source over targets. A tiny neighbour gate can therefore matter if the source token's own score is far below its largest.

**Expectation, stated in advance.** On random query/key values (CPU, fp32) a source keeps all but about 1–2·10⁻⁵ of its weight at these requests. If the real attention behaves similarly, fp32 will change little. The run decides.

## Test (`run20.py`, one scene)

**Change under test.** PISA's attention function (gate, exp, normalization, occlusion mask, dropout, accumulation) is called in float32, with autocast off. The rest of the model stays fp16, and inputs and seeds are fixed.
- Limit: the distance matrix and blur map arrive already rounded to fp16. That shifts gate arguments by < 0.02, small next to the underflow under test.

**Scene and sweep.**
- Scene S3 from round 19 (a held-out scene where the gap appeared).
- Focus d_f = 0.5; the anchor pair stays at ±0.5 so the map spans [0, 1].
- The other three pairs (6 cells) all get gap ±g, with g ∈ {0.05, 0.075, 0.10, 0.1125, 0.125, 0.1375, 0.15, 0.175, 0.20, 0.25}. That is r_nom 1.9, 2.9, 3.9, 4.4, 4.8, 5.3, 5.8, 6.8, 7.8, 9.7 px at K = 40.

**Runs:** {fp16, fp32} × seeds {0, 1} × (K = 0 + 10 sweep levels) = 44 images, plus 2 checks.

**Recorded:**
- PISA call count, tensor shapes and dtypes, and time per call;
- time per image;
- with seed 0, a dropout-free recomputation of each source token's attention mass leaving itself, and its number of non-zero targets, by blur-map value.

## Rules (fixed now)

**Gate:**
- through the wrapper, fp16 reproduces round 19's S3 R1 P1 output (max difference ≤ 1/255);
- the fp32 repeat is identical (≤ 1/255);
- no NaN or black outputs;
- estimator validation passes.

**Measure:** r̂ per cell with round 19's estimator; the median over the 6 swept cells.

| | rule |
|---|---|
| A, "fp32 raises mild blur" | at ≥ 2 of r_nom {2.9, 3.9, 4.8} px, the fp32 median exceeds the fp16 median by ≥ 1 px **for both seeds** |
| B, "fp32 covers 2–5 px" | for each of the bins [2, 3), [3, 4), [4, 5) px, some sweep level's fp32 median (both seeds pooled) falls inside |
| **P1** | A and B: precision causes the gap. An implementation finding; it does not by itself motivate a new steering method |
| **P2** | A only: precision contributes, but the gap is only partly closed |
| **P3** | not A: the gap persists with fp32 attention. Look next at the coarse 16 px token grid and gate shape, which is still not a proven architectural limit |

**Also reported:**
- focus-cell MAE against the input, fp32 versus fp16;
- r̂ of the focus cell and the anchors;
- the 5.8–9.7 px levels;
- seed noise;
- runtime;
- attention spread by blur value.

## Results

Pending.
