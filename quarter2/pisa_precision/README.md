# Q2 round 20: does half precision in BokehDiff's blur attention cause the 2–5 px gap?

**Question.** In round 19, requests up to 4.8 px gave only the baseline softness (about 1.5 px), and 5.8 px jumped to about 6 px. Is that gap caused by half-precision rounding in PISA's attention gate?

**Status.** Run Nov 17 (pod `q2-precision`, 46 images in 2 min). The test's own checks all passed. **Outcome P3: the gap persists with fp32 attention.** fp32 moves the blur onset down by about half a pixel of request (from 5.8 to 5.3 px), but requests of 1.9–4.8 px still give only the baseline softness. Plan and rules were committed (ad51075) before the run.

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

Files:
- `results/precision.json`: gate, decision, curves, runtime, dtypes, attention spread;
- `results/cells.json`;
- `results/precision_sheet.png`: input; then rows fp16 / fp32, seed 0, K 0 and the 10 sweep levels.

Images are in `outputs/q2_pisa_precision/`.

### Gate: passed

| check | result |
|---|---|
| wrapper in fp16 against round 19's S3 R1 P1 | identical (max difference 0) |
| fp32 repeat | identical |
| outputs | 46, no NaN or black; 9.5 GB peak GPU |
| estimator validation | passed (worst 0.25 px) |

**What PISA actually runs:**
- 2 calls per image (the two transformer blocks of `down_blocks.1.attentions.0`);
- query shape [1, 10 heads, 1024 = 32 × 32 tokens, 64];
- query, distance matrix and blur map all arrive in float16, under autocast, in both modes;
- in fp32 mode the wrapper converts them.

### Sweep: median r̂ of the 6 swept cells (both seeds pooled; seed 0 / seed 1 differ by a median of 0.02 px)

| r_nom (px) | 0 (K 0) | 1.9 | 2.9 | 3.9 | 4.4 | 4.8 | 5.3 | 5.8 | 6.8 | 7.8 | 9.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | 1.68 | 1.7 | 1.7 | 1.7 | 1.7 | 1.7 | 1.7 | **3.9** | 5.8 | 8.1 | 13.2 |
| fp32 | 1.65 | 1.7 | 1.7 | 1.7 | 1.7 | 1.8 | **2.6** | 3.5 | 5.6 | 8.0 | 13.1 |

**Rule results:**
- **Rule A fails.** At 2.9, 3.9 and 4.8 px, fp32 raises the median by 0.0, 0.0 and 0.1 px (≥ 1 px was needed).
- **Rule B fails.** fp32 lands in [2, 3) (2.6 at 5.3 px) and [3, 4) (3.5 at 5.8 px), but nothing in [4, 5). It jumps from 3.5 to 5.6.

**Effect of fp32.** fp32 makes the onset slightly earlier and smoother:
- fp32: 1.8 → 2.6 → 3.5 → 5.6;
- fp16: 1.7 → 1.7 → 3.9 → 5.8.

At 5.8 px and above, fp32 gives 0.1–0.4 px *less* blur than fp16.

**What stays the same between modes:**
- Focus cell: r̂ 0.83 vs 0.94 px; MAE against the input median difference 0.00/255.
- Anchors (19.4 px requested): 27.4 vs 27.1 px.
- Runtime: 1.42 s per image in both modes; PISA itself takes about 10 ms per image either way.

### Attention spread (dropout-free recomputation, seed 0)

"Leaves self" is the share of a source token's attention mass not kept on its own token, averaged over heads. It includes mass removed by PISA's occlusion mask.

| blur-map value (r_nom px) | 0.2 (1.9) | 0.3 (2.9) | 0.4 (3.9) | 0.5 (4.8) | 0.55 (5.3) | 0.6 (5.8) | 0.8 (7.8) | 1.0 (9.7) |
|---|---|---|---|---|---|---|---|---|
| fp16: leaves self | 0.3 % | 1.1 % | 1.3 % | 1.6 % | 1.7 % | 7.0 % | 11.9 % | 16.2 % |
| fp16: non-zero targets besides self | 0 | 0 | 0 | 0 | 0 | 3.8 | 3.8 | 3.8 |
| fp32: leaves self | 1.1 % | 2.1 % | 2.7 % | 4.2 % | 5.3 % | 6.5 % | 11.7 % | 15.9 % |
| fp32: non-zero targets | 31 | 31 | 31 | 36 | 36 | 36 | 36 | 36 |

- **fp16 rounding is real.** Below 5.8 px every neighbour weight underflows to 0, and the reviewer's calculation is confirmed inside BokehDiff. In fp32 the spread grows smoothly instead: 1 → 5 % between 1.9 and 5.3 px.
- **But the output barely registers that spread.** Spreading 2–4 % of each token's attention mass (fp32, 2.9–4.8 px) leaves the measured blur at the baseline. Visible blur appears only once about 5–7 % of the mass leaves self, at about 5.3 px in fp32 and 5.8 px in fp16. The spread is also very lopsided: the gate is ≈ 1 for the token itself and only ~10⁻⁹–10⁻⁸ for neighbours 16 px away. At the attention level, a small blur radius has no gradual representation on this grid; it appears as a little mass leaking to whole neighbouring tokens.

## Interpretation

- Half precision is not the main explanation. It accounts for about 0.5 px of request at the onset, and the flat region remains with fp32 attention.
- **Wording correction (after review, Nov 17).** This is not a missing 2–5 px *output* range. Outputs of 2.6, 3.5 and 3.9 px do occur. What is supported is **a flat response to small requests followed by a steep rise**: a control problem. Whether intermediate values are reachable with finer requests is tested in round 21. The frozen outcome P3 stands; only its reading is narrowed. **Round 21 follow-up:** with 0.1 px request steps, shipped fp16 gives a staircase that skips about 3 px, at exactly the fp16 gate-rounding thresholds; fp32 is continuous. So precision does decide whether mild blur can be dialled in continuously. It does not decide where the flat region ends (about 4.9 px of request in fp32).
- The evidence points next at **the 16 px token grid and the gate shape**. Between tokens, the gate goes from ≈ 10⁻⁸ to 0.5 only as the request rises to 16 px. So blur smaller than one token must come from small leaks the rest of the network mostly ignores.
- This is a description of PISA's mechanism, **not a proven architectural limit**. Other layers could in principle produce sub-token blur; they just do not here.

**Not shown:**
- other scenes (one scene, though the onset agrees with round 19's six);
- other K;
- what would happen with a finer attention grid.

## Suggested next step (not started)

A direct check of the grid explanation that keeps BokehDiff itself unchanged: rerun the same sweep with BokehDiff's own `--upsample 2` path. The image is processed at 1024², so PISA's grid is 64 × 64 tokens, 8 px apart in the 512 output.
- If the grid causes the gap, the onset should move to about half the request (about 2.5–3 px).
- Memory must be checked first: the attention matrix grows 16× (4096²).
