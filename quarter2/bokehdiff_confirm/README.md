# Q2 round 19: can a simple calibration predict and correct BokehDiff's blur on new scenes?

**Question.** Round 18 found that BokehDiff's blur is ordered correctly by depth but not proportional to the request. Can a simple calibration predict and correct the blur on new scenes, or does the same request produce different effects depending on its surroundings?

**Status.** Run Nov 17 (pod `q2-confirm`), 241 outputs in 7.5 min. The test's own checks all passed.

**Outcome O3:** no simple calibration corrects the held-out scenes. The best is a radius curve at 70 % (need 80 %). Neither the front/behind asymmetry nor an effect of surrounding blur holds up. What remains is a dependence on **focus position** and on the **individual patch**. Plan and rules were committed (5211a48) before the run.

Setup as in rounds 17–18:
- BokehDiff at commit `38ff937`, unchanged;
- fp16 on the GTX 1080 Ti;
- 512 × 512 images, one-step inference;
- the random generator is reseeded before every call.

**Fixed for this round:**
- K = 40 throughout.
- The requested ("nominal") radius is r_nom = 0.969 · K · |d_norm − d_focus| px. Round 18 derived this from the attention code: it is the radius the internal constraint imposes.

## Scenes and conditions (`design.py`)

**Six controlled scenes.** Each is a 3 × 3 grid of fronto-parallel textured cells (171/170/171 px), made from scikit-image grass, gravel and brick, tinted and rotated.
- The centre cell is the focus cell.
- The other 8 cells form 4 pairs, placed opposite each other through the centre. **Both cells of a pair carry the identical texture crop.**
- Pair 0 (the top-left and bottom-right corners) is the "anchor" pair.
- **Development scenes:** S1, S2. **Held-out test scenes:** S3–S6.
- Every condition's disparity map spans exactly [0, 1], as BokehDiff's loader always produces.
- Measurement uses cell interiors only: at least 48 px from any depth edge and 24 px from the border, which leaves 74–99 px squares.

**Conditions.** Each condition gives a focus d_f and a signed disparity gap per pair (cell a, cell b). In orientation P1 they are as listed; **P2 swaps the two cells of every pair**, keeping the same content and request but exchanging positions.

| run | d_f | gaps for pairs 0 / 1 / 2 / 3 (P1) | r_nom of pairs 1–3 (px) | purpose |
|---|---|---|---|---|
| R0 | 0.5 | ±0.5 / 0 / 0 / 0 | 0 | calm: only the anchors are blurred |
| R1 | 0.5 | ±0.5 / ±0.05 / ±0.10 / ±0.15 | 1.9, 3.9, 5.8 | small requests |
| R4 | 0.5 | ±0.5 / ±0.075 / ±0.125 / ±0.175 | 2.9, 4.8, 6.8 | finer steps around the plateau |
| R2 | 0.5 | ±0.5 / ±0.20 / ±0.25 / ±0.30 | 7.8, 9.7, 11.6 | medium |
| R3 | 0.5 | ±0.5 / ±0.35 / ±0.40 / ±0.45 | 13.6, 15.5, 17.4 | busy: large blur all around |
| R5 | 0.2 | +0.8/−0.2, +0.55/−0.05, +0.65/−0.10, +0.70/−0.15 | front 21–31; behind 1.9–7.8 | large requests in front |
| R6 | 0.8 | mirror image of R5 | behind 21–31; front 1.9–7.8 | large requests behind |

The anchors (±0.5 at d_f = 0.5) request 19.4 px.

**Runs (`run19.py`):**
- **Stage 1, uncorrected, all 6 scenes:**
  - a K = 0 reference;
  - R0–R6 in both P1 and P2;
  - R3 P1 with seed 1 (noise floor);
  - one in-process repeat.
- **Stage 2:** measure every non-focus cell of S1 and S2 → fit calibrations → save `calib.json`. They are frozen from here.
- **Stage 3:** S3–S6, runs R1–R6, both P1 and P2, rerun with corrected requests for each calibration.

That makes 241 outputs in total. A correction changes only the blur-map value of a cell; the disparity channel keeps the true geometry.

## Calibrations (fitted on S1 + S2 only, `measure19.py`)

| | rule |
|---|---|
| C0 | one global gain (round 18's baseline) |
| C1 | monotone curve of measured radius against requested radius: isotonic regression, pooled, anchored at request 0 = the K = 0 softness |
| C2 | monotone curves of requested radius, separately for **front / behind the focus** (the sign of d − d_focus) |

- **Correction:** to get target radius t (= r_nom), request g⁻¹(t), the smallest request whose curve value reaches t (linear interpolation).
- **Edge cases:**
  - a target below the curve's start requests 0 ("below floor");
  - a target above the curve's top requests its largest request ("unreachable").
- **Above-range fits** (estimator at 48 px) enter the curve fit at 48, as a lower bound.

## Measurement

- **Estimator:** the same as round 18 (disk radius 0–48 px, per-channel gain and offset), computed in closed form. A self-check confirms it equals round 18's least-squares fit within 10⁻³ px.
- **Validation, rerun on these layouts:**
  - known disk radii 0–40 px;
  - pass: |error| ≤ max(1 px, 10 %);
  - CPU debug run: worst error 0.25 px.
- **Above range:** a fit at 48 px is reported as "above range" and never counts as agreement.
- **Agreement:** |r̂ − t| ≤ max(2 px, 20 % · t), as in round 18.
- **In-focus detail:** reported against both the original input and the K = 0 output.

## Decision rules (fixed now)

**Gate:**
- estimator validation passes;
- repeat difference ≤ 1/255;
- no NaN or black outputs.

Otherwise the outcome is "repair the test".

**1. Calibration.** Agreement is measured on held-out scenes S3–S6: runs R1–R6, P1 + P2, all non-focus cells, 384 cells per method. Uncorrected and C0 are reported too.

| outcome | rule | meaning |
|---|---|---|
| O1 | C1 ≥ 80 % | a fixed curve of requested radius corrects new scenes: blur amount is a calibration issue, not a steering problem |
| O2 | not O1, C2 ≥ 80 % | radius + side-of-focus calibration suffices |
| O3 | neither | the same request has different effects on new scenes; check 2–4 below for what it depends on |

Breakdowns reported:
- by target radius;
- by side;
- by run (i.e. focus position and how busy the surroundings are);
- by texture;
- requests that are below the floor or unreachable.

**2. Symmetry.**
- **Data:** identical patches at equal gaps in front of and behind the focus, at d_f = 0.5, for gaps ≥ 0.175 (r_nom ≥ 6.8 px). That is R2/R3/R4 free pairs plus the R2 anchors: 8 per scene, 48 instances, all scenes, uncorrected.
- **Statistic:** the ratio behind/front, averaged over P1 and P2 (geometric mean), which cancels position.
- **Asymmetric if:** |median log-ratio| ≥ log 1.15 and ≥ 80 % of instances share its sign. Instances with an above-range cell are excluded and counted.

**3. Context.**
- **Data:** the anchor cells (same patch, same 19.4 px request) in calm R0 against busy R3; same scene and orientation; 24 instances.
- **"Surroundings alter the blur" if:** |r̂_busy − r̂_calm| > max(2 px, 20 %) in ≥ 50 % of instances.
- **Focus cell, 12 instances:**
  - "blur elsewhere alters in-focus detail" if its MAE against the K = 0 output rises by ≥ 1/255 from R0 to R3 in ≥ 80 % of instances;
  - "extra damage beyond reconstruction" if the same holds for MAE against the input.
- **Noise reference:** the R3 seed 0 vs seed 1 radius difference.

**4. Small blur.**
- **Data:** held-out cells with targets of 3–6 px under C1.
- **"Attainable" if:** ≥ 80 % land within ±1 px of the target.

**What would follow.** O1 or O2 would mean the blur amount is fixable by a lookup and does not motivate a learned intervention. O3 together with a context effect or an asymmetry would identify which dependence a targeted correction must handle. Not claimed: anything about real photos, occlusion boundaries, or other K values.

## Results

Files:
- `results/confirm.json`: gate, all frozen decisions, breakdowns;
- `results/cells.json`: every cell × condition;
- `results/calib.json`: the frozen calibrations and the development measurements;
- `results/estimator_validation.json`;
- `results/confirm_sheet.png`: rows = held-out S3/S4/S5; columns = input, K 0, calm R0, busy R3, R3 swapped, R3 with C1, R3 with C2, R5 with C1.

Images and maps are in `outputs/q2_bokehdiff_confirm/`.

### Gate: passed

| check | result |
|---|---|
| estimator on these layouts | all within tolerance, worst 0.25 px; closed form matches round 18's fit |
| in-process repeat | identical (max difference 0) |
| outputs | 241, no NaN or black; 9.5 GB peak GPU |
| seed 0 vs 1 (R3) | radius differs by a median of 0.38 px, at most 1.69 px |

**Frozen calibration** (from S1 + S2):
- the K = 0 softness is 1.46 px;
- C0 gain is 1.36;
- the C1 curve (requested → measured, px) runs 0–4.8 → 1.2 (flat), 5.8 → 5.8, 7.8 → 8.3, 9.7 → 12.2, 11.6 → 18.2, 13.6 → 22.4, 15.5–19.4 → 24.6 (flat), 21.3 → 34.6, 31 → 39.9.

### 1. Calibration on held-out scenes: O3

| method | agree (384 cells) | target 0–3 | 3–6 | 6–12 | 12–20 | 20–40 | front / behind | median residual front / behind |
|---|---|---|---|---|---|---|---|---|
| uncorrected | 45 % | 92 % | 46 % | 64 % | 19 % | 28 % | 42 / 47 % | +14 / +7 % |
| C0 gain | 59 % | 92 % | 14 % | 19 % | 94 % | 81 % | 57 / 61 % | −14 / −17 % |
| C1 radius curve | **70 %** | 92 % | 70 % | 60 % | 62 % | 78 % | 72 / 67 % | −17 / −19 % |
| C2 radius + side | **70 %** | 92 % | 70 % | 60 % | 62 % | 80 % | 71 / 69 % | −17 / −19 % |

- **Calibration helps** (45 → 70 %), but no calibration reaches 80 %.
- **C2 adds nothing over C1.** Side of focus is not what the curve is missing.
- **After correction, held-out cells blur less than the target** (median −17 to −19 %). This is not a difference between development and test scenes:
  - the C1 curve fits its own data with a median residual of +1 %;
  - it predicts the uncorrected held-out outputs with −2 %;
  - the per-scene median of measured/requested radius is 1.29–1.36 for all six scenes.

  The shortfall appears only at the corrected requests, which come out 17 % below the curve's own prediction. Corrected requests are often applied at a different focus position from where the curve measured them. For example, the large targets in R5/R6 (d_f = 0.2/0.8) are mapped to requests of 19–21 px, which the curve learned at d_f = 0.5, where the same request gives more blur (see focus position below). So the blur is not a single function of the requested radius.
- One above-range cell (uncorrected); none after correction.

### 2. Symmetry: not asymmetric

- **Data:** identical patches at equal gaps, positions swapped, 48 instances, none above range.
- **Result:** median ratio behind/front 0.99; 73 % share a sign. This fails both parts of the asymmetry rule.
- **What it means for round 18:** its "more blur behind the focus" does not survive texture and position controls. Within a scene, the same patch in front of and behind the focus gets nearly the same blur (e.g. 12.1 / 11.8, 27.7 / 26.9 px).

### 3. Context: surrounding blur does not change a patch's blur

- **Anchor cells** (same patch, same 19.4 px request), calm R0 against busy R3: 0 of 24 change beyond tolerance; median change −0.8 px.
- **Focus cell against the K = 0 output:** MAE rises ≥ 1/255 from calm to busy in 12 / 12 instances. So blur elsewhere does change the decoder's fine detail.
- **Focus cell against the input:** only 4 / 12 rise ≥ 1/255 (median +0.5/255). By the rule there is **no extra damage beyond reconstruction**. The detail is reconstructed differently, not worse; this matches round 18 after its review fix.

### 4. Small blur: not attainable as specified

- **Targets of 3–6 px under C1:** 26 % within ±1 px (uncorrected: 20 %). Need 80 %.
- **Why:** the measured curve is flat at the 1.2–1.5 px floor up to a 4.8 px request, then jumps to about 6 px by 5.8 px. Blur of 2–5 px is skipped, not merely mis-scaled. The plateau's cause is still open: the 16 px attention grid is one candidate, the rest of the network another.

### What the remaining error depends on (descriptive, after seeing the data)

1. **Focus position.** At the same request, cells blur less when the focus is near the ends of the depth range than when it is in the middle:
   - 7.8 px requested: 6.1–7.7 px at d_f = 0.2 / 0.8, against 9.4–9.5 px at d_f = 0.5;
   - 5.8 px requested: 3.7–6.0 px against 6.4–6.9 px.

   In these runs d_f is tied to the cell's absolute disparity (the cells concerned sit at d = 0 or d = 1) and to the overall layout (R5/R6 have large blur on one side). This round cannot tell which of these matters.
2. **The individual patch.** At a fixed request, the six scenes spread widely. For example, 6.8 px requested gives 5.5–10.7 px, and 7.8 px gives 7.3–11.4 px. Yet the same patch is consistent front and behind and across seeds. Texture class alone does not explain it (median ratio grass 1.33, gravel 1.32, brick 1.29), but C1 does worst on brick (51 %).
   - Caveat: r̂ is a disk-equivalent radius. Validation shows the estimator is unbiased across these patches for disk blur. For a Gaussian kernel, round 18 found the reading stable across content to about ±3 %. A content-dependent reading of an unusual kernel shape cannot be fully ruled out.

## Interpretation

- **What fails, under these controlled conditions:**
  - small blur (2–5 px) cannot be produced;
  - above it, the blur for a given request depends on focus position and on the patch itself;
  - so a fixed lookup by requested radius (or radius and side) gets only 70 % of new cells right.
- **What does not fail:**
  - front versus behind;
  - interference from blur elsewhere;
  - extra damage to in-focus detail beyond the decoder's reconstruction error.
- **This narrows the problem.** The open question is local and per-region: the blur a cell receives depends on its own content and on where the focus sits, not on how busy the rest of the image is. A candidate correction must use local information (the patch and its disparity relative to the range), or else fill the 2–5 px gap.
- **Not shown:**
  - that a learned or steering correction would do better than a richer calibration, e.g. one also conditioned on d_f or on simple patch statistics (not tested; this is the next baseline);
  - anything about real photos, occlusion boundaries or other K.
