# Q2 round 19: can a simple calibration predict and correct BokehDiff's blur on new scenes?

**Question.** Round 18 found that BokehDiff's blur is ordered correctly by depth but not proportional to the request. Can a simple calibration predict and correct the blur on new scenes, or does the same request produce different effects depending on its surroundings?

**Status.** Plan and decision rules fixed before any output exists.

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

Pending.
