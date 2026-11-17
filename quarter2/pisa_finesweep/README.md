# Q2 round 21: does BokehDiff skip mild blur, or does it need a very narrow range of requests?

**Question.** Rounds 19–20 found almost no blur for small requests, then a steep rise. If we step the request finely through that rise, can the editor deliver 2, 3, 4 and 5 px of blur on one fixed scene? Or does it jump over some of them?

**Status.** Run Nov 17 (pod `q2-finesweep`, 90 images in 3 min). The test's own checks all passed. Plan and rules were committed (b132a1a) before the run.
- **Released fp16 inference path: F2 (skips).** The response is a staircase that jumps over 3 px.
- **fp32: F1 (continuous but narrow).** 2, 3, 4 and 5 px are all reachable, but only within about 2 px of requested radius (4.9–6.8 px).
- **The staircase's steps sit exactly where half-precision rounding of PISA's neighbour gate predicts.**

## Test (`run21.py`)

**Fixed:**
- scene S3 from round 19;
- geometry: focus d_f = 0.5, anchor pair at ±0.5, the other three pairs (6 cells) at gap ±0.15;
- inputs and seeds;
- BokehDiff unchanged, at commit `38ff937`.

**Varied: only the blur strength K** = 33.0, 33.7, …, 47.0 (21 levels). That moves the requested radius r_nom = 0.969 · K · 0.15 from 4.79 to 6.83 px in 0.10 px steps.
- **Precision modes:** PISA attention as shipped (fp16, the primary result) and in fp32 (secondary), using round 20's wrapper.
- **Seeds:** 0 and 1, plus K = 0 per mode and seed.
- **Total:** 88 images, plus 2 checks.

**Measure:** r̂ of each of the 6 swept cells (round 19's estimator) and their median per level, both seeds pooled. Also recorded:
- the request each cell needs to reach each target (linear interpolation on its own curve);
- the spread of those requests across the 6 cells;
- the slope of the rise (px of blur per px of request);
- focus-cell MAE against the input.

## Rules (fixed now), per precision mode

**Gate:**
- fp16 at K = 40 reproduces round 20's matching output exactly (≤ 1/255);
- the repeat is identical;
- no NaN or black outputs;
- estimator validation passes.

**Definitions:**
- A target t ∈ {2, 3, 4, 5} px is **reachable** if some level's median r̂ is within ±0.5 px of t.
- A target is **skipped** if two neighbouring levels (0.1 px of request apart) straddle it, below t − 0.5 and above t + 0.5, with a step > 1 px.

| outcome | rule | meaning |
|---|---|---|
| F1 | every target reachable and no step > 1 px | continuous but narrow: mild blur exists, it just needs precise requests. A calibration and transfer problem (round 19: a fixed curve gets 70 % on new scenes) |
| F2 | some target skipped | missing values under this configuration: stronger evidence that mild blur cannot be dialled in |
| F3 | neither | reported as observed |

**Interpretation limits:** one scene, one geometry, K varied around 40. A target reachable here does not mean the same request works on other scenes (round 19).

## Results

Files:
- `results/finesweep.json`;
- `results/cells.json`;
- `results/finesweep_sheet.png`: 128 px crops of one swept cell; input, then every other level, fp16 row and fp32 row.

Images are in `outputs/q2_pisa_finesweep/`.

**Gate passed:**
- fp16 at K = 40 is identical to round 20;
- the repeat is identical;
- 90 outputs, none bad;
- estimator validation passed.

Seed 0 and seed 1 differ by a median of 0.2 px.

### Median r̂ of the 6 swept cells (both seeds) against requested radius

| r_nom (px) | 4.8 | 4.9 | 5.0 | 5.1 | 5.2 | 5.3 | 5.4 | 5.5 | 5.6 | 5.7–6.2 | 6.3 | 6.4–6.5 | 6.6 | 6.7 | 6.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 (shipped) | 1.7 | 1.7 | 1.7 | 1.7 | 1.7 | 1.7 | 1.7 | 1.7 | **3.9** | 3.9 | **4.6** | 4.5 | **5.1** | 5.0 | **5.5** |
| fp32 | 1.8 | 1.8 | 1.9 | 2.1 | 2.4 | 2.6 | 2.8 | 3.0 | 3.1 | 3.3–4.2 | 4.4 | 4.6–4.7 | 4.9 | 5.1 | 5.4 |

**fp16, as shipped (F2).** The output takes only five values in this range: 1.7, 3.9, 4.5, 5.05 and 5.5 px.
- The first step jumps 2.2 px between requests 5.51 and 5.61, so **3 px is skipped**.
- 2 px is "reachable" only within the ±0.5 px tolerance of the 1.7 px floor.
- All 6 cells step at the same requests. Every cell has its own levels, but none lands between floor and about 3.6 px.

**fp32 (F1).** The output rises smoothly, with the largest step 0.32 px per 0.1 px of request. All four targets are reachable.
- **Narrow:** the slope is about 1.9 px of blur per px of request. Outputs of 2 → 5 px need requests of only about 5.0 → 6.7 px.
- **Below about 4.9 px the response is still flat at the floor.**
- **Per-cell request needed** (min–max over the 6 cells):

  | target | 2 px | 3 px | 4 px | 5 px |
  |---|---|---|---|---|
  | request needed | 4.8–5.3 | 5.41–5.67 | 5.90–6.19 | 6.35–6.78 |

  These spread by 0.3–0.5 px of request, which is about 0.6–0.9 px of output at this slope. Even within one scene the patches need slightly different requests; the two brick cells have the lowest floor (≈ 0–0.3 px) and start lowest.

**Focus cell:** MAE against the input is 2.8/255 in both modes, so no additional in-focus error was measured.

### Mechanism: the staircase is fp16 rounding of the neighbour gate

In fp16, PISA's neighbour gate sigmoid(ln 10⁷ · (coc − 1.65)) can only be 0 or a whole multiple of the smallest fp16 number, 2⁻²⁴ ≈ 6·10⁻⁸. It rounds up to n of these units at:

| gate becomes | 1 unit | 2 units | 3 units | 4 units |
|---|---|---|---|---|
| predicted request (px) | 5.58 | 6.25 | 6.55 | 6.75 |
| observed jump between requests (px) | 5.51 → 5.61 | 6.22 → 6.32 | 6.52 → 6.63 | 6.73 → 6.83 |

All four observed steps fall exactly where predicted. In fp32 the gate is continuous, and so is the output.

### Revised reading of round 20

Round 20's frozen outcome (P3) stands: fp32 did not raise blur at requests of 2.9, 3.9 and 4.8 px.
- **This round shows what precision does:**
  - in fp16 it turns the mild-blur response into a few discrete levels and skips values such as 3 px;
  - fp32 makes it continuous.
- **What precision does not change:**
  - the flat region below about 4.9 px of request;
  - the steepness above it.

  Those come from the gate's shape on the 16 px token grid, or from the rest of the network. That question is still open.

## Interpretation

Two separate problems, now distinguishable:
1. **An implementation issue (the released fp16 inference path, in our configuration):** mild blur is quantized and some values cannot be dialled in. Computing PISA's attention in fp32 fixes this, at no measurable runtime cost (round 20: 1.42 s per image either way). This is an engineering finding, not a reason for a new steering method.
2. **A control-range issue (fp32):**
   - all mild blur lives within about 2 px of requested radius, with a slope near 2;
   - below that there is no response;
   - the needed request differs between patches (here by 0.3–0.5 px; across scenes, round 19's calibration reached only 70 %).

   Mild DoF is reachable but precise only with per-patch tuning in a narrow window. This is the problem a targeted intervention (or richer calibration) would have to address.

**Not shown:**
- one scene;
- one geometry;
- K around 40;
- the per-cell spread in other scenes (round 19 suggests larger).
