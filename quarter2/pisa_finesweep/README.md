# Q2 round 21: does BokehDiff skip mild blur, or does it need a very narrow range of requests?

**Question.** Rounds 19–20 found almost no blur for small requests, then a steep rise. If we step the request finely through that rise, can the editor deliver 2, 3, 4 and 5 px of blur on one fixed scene? Or does it jump over some of them?

**Status.** Plan and rules fixed before running.

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

Pending.
