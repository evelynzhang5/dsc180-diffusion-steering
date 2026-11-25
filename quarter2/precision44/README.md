# Round 44: does fp16 PISA attention explain BokehDiff's weak-blur misses? (precision diagnostic)

Branch `q2-precision44` (from `q2-control42`). Review #7 found that rounds 40-43 ran BokehDiff's PISA attention in fp16
as shipped, without round 20-22's fp32 wrapper. Round 21 showed that fp16 PISA turns mild blur into a staircase that
skips about 3 px. Round 42's depth rule missed almost every 4 px target. This round separates precision from calibration.

## Design (frozen before any round 44 output; `bd44.py`, `score44.py`, `dsmlp44.sh`)

**Only PISA's attention function differs between the two modes.** fp16 = as shipped (a pass-through that logs dtypes);
fp32 = round 20-22's wrapper (inputs cast to float32 with autocast off inside PISA, result cast back). Every call's dtype
is logged per mode, so the precision actually used is recorded rather than assumed.

**Part A, development curves.** Round 40's 8 mapping scenes × K = 4, 6, 8, 10, 12, 14, 16, 20, 28, 36, 48, 64, in both
modes, at identical inputs and settings. fp16 at round 40's grid K must reproduce round 40's outputs (≤ 2/255).
Reported per scene and mode:
- the response curve;
- the largest jump between neighbouring K in the weak range (both responses ≤ 8 px); a scene is "jumping" if it is > 3 px;
- whether any K gives 4 ± 2 px.

**Part B, test diagnostic (already-used scenes).** Round 42's 24 test scenes × 4/8/12 px at round 42's depth-rule K,
in fp32. The fp16 counterparts are round 42's outputs (3 scenes re-rendered as a check). Reported:
- within-tolerance rate per target;
- absolute error;
- additional subject change (hf_ratio, colour shift), paired fp32 − fp16 with scene bootstrap.

**Reading (descriptive; no controller is chosen here).**
- 4 px effect at identical settings: LARGE (≥ +0.25, CI excludes 0), PRESENT (CI excludes 0), or NOT SHOWN.
- NOT SHOWN would not mean fp32 is ineffective, since precision can shift the whole response; recalibration happens either way.
- Continuity: jumping scenes, fp16 vs fp32. Subject: paired hf_ratio change.

**What comes next regardless of the outcome** (frozen after this round, before any of its results are used for testing):
- Fit the controller on **fp32** development curves, adding a monotone low-end fit if the offset rule still misses
  at 4 px.
- Test it once on fresh generated scenes. Search and feedback choose settings on one tile subset and are scored on
  the other, with no fallback that lets scoring tiles into selection. Scenes without enough tiles are reported separately.

Renders: 8 × 12 × 2 = 192 (part A) + 72 + 9 (part B) ≈ 273, about 60-75 min on the 1080 Ti.
