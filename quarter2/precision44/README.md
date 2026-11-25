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

## Result (`results/run/score44.json`; run 2026-11-25, 273 renders)

**Checks.**
- fp16 renders reproduce round 40 (dev) and round 42 (test) exactly (max diff 0).
- The wrapper saw every PISA call in both modes (1,890 fp16-mode and 3,024 fp32-mode calls, 18 per render). Inputs
  arrive as float16 in both modes; the fp32 mode casts them inside the wrapper.
- The log records the incoming dtype, not the compute dtype. That fp32 mode really changed the computation is shown by
  the outputs, which differ (e.g. fur_1 at K 8: 1.1 → 2.0 px).

**Part A, development curves (identical inputs and settings).** The two precisions give nearly the same response:
median fp32 − fp16 difference per K is between −0.20 and +0.12 px. Every scene has the same shape:
- a **dead zone**: about 0.1-1.2 px of blur until K passes a scene-specific threshold (K 8 for fur_1, K 20 for thin_0);
- then a roughly linear rise.

| | fp16 | fp32 |
|---|---|---|
| scenes where some grid K gives 4 ± 2 px | 8/8 | 8/8 |
| "jumping" scenes (jump > 3 px between neighbouring K, both ≤ 8 px) | 3/8 | 1/8 |

fp32 rounds the corners slightly (e.g. hair_1 at K 10: 1.3 → 2.3 px). The jump count is partly the grid's fault: above
K 16 the grid steps by 4-8, so a steady slope also produces jumps over 3 px.

**Part B, test diagnostic (round 42's 24 used scenes, same depth-rule K).**

| | fp16 | fp32 | fp32 − fp16 [95% CI] |
|---|---|---|---|
| within tolerance, 4 px | 0.04 | 0.08 | +0.04 [0.00, +0.13] |
| 8 px | 0.79 | 0.75 | −0.04 [−0.13, 0.00] |
| 12 px | 0.71 | 0.71 | 0.00 |
| all | 0.51 | 0.51 | 0.00 [−0.04, +0.04] |
| median abs error | 1.98 px | 1.86 px | |
| subject hf_ratio | 0.62 | 0.62 | 0.000 [−0.003, +0.003] |

**Reading (pre-set).**
- **4 px precision effect at identical settings: NOT SHOWN** (+0.04).
- No additional subject change.
- Precision does not explain round 42's 4 px misses. A setting that reaches 4 px exists in every development scene, in
  both precisions; the depth rule simply asks for a K inside the scene's dead zone.

**Post hoc, development scenes only (not a test; motivates the next round).**
- Leave-one-scene-out on the development curves: a monotone (isotonic) curve of blur against K × gap reaches 4 px in
  7/8 held-out scenes, in fp16 and in fp32 alike.
- Round 42's offset line reaches 0/8, even though it was fitted on all 8 scenes (which favours it).
- 8 px: 7/8 both. 12 px: 7/8 monotone vs 3-4/8 line.
- Outcomes are interpolated between grid K, not rendered.

**Next round (to be frozen from this one).**
- Precision: fp32, as review #7 advised (no cost to the subject; slightly smoother low end).
- Controller: a monotone curve of blur against K × gap, fitted on the 8 fp32 development curves.
- Test once on fresh generated scenes, against the round 42 offset rule (fp32) and the fixed setting. Search and feedback
  choose K on even tiles and are scored on odd tiles only.
