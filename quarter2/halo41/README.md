# Round 41: is BokehDiff's "sharp halo" a model failure? (step 1: depth-input control)

Branch `q2-halo41` (from `q2-compare40`). Goal set by the user after round 40: find one clear failure of BokehDiff, then
propose and test a change that improves the original model.

## The candidate failure (from round 40)

Round 40's matched-blur comparison found BokehDiff's `halo_sharp` higher than Bokehlicious's in 8 of 11 scenes (median
0.22): the 2-12 px band just outside the subject keeps more detail than the far background.
- Outputs: `outputs/q2_compare40/s3/bd_test/<scene>_t<target>.png` (inputs `outputs/q2_compare40/data/input/`).
  Highest-halo matched cells: hair_5 (t4, t12), fur_2 (t4), hair_6 (t4), highlights_6 (t4, t8).
- Zoomed crops with the measured band in magenta: `figures/failure_halo.jpg`; where the crop is: `figures/failure_*_where.jpg`.

**Caveat found while preparing this round (before any round 41 output).** Wider crops suggest much of the measured
halo may be correct physics, not a defect:
- hair_6: the stone wall beside the subject stays sharp while the far alley blurs, and the wall is at nearly her depth.
- highlights_6: the nearest people in the crowd stay sharper.
- fur_2: the band partly lies on the animal's own hair strands outside the mask.
- `halo_sharp` compares the band with the FAR behind tiles, so background genuinely near the focus depth counts as
  halo. The depth gate (|d - focus| >= 0.15) removes only the nearest of it.

## Step 1 design (frozen; `bd41.py`, `measure41.py`, `dsmlp41.sh`)

Same driver, scenes and settings as round 40. Only the depth map handed to BokehDiff changes:

| condition | subject | outside the subject |
|---|---|---|
| A original | as round 40 | as round 40 |
| B two_layer | focus | one value: the median over the frozen behind tiles, so the requested blur is identical everywhere |
| C hard_edge | focus | as A, except within 24 px of the mask: copied from the nearest pixel >= 24 px away |

- Scenes: the 8 round 40 mapping scenes (development only). K: round 40's three frozen settings. 72 renders.
- CPU check before launch: B has a single outside value in every scene; C equals A beyond 24 px. In fur_0,
  highlights_1 and thin_0 the original depth just outside the subject is much nearer the focus than the far tiles.
- Measures (unchanged round 40 code): halo_sharp on the same band pixels in every condition, background blur on the
  behind tiles, subject high-frequency retention.
- Decision rule: see the `measure41.py` docstring.
  - INPUT: the halo follows the depth. Not a model failure; pick a different failure.
  - MODEL: the halo survives a flat background request, so it is made inside the model. Next: an in-model change
    (relax the occlusion-aware blending near the edge, or steer the edge activations toward background ones).
  - MIXED: both contribute.

## Backup candidates if the halo is not a model failure (looked at 2026-11-24, while step 1 ran)

- **In-focus subject loses fine detail** even when almost no blur is requested (K = 4): it keeps 0.56-0.79 of the
  finest-scale detail, vs 0.42-1.05 for Bokehlicious at f/16. At 2.5x zoom the difference is barely visible (grain
  and pores), so this is weak as a "clear" failure.
- **Blur in front of the subject**: 7 scenes; sometimes under-blurred, but behind-tile blur misses in the same scenes.
  Mixed, so weak.
- **Transparent subjects** (`figures/candidate_transparent.jpg`; top: highlights_3 at 12 px, input / BokehDiff / mask;
  bottom: rb_48 at f/2.0, input / BokehDiff / real f/2.0 capture). BokehDiff keeps the garden seen through the glass
  sharp, and on rb_48 leaves the shelves behind the clear acrylic sign sharp, where the real lens blurs them. One
  depth per pixel cannot say "sharp surface, far-away scene behind it", so the conditioning cannot ask for the right
  answer. Only 2 examples so far.

## Step 1 result (8 mapping scenes × 3 targets, `results/measure41.json`)

| scene | halo original | halo two-layer depth | halo hard-edge depth |
|---|---|---|---|
| fur_0 | 0.348 | 0.099 | 0.351 |
| fur_1 | 0.167 | 0.188 | 0.194 |
| hair_0 | 0.025 | 0.049 | 0.032 |
| hair_1 | 0.328 | 0.334 | 0.331 |
| highlights_0 | 0.296 | 0.200 | 0.210 |
| highlights_1 | 0.469 | 0.430 | 0.519 |
| thin_0 | 0.148 | 0.043 | 0.156 |
| thin_1 | 0.164 | 0.071 | 0.163 |
| **median** | **0.231** | **0.143** | **0.202** |

**Pre-set verdict: MIXED.** A flat background depth lowers the median halo by 38% and lowers it in 5/8 scenes:
neither the "follows the depth" threshold (≤ 50%, ≥ 6/8) nor the "model" threshold (≥ 75% or ≤ 4/8) is met.
Background blur is unchanged within tolerance in every scene. Hard-edging the depth within 24 px of the mask barely
changes anything (0.231 → 0.202), so smoothed depth at the subject edge is not the cause.

**What the remaining halo is.** In fur_0, thin_0 and thin_1 the halo disappears once the depth map says the background
is far: there it was near-focus background, as suspected. In the scenes where it persists (hair_1, highlights_1, fur_1),
the sharp band is the subject's own material just outside the mask: flyaway hair strands (`figures/hair_1_41.jpg`)
and the machine's outer edges (`figures/highlights_1_41.jpg`). BokehDiff keeps these sharp even when the depth input
says "far", which looks correct to the eye rather than defective.

**Conclusion.** Round 41 does not establish a halo failure. The metric is dominated by near-focus background and by
mask imprecision. The halo is dropped as a failure candidate; transparent subjects stay the strongest remaining
candidate, and the control question moves to round 42.
