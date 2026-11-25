# Round 45: fresh-scene confirmation of the monotone blur controller (final round of the controller workstream)

Branch `q2-confirm45` (from `q2-precision44`). Round 44 showed that fp32 attention alone does not fix round 42's 4 px
misses at the tested settings (it does not rule out precision effects in general). Every development scene has a
scene-specific dead zone before blur rises. A monotone curve fitted on the development curves reached 4 px in 7/8
held-out development scenes, but only by interpolation between rendered settings, not by corrected renders. This round
renders the selected settings on fresh scenes. If it succeeds, the controller is written up; if not, its limits are
reported and no further controller search follows.

## Design (frozen before any round 45 image exists)

**Scenes.** 24 fresh generated scenes (`prompts45.py`): the same 4 initially-sharp categories, generator and settings as
round 40, with new subjects, backgrounds and seeds (4500+). Masks and depth from BokehDiff's own preparation. Regions
from round 40's `regions40.py`, unchanged.

**Coverage.** Every loss is counted: generated, prepared, eligible (≥ 4 behind tiles), excluded by the input-only review,
scored.

**Input-only review.** Mask failures and measurement flags are decided on the input contact sheets and committed as
`review45.json` before any output is pulled or viewed.

**Measurement tiles.**
- The behind tiles are split into selection tiles (even-indexed) and evaluation tiles (odd-indexed).
- No arm in this round measures its own output, so the selection tiles are reserved and unused.
- Every arm is scored on the same evaluation tiles. There is no fallback, so a scene with fewer than 2 evaluation tiles
  is not scored.
- All-tile results are secondary.

**Arms** (all fp32 PISA; K planned from inputs only, `plan45.py`; one render per cell):

| arm | K | role |
|---|---|---|
| monotone | isotonic curve of blur against K × gap, fitted on round 44's 8 fp32 development curves (`fit45.py`, `results/rule45.json`). Linear interpolation between knots; first crossing of the target; out of range → end knot (flagged); K clipped to [4, 64] | tested controller |
| linear | round 42's frozen offset rule (r = 0.950 (K·gap − 0.739)), unchanged | primary comparison |
| fixed | round 40's one setting per target | secondary baseline |

The monotone form was fixed in advance. Only one curve is fitted, with no selection among curves.

**Precision.** The fp32 wrapper (rounds 20-22, 44) now asserts, inside the attention call, that PISA's attention result
is float32, and logs input and compute dtypes for every call.

**Outcomes.**
- Primary: scene-averaged target attainment over 4, 8 and 12 px on the evaluation tiles, monotone − linear, paired
  scene bootstrap.
- Each target is also reported separately.
- Also: |error|, additional subject-detail loss (hf_ratio), colour change and render time, relative to linear and to fixed.

**Decision (frozen).**
- A: monotone − linear attainment ≥ +0.15, with CI lower bound > 0.
- B: no target deteriorates (paired point estimate ≥ −0.10 at each of 4, 8 and 12 px).
- C: paired hf_ratio CI lower bound ≥ −0.05 and paired colour-shift CI upper bound ≤ +1.0 (0-255 units).
- **CONFIRMED** = A, B and C. **PARTIAL** = A only. **NOT CONFIRMED** = not A.

**Dry run** (CPU; round 40 test scenes as stand-in data): the plan reproduces round 42's linear K exactly, and the scorer
and decision run end to end.

Renders: 24 × 3 targets × 3 arms = 216, about 45 min, plus about 1.5 h of generation.
