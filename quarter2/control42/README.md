# Round 42: can a depth-aware controller give BokehDiff reliable, image-specific blur control?

Branch `q2-control42` (from `q2-compare40`). Proposed in review #6 after round 40: a fixed aperture-like setting
*should* give different blur at different depths, so the useful problem is translating "about 8 px of background
blur" into a setting for each scene, without changing the model or damaging the subject.

## Design (frozen before any round 42 output; `fit42.py`, `bd42.py`, `score42.py`, `dsmlp42.sh`)

**Scenes.** The 24 initially-sharp, held-out generated test scenes of round 40 (no mask failure or measurement flag
among them). Targets 4, 8 and 12 px of additional background blur. Development scenes are the 8 round 40 mapping scenes.

**Arms** (all BokehDiff, round 40 driver unchanged, only K differs):

| arm | how K is chosen | renders |
|---|---|---|
| fixed | round 40's one K per target (outputs reused; first 3 scenes re-rendered as a determinism check) | 1 |
| depth | frozen depth rule from the inputs only | 1 |
| correct | depth K, measure the delivered blur on half the background tiles, adjust K once, render again | 2 |
| oracle | render the K grid, pick the crossing of the target, render there: an offline upper bound, not a controller | 10 |

**Depth rule** (`results/rule42.json`, fitted on development points only):
- gap = median over the behind tiles of |depth − focus|, from BokehDiff's own preprocessing, available before rendering;
- two forms compared by leave-one-scene-out error on development points with 2–20 px blur: proportional
  (r = a·K·gap) vs offset (r = b·(K·gap − c)). They were nearly tied (median error 2.10 vs 2.09 px), and the
  pre-set criterion chose the offset form: **r = 0.950 × (K × gap − 0.739)**, so K = (target / 0.950 + 0.739) / gap;
- K clipped to [4, 64]. On the test scenes the planned K runs from 7.0 to 58.9, none clipped (`results/plan42.json`).

**Disclosure.** The proportional form was examined post hoc on round 40's test outputs (correlation 0.88). The rule
here is chosen and fitted on development data only, but the test scenes are the same 24.

**Correction arm.** Feedback uses the even-indexed behind tiles; the odd-indexed tiles are never seen by the
controller, and every arm is also scored on them as a check against the controller tuning itself to its own
measurement. K2 = K1 + (target − measured) / (0.950 × gap), clipped to [4, 64]; no correction if the measurement fails.

**Scoring** (`score42.py`; round 40 estimator and measures): target attainment (within max(2 px, 20%)), absolute
error, subject detail kept (hf_ratio), colour shift, focus check, renders and runtime. Each scene is aggregated
first, then bootstrapped over scenes for 95% intervals.

**Decision rules.**
- D1, depth vs fixed (paired scene-level within-tolerance fraction): SUBSTANTIAL if the gain is ≥ +0.25 and its
  interval excludes 0; IMPROVED if only the interval excludes 0; otherwise NOT SHOWN.
- D2, subject not damaged: paired hf_ratio interval lower bound ≥ −0.05 (depth vs fixed, correct vs fixed).
- D3, does one measured correction add? Correct vs depth, same thresholds as D1, on all tiles and on held-out tiles.
- D4, oracle: the ceiling for per-image K selection.
- D5, a learned predictor (e.g. a CNN) is considered justified only if the depth rule misses ≥ 1/3 of cells and
  ≥ 50% of its log error is between scenes (consistent per-scene errors). A necessary condition, not proof.
- Determinism: re-rendered fixed outputs must match round 40 within 2/255, otherwise the comparison is provisional.

**CPU dry run** (round 40 images as stand-ins for the arms): the scorer, pairing and decisions run end to end.

**Order.** Launched after round 41 step 1 finishes (one GPU at a time).
