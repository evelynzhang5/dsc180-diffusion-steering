# Quarter 2: What Makes a Steering Intervention Control the Final Image?

**Status (Nov 8, 2026):** the text-to-image pilots (SD 1.5, `sd_pilots/`) are finished and archived as a
bounded negative result: the tested nudges showed no robust advantage over random pushes. The work now continues
on MNIST (`mnist/`): when an attribute is readable from the generator's activations, what decides whether an edit
changes it and whether the change lasts? The original question (predicting when an intervention helps) and the
smoke-test plan are kept below for the record. Quarter 1 results and the earlier MNIST attribute-preservation idea
are in [../notes.md](../notes.md).

## Smoke test 1 result (Nov 5, 2026): too hard for SD 1.5, so inconclusive

Full report: `outputs/q2_smoke/main/report.md` (not in git). Automated scores, not yet audited by hand; a
spot check of 24 images agreed with the main failure modes.

- **Baseline success 4.2%** (8/192), below the 10% minimum. Both objects found 51%; both colors right
  given both objects 14%; relation right given both objects 48%, which is chance level.
- **SD 1.5 nearly ignores "left/right":** swapping it in the prompt, same seed, changes the image by
  only 15/255 per pixel on average. Failures are mostly missing objects and color bleeding.
- **Pulses barely change anything:** 91–99% of branches keep the same outcome. Picked strengths:
  global CFG 14, targeted 40, each chosen on a single dev repair, so the pick is noise.
- **Oracle minus best fixed policy: 2.6 points** (95% interval 0.5 to 4.2). That is below the 5-point
  bar, but the floor effect makes the result uninformative, not a clean "no".

Next try: an easier failure type where SD 1.5 sits at 30–70% success (e.g. two-object color binding
without a spatial relation, or counting), stronger or longer pulses, and the blinded audit of the
color scorer.

## Pilot 2 result (Nov 7, 2026): modest headroom, needs a placebo control

`dsmlp_pilot2.sh`, 6 h 50 min on a GTX 1080 Ti. Reports: `outputs/q2_pilot2/<task>/main/report.md` (not in
git). Automated scores, not yet audited by hand.

| | two objects | counting (2–4) |
| --- | --- | --- |
| baseline success | 74.5% | 41.1% |
| picked on dev | global CFG 14, targeted 15 (3 steps) | global CFG 20, targeted 15 (3 steps) |
| best fixed policy (dev-chosen) | 75.0% | 43.2% |
| per-sample hindsight oracle | 81.8% | 51.6% |
| **oracle minus best fixed** | **5.7 pts (95% CI 2.1–7.8)** | **8.3 pts (3.6–10.9)** |

- **Robust to the scorer's settings:** the gap stays 5.7–8.9 (two objects) and 6.8–8.3 (counting) for
  detection thresholds 0.15–0.3, and 8.3 for counting without books and balloons.
- **Every fixed action repairs about as often as it harms** (e.g. counting at step 6: 5.7% repairs, 3.6%
  harms), so a single rule gains at most ~2 points. Longer (8-step) and stronger pulses did not help on dev.
- **Many flips are the scorer, not the image:** 92% of counting flips and 47% of two-object flips involve
  a detector score near the 0.2 cutoff (0.1–0.3), and a quarter barely change the image (< 2/255 per
  pixel). Example: `t03_s4` (umbrella and dog) fails with umbrella score 0.05, while six near-identical
  branches all pass at about 0.81. So "repaired by several branches" can mean "borderline verdict", not
  "repairable state". See `nudges_explained.ipynb`, section 6.
- **Scorer spot check (24 images per task):** two-object scores look right except borderline cases;
  counting is right for dogs, apples, birds, sometimes one too many for cats, cups, cars, and broken for
  books and balloons (always "many").

Against the go/no-go table this sits between "weak" (< 5) and "strong" (10+). The main open question: is
the oracle gain targeted repair, or just what any random re-roll of the image would give (best of 7
tries)? Next run: a placebo pulse (random direction, same size) as a control, more seeds, counting
objects that the scorer handles, and the blinded hand audit.

## Scorer checks on pilot 2 (Nov 7, 2026, local): counting's gap is mostly borderline verdicts

`scorer_check.py` and `early_check.py`, run on a laptop from pilot 2's saved images and step 6/15/24 latents
(seeds 0-7, counting without books and balloons). Reports: `outputs/q2_scorer_check/`, `outputs/q2_early_check/`.

- **Counting only flips that clear the 0.2 cutoff by a margin:** at margin 0.05 / 0.1 the gap falls from
  5.7 to 5.2 / 3.6 points (two objects) and from 8.3 to 2.8 / 0.7 (counting).
- **Re-scoring after a mirror flip and a 90% crop:** baseline verdicts hold for 97% (two objects) and 87%
  (counting). Repairs hold in all three views for 15 of 30 (two objects) and 9 of 24 (counting); harms for
  16 of 21 and 4 of 24.
- **The model's early preview predicts the final outcome:** decoding the step's predicted final image and
  scoring it gives AUC 0.79 / 0.83 at step 6 and 0.96 at step 15. But the images a nudge "repairs" are the
  failures whose preview was already close to passing, and nudging only when the preview fails gains nothing
  (within 1 point of always nudging).

So two objects keeps a small (about 3-5 point) gap that survives stricter scoring; counting's gap is mostly the
detector's borderline calls. The placebo comparison (pilot 3) is the test of what remains.

## Pilot 3 and audit (Nov 7, 2026): the nudges show no robust advantage over random pushes

`dsmlp_pilot3.sh`, 9 h 35 min on a GTX 1080 Ti: placebo pulses (random direction, 15% and 50% of the noise
prediction) and seeds 8-15. Reports: `outputs/q2_pilot2/<task>/main/report_pilot3.md`, margins in
`outputs/q2_scorer_check/pilot3_<task>/`.

| hindsight oracle, 6 options per image | two objects | counting (2-4, no books/balloons) |
| --- | --- | --- |
| no intervention | 74.5% | 54.9% |
| nudges picked on dev | 82.3% | 70.1% |
| placebo | 80.5% | 62.5% |
| **picked minus placebo** | **+1.8 (-0.8 to +5.2)** | **+7.6 (+4.5 to +11.1)** |
| same, flips must clear the cutoff by 0.05 / 0.1 | +1.8 / +1.3 | +2.1 / +0.7 |
| re-roll the seed, best of 4 (about the same cost; also a hindsight oracle) | 95.8% | 88.2% |

- **Two objects:** not distinguishable from placebo. **Counting:** the lead comes from borderline detector calls.
- **Blinded audit** (`make_audit.py`, 12 counting flip pairs, 6 real and 6 placebo, chosen at random): in 11 of 12
  pairs the person counted the same number before and after; 0 of 7 sampled automated repairs confirmed (the one
  real change was a harm, 4 -> 3). On these borderline images the scorer matched the person on 13 of 24. A small
  sample of borderline flips: strong evidence against those cases, not a precise rate for all repairs.
- **Re-rolling the seed** beats every nudge menu by 13-18 points at about the same compute, but both are hindsight
  oracles: a usable system would need a selector without ground truth (e.g. the detector itself), not yet measured.

Conclusion (bounded): the tested pulses (global CFG and targeted, 3 steps, SD 1.5) show little demonstrated,
robust benefit over random pushes in this setup, so a predictor of their benefit has little to predict. The SD
pilots are archived here; a wrong-sign control (`dsmlp_pilot4.sh`) is written but deliberately not run.

## MNIST thickness diagnostic (Nov 7, 2026, local): readable is not the same as steerable

`mnist_thickness.py` on the frozen conditional MNIST generator (`generator.pt`, w = 1, 1,000 digits, dev seed 0).
A within-digit thick-minus-thin direction is learned from real digits (split before noising) at two blocks (`d3`,
the NA-RFM choice; `u2`, a decoder block) and three noise levels, then added to the block's output as
strength x |h|. Compared with its negative and 4 random directions of the same size. Thickness change in units of
the within-digit SD of real digits (0.87 px). Reports: `outputs/q2_mnist_thickness/steps{16,64}/`.

| `u2`, strength 0.5 | t = 0.25 | t = 0.5 | t = 0.75 |
| --- | --- | --- | --- |
| readable: probe R^2 on half B (penalty picked on B; corrected values in the transfer test) | 0.57 | 0.82 | 0.87 |
| immediate change in the clean-image guess (random: <= 0.05) | +1.2 | +0.5 | +0.2 |
| final change, edit on one step (16 / 64 steps) | +0.19 / +0.05 | +0.09 / +0.02 | +0.05 / +0.01 |
| final change, edit held on every remaining step (16 / 64 steps) | +1.4 / +1.4 | +0.6 / +0.6 | +0.2 / +0.3 |
| accuracy drop, held edit (16 steps) | 0.7% | 0.2% | -0.2% |

- **For this direction, normalization and attribute, readability and leverage move in opposite directions:**
  thickness is easiest to read late (R^2 0.88) where the same edit changes it least (about 7x less than early).
  Not yet a general mechanism: late edits have less sampling left, and equal activation norms need not give equal
  changes in the denoiser's output.
- **A one-step edit leaves little in the final image:** 10-30% of its immediate effect at 16 steps, 4-8% at 64
  steps. This mixes two things: how much change enters the trajectory (a step covers less time at 64 steps) and
  whether later steps attenuate it. A held edit gives the same final change at both step counts, which shows
  sustained control and some robustness to step count, not retention after release; NA-RFM (Appendix D) already
  finds that longer steering windows work better, so holding is a baseline, not a contribution.
- **The learned direction is specific:** 20-50x the effect of random directions of the same size, with digit
  identity intact except for strong early edits (strength 1.0 held from t = 0.25 at `u2`: accuracy -17 to -20%,
  random -0.4%). Thickening is easier than thinning early at large strength (+3.3 vs. -1.3 at strength 1.0;
  nearly symmetric at 0.25, possibly a floor on stroke width), and `u2` responds about 3x more than `d3`.

### Release-and-follow test (Nov 7, 2026): the change does not fade after release; less of it gets in late

`mnist_release.py`: the edit is on for the same stretch of time, t in [t0, t0 + 0.125) (2 of 16 steps or 8 of 64,
same per-step amplitude), then off; the edited and unedited trajectories are followed to the end. Report and
curves: `outputs/q2_mnist_release/`.

| `u2`, strength 0.5, 16 / 64 steps | t0 = 0.25 | t0 = 0.5 | t0 = 0.75 |
| --- | --- | --- | --- |
| thickness gap of the guess while the edit is on | +1.2 | +0.56 | +0.18 |
| thickness gap at release (edit off, unedited call on the edited state) | +0.39 / +0.38 | +0.19 / +0.17 | +0.09 / +0.09 |
| thickness gap at the end | +0.37 / +0.36 | +0.18 / +0.16 | +0.10 / +0.09 |
| kept after release (+d / -d) | 0.93 / 1.00 | 0.94 / 1.05 | 1.02 / 1.00 |
| random directions, gap at the end | -0.01 | 0.00 | 0.00 |
| accuracy drop | 0.1% | -0.1% | -0.1% |

- **No attenuation of the measured attribute after release:** once the edit is off, the mean thickness
  difference in unedited clean-image guesses stays about flat through the final image (end / release 0.85-1.13
  across layers, starts, signs and both step counts). The earlier "one-step edits fade" was the edit's direct
  effect on the network's guess disappearing when switched off, not later steps undoing it. These are ratios of
  averages; the per-image picture is in the transfer test below.
- **Step count does not matter at matched duration:** 16 and 64 steps agree within 0.02 SD everywhere.
- **What shrinks late is sensitivity:** the same relative activation edit moves the guess about 7x less at t0 = 0.75
  than at 0.25, while the sampler's per-step weight on the guess only falls about 2x (DDIM: 1 - tan(pi t/2) /
  tan(pi s/2) for a step t -> s). In the local model dz_(k+1) = A_k dz_k + B_k u_k, the thickness component behaves
  as if it is carried along (not that A_k is the identity: the full state difference grows after release) while
  the injected part B_k u_k falls with t. Caveat: each t0 here uses a direction learned at that t0, so this
  compares different directions; the transfer test separates the two.
- **The state change is thickness-aligned:** per unit of state change at release, the learned direction gives
  4-6 SD of thickness, random directions about 0. The state difference itself keeps growing after release (x1.5-3,
  for random directions too), so state norm is not the right persistence measure; the attribute is.

### Direction-transfer matrix (Nov 7, 2026): the late decline belongs to when you apply, not when you learned

`mnist_transfer.py`: directions learned at t = 0.25, 0.5, 0.75, each applied at each of those times (same layer,
window 0.125, strength 0.5 x |h|, then off), plus 8 random directions; per-image results; an untouched test seed
(seed 1) alongside the dev seed. Probe R^2 now picks the penalty and centering on half A only (values unchanged:
u2 0.57 / 0.82 / 0.87). Report: `outputs/q2_mnist_transfer/`.

| `u2`, final change (test seed, 95% CI over images) | applied 0.25 | applied 0.5 | applied 0.75 |
| --- | --- | --- | --- |
| learned at 0.25 | +0.38 (0.35, 0.40) | +0.17 (0.16, 0.18) | +0.10 (0.09, 0.10) |
| learned at 0.5 | +0.40 (0.38, 0.42) | +0.18 (0.17, 0.19) | +0.10 (0.10, 0.11) |
| learned at 0.75 | +0.35 (0.33, 0.37) | +0.16 (0.15, 0.17) | +0.10 (0.09, 0.10) |
| random, mean absolute change per image | 0.046 | 0.029 | 0.022 |
| RMS change of the guess while on (learned = applied) | 0.170 | 0.103 | 0.042 |
| thickness per unit of that change | 7.4 | 5.5 | 4.4 |

- **Every direction weakens late by about the same factor (about 4x):** rows are within about 10% of each other,
  so this is not a badly learned late direction. Directions are similar across t (cosine 0.72-0.92) and their
  control transfers with them; the one learned at 0.5 is marginally best everywhere.
- **Most of the decline is the size of the denoiser's response:** the same relative activation edit moves its
  guess about 4x less at 0.75 than at 0.25; the share of that movement that is thickness falls less (7.4 -> 4.4).
- **Per image:** 83-94% of digits change in the requested direction (u2), the effect is similar for all ten
  digits, and dev and test seeds agree within 0.01. The per-image slope of final on release change is 0.25-0.70
  with low R^2 (0.13-0.46), so individual digits vary much more than the averages; part of that is measurement
  noise in the blurry guess at release, so it does not show per-image attenuation either way.
- **d3 is weak late:** at t = 0.75 its effect (+0.02) is about the size of random edits and only 38-44% of
  digits move the requested way.
- The share of guess pixels at the +-1 clamp drops when editing (mostly the black background moving slightly off
  -1), so clipping is not absorbing the edit; this measure is too coarse to say more.

### Fixed-size edits and calibration (Nov 7, 2026): the late window is weaker but not worse

`mnist_calibrate.py`. **Part 1 (dev seed 0):** one direction (learned at 0.5) applied for 2 steps at each time,
either as the usual relative edit (a x |h_t|) or as a fixed absolute edit (eps x d, same eps at every t). At the
first edited call: the raw (unclipped) denoiser response, the clipped response, the sampler-state change at release,
and the final thickness. Report: `outputs/q2_mnist_calibrate/`.

| `u2` | t = 0.25 | t = 0.5 | t = 0.75 | late / early |
| --- | --- | --- | --- | --- |
| activation norm \|h\| (unedited) | 53.8 | 44.4 | 34.4 | 0.64 |
| final change per unit eps, fixed-size edit | 0.015 | 0.008 | 0.0065 | 0.43 |
| raw output change per unit eps (network gain along d) | 0.0064 | 0.0046 | 0.0027 | 0.42 |
| final change, relative edit a = 0.5 | +0.39 | +0.18 | +0.10 | 0.26 |
| clipped / raw output change | 1.00 | 0.99 | 0.97 | |

- **Two causes, multiplied:** relative edits shrink late because |h| shrinks (x0.64), and at a fixed size the
  network responds about 2.4x less along d (x0.42). 0.64 x 0.42 = 0.27 matches the observed 0.26. Final change
  per unit of raw output change does not fall late, so the sampler's conversion into thickness is not the cause.
- **Clipping absorbs at most 3%** of the output change (raw vs. clipped compared directly).
- **Locally linear and symmetric:** doubling eps doubles the change (0.08 / 0.17 / 0.34 at t = 0.25), and +eps
  / -eps are within about 15%.
- **d3:** |h| is flat over t, so relative and fixed edits are the same; its gain falls 5x (0.0067 -> 0.0013).
  The decline there is all sensitivity.

**Part 2: same requested change, compare the damage (u2).** For each window, the relative strength was tuned on dev
seed 0 to give a mean change of +-0.2 SD and then +-0.5 SD. Strengths were then frozen and evaluated on a fresh seed
(2). Seed 1 informed this design; seed 3 is untouched. Reports: `outputs/q2_mnist_calibrate/` (0.2) and
`outputs/q2_mnist_calibrate_t05/` (0.5).

| target +-0.5 SD, fresh seed (thicken / thin) | window 0.25 | window 0.5 | window 0.75 | every step (NA-RFM-style) |
| --- | --- | --- | --- | --- |
| strength a | 0.66 / 0.85 | 1.84 / 1.70 | 1.98 / 1.90 | 0.10 / 0.12 |
| activation change, total over edited calls | 69 / 89 | 160 / 148 | 130 / 125 | 77 / 92 |
| achieved change (target +-0.50) | +0.52 / -0.49 | +0.51 / -0.49 | +0.51 / -0.50 | +0.51 / -0.49 |
| digits moved the requested way | 96% / 97% | 96% / 98% | 97% / 99% | 97% / 98% |
| correct -> wrong (of 1000; wrong -> correct) | 8 (4) / 5 (2) | 7 (4) / 2 (1) | 3 (4) / 0 (1) | 4 (3) / 7 (4) |
| mean abs. slant change (SD of real slant) | 0.114 / 0.114 | 0.106 / 0.088 | 0.096 / 0.082 | 0.108 / 0.124 |
| mean centroid shift (px) | 0.25 / 0.29 | 0.25 / 0.26 | 0.22 / 0.26 | 0.24 / 0.29 |

Random direction (window 0.5, a = 0.5): change 0.00, slant 0.013, shift 0.04 px, 0 broken.

- **Hitting a requested mean change is easy:** a strength tuned on one seed lands within 0.02 SD of the target
  on the fresh seed, for every window and both targets. Calibration needs a short search, not a single slope:
  the response bends at larger strengths (window 0.5 needs a = 0.57 for 0.2 but 1.84 for 0.5).
- **Weaker is not worse:** the late window needs about 2x the activation change of the early one, but it is
  slightly more selective. It has the least slant drift (0.08-0.10 vs. 0.11 early), the fewest broken digits
  (3 vs. 13 over both signs) and a narrower per-image spread. At +-0.2 SD the same ordering holds (slant 0.040-0.045
  late vs. 0.057-0.062 early), with 0-3 broken digits everywhere.
- **Holding the edit on every step is not better** than one tuned 2-step window. It sits with the early window.
- **The differences are small.** Identity damage stays under 1% everywhere, and slant drift differs by about
  25%. Some slant and position change may simply come with a thicker stroke. On this task, a fixed window with a
  tuned strength already controls the attribute; a calibration method would have little to improve. This is the
  stopping result the review anticipated for MNIST thickness, unless a harder attribute separates the methods.

## Reference-set bias test (Nov 8, 2026): a confounded reference set makes a confounded edit

The timing work above closed with "a tuned fixed window is enough". The next question is what makes a learned
direction reliable. `mnist_bias.py` learns a thickness direction from four reference sets of real digits:
- 60 thick + 60 thin per digit in each set;
- natural, balanced on slant, aligned (thick digits lean one way) and opposed (they lean the other way);
- three learners: mean difference, ridge probe, and RFM-style (top eigenvector of the AGOP on the top 256 PCs).

Each direction is tuned on dev seed 0 to change thickness by +-0.3 SD and evaluated on seed 2. The outcome is
the slant change at matched thickness. Report: `outputs/q2_mnist_bias/`.

| signed slant change (SD of real slant), thicken / thin | mean difference | ridge | RFM |
| --- | --- | --- | --- |
| natural (corr 0.04 in MNIST) | +0.02 / -0.03 | -0.00 / -0.00 | -0.02 / +0.01 |
| balanced | +0.01 / -0.02 | +0.00 / -0.01 | -0.00 / -0.01 |
| aligned (corr +0.65 in the set) | **+0.33 / -0.16** | +0.20 / -0.13 | +0.21 / -0.12 |
| opposed (corr -0.63) | **-0.29 / +0.16** | -0.18 / +0.14 | -0.27 / +0.13 |

Thickness change is +-0.30 in every row. Random-direction slant floor: 0.013.

- **The bias in the reference set reaches the generated digits.** It shows in all ten digits (+0.13 to +0.52
  when thickening with the aligned set), and its sign follows the bias. The confounded directions also need
  2-4x more strength for the same thickness and break more digits (up to 20 of 1000 vs. 5).
- **No learner removes it.** Ridge and RFM reduce the leak by different amounts depending on the learner and
  the bias sign: from about 7% (RFM, opposed, thickening) to about 40% (ridge, aligned, thickening). A better classifier of the
  labeled groups does not give a cleaner edit: ridge accuracy is higher on the biased sets (0.95 vs. 0.87),
  because slant makes the groups easier to tell apart.
- **Thickening leaks about twice as much slant as thinning.**
- **A slant-balanced reference set reduces measured leakage to about the balanced-reference baseline**, but
  only if you know slant is the nuisance and can label it. MNIST's own thickness-slant
  correlation is tiny (0.04), so natural sampling already behaves like balanced. The bias here is deliberately
  strong (slant gap 1.6 SD between groups). Whether this is a research problem depends on the next test:
  milder bias, and fixes under a limited budget of nuisance labels.

### Bias under a limited slant-label budget (Nov 8, 2026, DSMLP GPU): about 100 labels fix moderate bias

`mnist_bias_budget.py` (pod script `dsmlp_bias_budget.sh`, 72 minutes on one GPU).
- **Reference sets:** seven sets of the same size and digit mix, with actual within-digit thickness-slant
  correlations of 0.02, +0.21, -0.17, +0.37, -0.41, +0.67 and -0.66.
- **Corrections**, each given 20 or 100 slant labels (5 independent draws) or all 1,200 (upper bound):
  - stratify: predict slant, then compare thick vs. thin within slant quarters;
  - subtract: project a slant direction out;
  - controlled: ridge on thickness with slant regressed out.
- **Matching:** each direction is re-tuned to +-0.3 SD of thickness, per sign, on dev seed 0, then evaluated
  on seed 2.

Report: `outputs/q2_mnist_bias_budget/report.md`.

| slant change when thickening (mean over draws) | none | stratify 20 / 100 / all | subtract 20 / 100 / all | controlled 20 / 100 / all |
| --- | --- | --- | --- | --- |
| corr 0.02 (baseline) | +0.004 | +0.009 / -0.000 / +0.002 | +0.004 / +0.003 / +0.003 | -0.06* / +0.010 / -0.001 |
| corr +0.21 | +0.067 | +0.024 / -0.006 / +0.001 | +0.055 / +0.037 / +0.035 | +0.04* / -0.017 / -0.016 |
| corr +0.37 | +0.148 | +0.062 / +0.016 / +0.001 | +0.128 / +0.094 / +0.066 | +0.02* / +0.007 / +0.006 |
| corr -0.41 | -0.135 | -0.050 / -0.012 / -0.009 | -0.122 / -0.101 / -0.083 | +0.01* / -0.013 / -0.034 |
| corr +0.67 | +0.336 | +0.266 / +0.092 / (no overlap) | +0.322 / +0.206 / +0.181 | +0.02* / +0.034 / +0.014 |
| corr -0.66 | -0.334 | -0.228 / -0.071 / (no overlap) | -0.319 / -0.295 / -0.191 | -0.01* / -0.055 / -0.033 |

\* = at least one draw could not reach the thickness target (see the report). Thinning shows the same pattern at
about half the size.

- **Leakage grows about linearly with the bias:** 0.3-0.5 SD of slant per unit of correlation when
  thickening. Identity damage rises only at the strongest bias (24-28 broken digits vs. 9).
- **100 slant labels (8% of the reference set) restore about the baseline for mild and moderate bias.**
  Stratifying on predicted slant brings |leak| to 0.016 or less at |corr| <= 0.4, consistently across draws and
  both signs.
- **Strong bias needs the controlled ridge.** At |corr| 0.66 the set has no thick low-slant digits, so stratifying
  with true labels has nothing to compare. With 100 labels it removes about 75% of the leak, and only because
  prediction noise creates some overlap. The controlled ridge with 100 labels gets to 0.03-0.06, but draws vary
  (up to 0.12), it needs about 2x the strength, and it breaks slightly more digits (13-14 vs. 9).
- **Subtracting a slant direction is weak.** It removes 20-55% of the leak even with every label. This does not
  show that slant occupies several directions: the slant direction may be inaccurate, a full projection may be the
  wrong amount to remove, or the direction that predicts slant may not be the one that changes it in the output.
- **20 labels are not enough for any correction.** Stratify roughly halves the leak; the controlled ridge often
  cannot reach the thickness target and breaks 30-40 digits.
- **Decision (by the review's table):** for a known nuisance on MNIST, small label budgets of about 100 fix mild
  and moderate bias with ordinary stratification. Only strong bias with very few labels stays unsolved, which
  is a stress-test result rather than a broad practical gap. Stop method development for known-nuisance
  correction on MNIST. The open problem is discovery: finding which unintended attribute contaminated a
  reference set, without labels for it.

## Files and how to run

Two folders. `mnist/` is the current work; `sd_pilots/` is the finished text-to-image study (archived: kept so the
negative result can be reproduced). Results live in `outputs/` (not in git). The SD images and latents were removed
locally to save space; full copies are on DSMLP under
`/dsmlp/workspaces-fs05/DSC180A_FA26_001/home/yuz271/dsc180-diffusion-steering/outputs/`. Reports, `scores.csv` and
the `.jsonl` records are still local, so `analyze.py` and `scorer_check.py` run without them; `early_check.py`,
`scorer_check.py --aug`, `make_audit.py build` and the notebook's example figures need the images or latents back.

**`mnist/` (current)**

| File | What it does |
| --- | --- |
| `mnist_thickness.py` | Thickness readability vs. effect of activation edits (learned direction, its negative, random), one-step vs. held edits |
| `mnist_transfer.py` | Direction-transfer matrix: learned-at x applied-at, per-image results, test seed |
| `mnist_calibrate.py` | Fixed-size vs. relative edits (network gain, clipping) and windows calibrated to the same thickness change on a fresh seed |
| `mnist_bias.py` | Reference-set bias: thickness directions learned from slant-balanced vs. slant-confounded sets (mean difference, ridge, RFM), slant change at matched thickness |
| `mnist_bias_budget.py`, `dsmlp_bias_budget.sh` | Bias levels x slant-label budgets x simple corrections (stratify, subtract, controlled ridge), matched thickness; DSMLP pod script |
| `mnist_release.py` | Release-and-follow: edit on for a fixed time window, then off; follows both trajectories at 16 and 64 steps |

```bash
python quarter2/mnist/mnist_thickness.py --out outputs/q2_mnist_thickness/steps16 --steps 16 --n 1000
python quarter2/mnist/mnist_release.py --out outputs/q2_mnist_release --n 1000
python quarter2/mnist/mnist_transfer.py --out outputs/q2_mnist_transfer --n 1000
python quarter2/mnist/mnist_calibrate.py --out outputs/q2_mnist_calibrate --target 0.2   # and --target 0.5
python quarter2/mnist/mnist_bias.py --out outputs/q2_mnist_bias
```

All use `generator.pt`, `evaluator.pt` and `mnist_guidance_lab.py` from the repository root and run on a laptop
(Apple MPS or CPU) in minutes.

**`sd_pilots/` (archived)**

| File | What it does |
| --- | --- |
| `sd_branch.py` | SD 1.5 baselines (saving latents after steps 6, 15, 24) and branched continuations |
| `evaluate.py` | Automated checklist scores: OWLv2 boxes (objects, left–right, counts), CLIP ViT-L/14 colors |
| `analyze.py` | Pass/fail checks, success mix, strength pick, the repair/damage/oracle report, and real vs. placebo/wrong-sign/re-seed menus |
| `prompts.json`, `dev_prompts.json`, `dsmlp_smoke.sh` | Smoke test (color + left/right prompts) |
| `two_object_*.json`, `counting_*.json`, `dsmlp_pilot2.sh` | Pilot 2: two-object and counting prompts (24 main, 6 dev each) |
| `dsmlp_pilot3.sh`, `counting_countable.json` | Pilot 3: placebo nudges and seeds 8–15; counting without books and balloons |
| `dsmlp_pilot4.sh` | Wrong-sign control for two objects (written, deliberately not run) |
| `scorer_check.py` | How much of the gap survives stricter scoring (margin from the cutoff; `--aug` re-scores flipped/cropped images) |
| `early_check.py` | Scores the model's predicted final image at steps 6/15/24: does it predict failures, repairs, a gated policy? |
| `make_audit.py` | Blinded hand audit: builds `audit.html` (shuffled images, id and question only) and scores the labels |
| `nudges_explained.ipynb` | The nudges as formulas, how strengths and policies are chosen, examples from pilot 2 |
| `requirements.txt` | Python packages for the GPU pod |

Each pilot runs from the repository root in a GPU pod, e.g. as a 12-hour batch pod from dsmlp-login:

```bash
K8S_TIMEOUT_SECONDS=43200 launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 8 -m 32 -B -N q2-smoke -- \
    bash -c "cd ~/dsc180-diffusion-steering && bash quarter2/sd_pilots/dsmlp_smoke.sh"
```

Smoke-test settings: DDIM (eta 0), 30 steps, CFG 7.5, float32, 512×512, seeds 0–7 (dev
100–103). Pulses last 3 steps. Dev strengths tried: global CFG 10, 14, 20;
targeted 5, 15, 40; plus `none:0`, a resumed run with no change, which checks
that resuming is exact. The main run uses, for each action, the dev strength
with the highest mean change in all-requirements success (ties keep the weaker
one). That rule was fixed before the run. The automated scores still need a
blinded manual audit before any result is trusted.

## Question

Detecting that a text-to-image sample is going wrong and then adjusting the
diffusion process is already established (see [Related work](#related-work)).
The narrower question here:

> **Can we predict whether a particular intervention will improve the final
> image, rather than only whether the image will be wrong?**

A sample can be wrong but no longer cheaply repairable; another can look
ambiguous mid-generation yet resolve correctly on its own. The smoke test checks
whether that distinction matters, in three steps, in order:

1. **Headroom:** can any tested action repair failures without comparable damage?
2. **Sample dependence:** does the best action depend on the sample, or does one
   fixed timing rule do just as well?
3. **Predictability:** can information available during generation predict the
   benefit?

If the answer to 1 or 2 is no, training a controller is unlikely to help, and we
stop cheaply.

## Setup

**Generator:** Stable Diffusion 1.5, frozen, standard CFG, one fixed
deterministic sampler, 30 steps. Nothing is trained for the smoke test. A DiT
comes later, only if the smoke test succeeds and we want architecture-general
claims.

| Component | Role | Trained? |
| --- | --- | --- |
| CLIP text encoder | prompt → token embeddings | No |
| Conditional U-Net (with cross-attention) | predicts noise from latent z_t, timestep t, text | No |
| Sampler | updates the latent from the noise prediction | No parameters |
| VAE decoder | final latent → image | No |
| Evaluator (VLM/detector + manual audit) | scores the finished image; not part of the generator | No |

Link to Quarter 1: the MNIST model used a digit embedding in pixel space; SD 1.5
uses text-token embeddings in a compressed latent space. Both are conditional
U-Nets that combine conditional and unconditional predictions through CFG.

**Prompts:** [prompts.json](prompts.json) has 24 prompts. Each tests one failure
type, the left–right relation between two colored objects, e.g. *"A red mug to the
left of a blue bowl."* There are 6 object/color pairs, and each appears in 4
prompts (left/right × which object is named first). That way neither the model
nor the evaluator can score well by always putting the first-named object on the
left. Each prompt carries a checklist:

```text
object_1: mug     color_1: red
object_2: bowl    color_2: blue
relation: object_1 left of object_2
```

There is no ground-truth image; the checklist is the evaluation criterion.
24 prompts × 8 seeds = **192 baseline trajectories**, a screening set only.

**Check first:** the baselines must contain both successes and failures. If
nearly all succeed, or nearly all miss an object, change the difficulty
before branching.

## Smoke test

### 1. Branch from identical intermediate states
Run each baseline for 30 steps and save the latent after steps **6, 15 and 24**
(step 1 = start). From each saved state, independently try:

| Action | Intervention |
| --- | --- |
| No action | continue the original trajectory (reuse the baseline image) |
| Global guidance pulse | raise CFG for 3 steps |
| Targeted semantic pulse | for 3 steps, add a scaled difference between predictions for the prompt and its counterfactual ("…left of…" vs. "…right of…") |

The targeted pulse is a **diagnostic intervention**, not the proposed method;
it can affect other properties, which is why collateral damage is measured.
Choose strengths on a separate small development batch, then freeze them. Match
update norms where practical and count extra forward passes. Every branch starts
from the original saved trajectory, never from another branch.

192 × 3 checkpoints × 2 actions = **1,152 partial continuations**. Time 10
trajectories before estimating GPU hours.

### 2. Measure repair and damage
Score four requirements per image: both objects present, color 1, color 2,
relation. Primary outcome: **all requirements satisfied**; report relation
accuracy and collateral damage separately (CLIP similarity alone can't answer
this).

| Transition vs. no action | Meaning |
| --- | --- |
| wrong relation → correct, objects/colors kept | repair |
| correct → wrong | harm |
| relation fixed, but an object or color lost | repair with collateral damage |
| no material change | ineffective |

Use one consistent automated evaluator, then manually audit a blinded subset
(including evaluator disagreements). Final evaluation must be independent of any
evaluator later used for control.

The label to predict eventually:

$$U(t,a) = R_{\text{all}}\big(X^{a}_{\text{final}}\big) - R_{\text{all}}\big(X^{\text{no action}}_{\text{final}}\big)$$

which separates helpful, harmful and ineffective actions; the cost of the action
enters the selection objective separately.

### 3. Check the oracle before training anything
From the finished branches, pick in hindsight the best action and checkpoint per
sample (including doing nothing). This is an **upper bound over our action menu**,
not a method. Compare it with:
- no intervention
- the best fixed action + checkpoint (chosen on development data)
- tuned constant CFG and limited-interval CFG

**Key number: how much does per-sample hindsight selection beat the best fixed
policy?** If substantial, check whether the gain comes from choosing *when*,
*which action*, or just *whether* to intervene.

### 4. Only then: is the benefit predictable?
Only if the oracle shows headroom, collect more data and train a small predictor
(linear model or small MLP):

$$g_\phi(\text{state features}, \text{prompt features}, t, \text{action}) \rightarrow \text{predicted gain in final success}$$

Inputs available at the checkpoint: timestep, prompt attributes, features of the
predicted clean image, optionally a few layers' activations. For left–right
relations, keep coarse spatial structure (global average pooling would erase it).

| Predictor | Predicts |
| --- | --- |
| Time-only rule | average usefulness of an action at that stage |
| Failure predictor | whether the unchanged image will violate the prompt |
| **Benefit predictor** | whether this action improves final success |

**The benefit predictor must beat the failure predictor**, or the central
distinction has no practical value. Keep all seeds, checkpoints and branches of
a prompt in the same split; hold out object combinations and paraphrases.
Allow only one intervention per trajectory at first.

One collected record:

```text
prompt, seed, checkpoint
intermediate latent / selected features
action, action strength
baseline final requirement scores
intervention final requirement scores
extra compute (forward passes)
```

### 5. Go/no-go criteria (decided in advance)

| Finding | Decision |
| --- | --- |
| Neither action gives reliable repairs | Drop this action menu; try one stronger established intervention before abandoning the question |
| Hindsight beats best fixed policy by < ~5 points | Weak case for adaptive control |
| Hindsight beats it by ~10+ points with acceptable quality | Strong reason to test predictability |
| Benefit predictor beats fixed timing but not failure prediction | Useful engineering, weak evidence for the claim |
| Benefit predictor captures much of the oracle gain on unseen prompts and beats failure-based control | Basis for a larger study |

With 192 trajectories the uncertainty is wide. Estimate it by resampling
prompts (not treating branches as independent), and confirm any positive result
on a larger held-out set.

**First deliverable:** one table with repair rate, damage rate, best
fixed-policy performance, and per-sample oracle headroom.

## Related work

| Paper | What it does | What it means for us |
| --- | --- | --- |
| Dynamic CFG via Online Feedback ([arXiv 2509.16131](https://arxiv.org/abs/2509.16131)) | Noise-conditioned latent evaluators pick guidance strength greedily; evaluator weights change over time | Early evaluation and per-sample schedules are covered; "apply feedback when it becomes informative" is not new |
| Concept Reachability ([arXiv 2505.19313](https://arxiv.org/abs/2505.19313)) | Optimizes steering vectors in prompt embeddings or U-Net bottleneck, generator frozen, fit with target images | "Steering reaches concepts prompting misses" is covered |
| Reflect-DiT | VLM critic plus a DiT that reads earlier images and feedback | Feedback-driven regeneration is covered |
| AnchorSteer ([arXiv 2607.26647](https://arxiv.org/abs/2607.26647)) | VLM diagnosis and targeted latent correction during generation | Generic mid-generation self-correction is covered; **full text not yet read; check overlap before claiming novelty** |

The possible gap, predicting the *incremental benefit of a bounded action at
the current state*, is a hypothesis to test, not a confirmed gap.

## What would make it publishable

1. **A repeatable finding:** detectability, failure probability and
   repairability differ in ways that matter.
2. **A useful method:** predicting benefit beats failure-triggered and
   score-based control.
3. **Strong evidence:** gains hold across error types, unseen prompts and a
   second generator, with quality and total compute accounted for.

Baselines must include feedback-based scheduling and correction/search methods,
not only constant CFG. If we use extra compute, best-of-N sampling is a
required competitor. DiTs and reinforcement learning are later extensions, not
needed for the first question: *do useful, sample-dependent intervention
opportunities exist?*

## Next steps
- [ ] Read AnchorSteer's full text and check its overlap with this question.
- [ ] Load SD 1.5 and a deterministic sampler, run the 192 baselines, and check
      the mix of successes and failures.
- [ ] Pick the automated evaluator; label a blinded subset by hand to check it.
- [ ] Tune pulse strengths on a development batch; time 10 branched trajectories.
- [ ] Run the branches, then build the repair/damage/oracle table.
