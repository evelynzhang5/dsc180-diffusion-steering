# Q2 round 2: MNIST steering diagnostics

Branch `q2-mnist-steering`. Nov 7-8, 2026. Question: when an attribute (stroke thickness) is readable from the
generator's activations, what decides whether an edit changes it, whether the change lasts, and what else the edit
changes? The index of all Q2 rounds is in [../README.md](../README.md).

**Results in git.** The scripts write to `outputs/` (not in git). Copies of the small result files (reports, figures, logs, small tables; files over 1 MB left out, large PNGs converted to JPEG) are in `results/`, under the same folder names: `outputs/q2_mnist_thickness/` -> `results/q2_mnist_thickness/`, `outputs/q2_mnist_release/` -> `results/q2_mnist_release/`, `outputs/q2_mnist_transfer/` -> `results/q2_mnist_transfer/`, `outputs/q2_mnist_calibrate/` -> `results/q2_mnist_calibrate/`, `outputs/q2_mnist_calibrate_t05/` -> `results/q2_mnist_calibrate_t05/`, `outputs/q2_mnist_bias/` -> `results/q2_mnist_bias/`, `outputs/q2_mnist_bias_budget/` -> `results/q2_mnist_bias_budget/`, `outputs/q2_mnist_audit/` -> `results/q2_mnist_audit/`.
Left out for size: `q2_mnist_transfer/per_image.csv`, `q2_mnist_bias/per_image.csv`, `q2_mnist_calibrate*/fixed.csv`,
`q2_mnist_bias_budget/runs.jsonl` (the scripts regenerate them).

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

## Side-effect audit (Nov 8, 2026, DSMLP GPU): output evidence beats reference differences, but MNIST is too easy

`mnist_audit.py` (pod script `dsmlp_audit.sh`) is a toy version of "predict unintended steering effects on unseen
prompts". Digit classes play the role of prompts, and a bank of 7 measured attributes plays the role of a VLM:
slant, width, height, horizontal / vertical position, stroke length and darkness.

**Setup.**
- **Reference sets:** each one secretly confounds thickness with one attribute, at strong (q = 1) or moderate
  (q = 0.75) strength, plus one natural set.
- **Edit:** the thickness edit is tuned to +0.3 SD on 5 digit classes.
- **Audits:** run on those 5 classes and scored against the true side effects on the other 5 (both splits; 210
  items). All effects are per 1 SD of thickness.
- **REF:** reference thick-minus-thin difference.
- **OUT-1:** one edited-vs.-unedited run.
- **OUT-rep:** two runs that must both be significant with the same sign.
- **BOTH:** REF and OUT-rep agree.

Report: `outputs/q2_mnist_audit/report.md`.

| tau = 0.15 (89 true side effects of 210) | flagged | false alarms | recall | rank corr. with truth |
| --- | --- | --- | --- | --- |
| REF (reference differences) | 110 | 33 | 0.87 | 0.63 |
| OUT-1, 20 digits per class | 129 | 42 | 0.98 | 0.79 |
| OUT-1, 100 per class | 105 | 22 | 0.93 | 0.82 |
| OUT-rep, 20 per class | 62 | 0 | 0.70 | 0.72 |
| OUT-rep, 100 per class | 75 | 3 | 0.81 | 0.75 |
| BOTH, 100 per class | 67 | 1 | 0.74 | 0.75 |

- **Reference differences mispredict in both directions.**
  - **Bias that doesn't reach the output:** a position bias (centroid +0.86 SD in the references) does not reach
    the output (-0.17 and -0.05). Height is overpredicted 2x, and the natural set's stroke length has the wrong
    sign (-0.25 predicted, +0.09 true).
  - **Effects the references don't show:** with the slant set, the edit adds more slant than the references show
    (+1.27 vs. +0.84) and widens digits (+0.63).
- **Paired output runs are the better evidence.** Repeating the run and requiring significance in both cuts false
  alarms from 22-42 to 0-3. A single 100-per-class run already has the best rank correlation. With 20 per class,
  repetition is what keeps false alarms out.
- **Side effects transfer across digit classes almost perfectly:** 98% same sign, median error 0.10. On MNIST, a
  side effect seen on 5 classes predicts the other 5.
- **Some side effects come with the target itself.** Darkness rises about 0.9 SD per SD of thickness in every set,
  and width about 0.3. These may be partly how the meters work (thick strokes have more full-intensity pixels),
  not leakage.
- **Verdict.** The gate passes, since intervention evidence clearly beats reference differences. But the result
  is unsurprising, because the output audit measures the effect directly. MNIST also leaves out the two hard parts
  of Proposal 1:
  - side effects that are not in a fixed list (the VLM's job);
  - contexts where effects do not transfer, which are 98% consistent here.

  Both need natural images, so the next step is the Stable Diffusion check that reference bias propagates
  (blur vs. portrait framing), scored with programmatic meters.

## Files and how to run

| File | What it does |
| --- | --- |
| `mnist_thickness.py` | Thickness readability vs. effect of activation edits (learned direction, its negative, random), one-step vs. held edits |
| `mnist_transfer.py` | Direction-transfer matrix: learned-at x applied-at, per-image results, test seed |
| `mnist_calibrate.py` | Fixed-size vs. relative edits (network gain, clipping) and windows calibrated to the same thickness change on a fresh seed |
| `mnist_bias.py` | Reference-set bias: thickness directions learned from slant-balanced vs. slant-confounded sets (mean difference, ridge, RFM), slant change at matched thickness |
| `mnist_bias_budget.py`, `dsmlp_bias_budget.sh` | Bias levels x slant-label budgets x simple corrections (stratify, subtract, controlled ridge), matched thickness; DSMLP pod script |
| `mnist_audit.py`, `dsmlp_audit.sh` | Side-effect audit: reference differences vs. paired edited outputs as predictors of side effects on held-out digit classes; DSMLP pod script |
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
