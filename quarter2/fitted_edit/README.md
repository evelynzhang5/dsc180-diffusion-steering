# Round 16: can a small spatial activation edit make the right depth-of-field edit at all?

**Status:** closed (Nov 16). By the frozen rule: NO-GO, inconclusive, because the optimizer failed its control. The run stopped after 3 of 8 scenes (DSMLP's 6-hour pod limit), and those 3 already fix the decision. See [Results](#results).

This is a privileged feasibility test with a stopping rule. Before training any reusable controller, we give the proposed mechanism the easiest version of the task. Each edit is fitted separately for one image, with the target image visible to the optimizer.

- **If it passes:** this justifies the reusable-controller experiment. It is not a publication claim: it shows nothing about generalization, inference cost, or an advantage over a paired-image LoRA or Bokeh Diffusion.
- **If it fails:** first the optimizer and implementation controls are checked. A failure does not show that all activation steering is impossible.

Code: [fitted.py](fitted.py) (stages `gen` and `fit`), pod script [dsmlp_fitted.sh](dsmlp_fitted.sh).

## Setup (fixed before any fitting)

### Scenes

There are 8 scenes, with seed 42 for all except the off-centre scene (see below):

| # | type | caption | mask noun |
|---|---|---|---|
| 0 | person | A woman holding a red umbrella on a city street | woman |
| 1 | animal | A dog sitting in front of a wooden fence in a garden | dog |
| 2 | animal, cluttered background | A deer standing in a forest, the whole forest in sharp focus | deer |
| 3 | small subject | A small bird standing far away on a long empty beach | small bird (threshold 0.3, as in round 13) |
| 4 | off-centre | A horse standing at the far left edge of a wide green pasture, rolling hills behind it | horse |
| 5 | object | A red ceramic coffee mug on a wooden table in a busy cafe | coffee mug |
| 6 | hair boundary | A woman with long curly hair standing in a field of tall grass | woman |
| 7 | thin structures | A man riding a bicycle on a country road | man riding a bicycle |

The off-centre scene uses the first of seeds 42-46 whose subject centre lies at least 20% of the image width from the centre.

Settings held fixed for every method:
- SD 1.5, fp32 on the Pascal GPU
- text CFG 7.5
- 50-step DDIM sampler
- caption and initial noise

### Masks

Masks come from CLIPSeg on the unedited image only. I check them by eye on `masks_check.png` after `gen`. Any correction goes into `MASK_FIX` in the code and is recorded here. After that the masks are never recomputed.

Regions are as in round 12:
- **interior:** more than 6 px inside the subject
- **background:** more than 12 px outside the subject
- **band:** everything between them

### Targets

Each target is the unedited image with its background Gaussian-blurred and the subject pasted back with a feathered alpha (round 12's post-hoc render). The radius is found by bisection so that the background log-sharpness change is exactly −0.5 or −2.0. These are diagnostic endpoints, not evidence that steering beats rendering.

### Interventions

| method | what it is |
|---|---|
| no edit / rendered target | endpoint checks for the measures |
| tuned masked RFM | Round 13's setup: rfm_eig1 at down_blocks[2] in a third CFG branch at s 4, masked to the background. The strength is tuned per scene and target (secant search, at most 4 generations) in each of round 13's windows (full, early, middle, late). The baseline is the window that reaches the target with the lowest subject-interior error; if none reaches it, the one closest to the target. |
| fitted global edit | 8 basis coefficients per time segment (16 numbers), applied at every position and multiplied by the background mask (the fraction of non-subject pixels in each 16 × 16 px cell) |
| fitted spatial edit | 8 basis coefficients per cell of a 16 × 16 grid per time segment (4,096 numbers), bilinearly upsampled to 32 × 32; no mask, so it is free to vary across the image |

Shared settings for the two fitted edits:

- **Layer.** `up_blocks[2].resnets[-1]`, a higher-resolution decoder layer (640 × 32 × 32). Round 4 used an encoder-side 1280 × 16 × 16 block.
- **Edit.** h(x) ← h(x) + |h(x)| Σ_r a_r(x, t) b_r on both CFG rows, where |h(x)| is the channel norm at position x.
- **Basis.** b_1..b_8 are the top 8 principal directions of the layer's feature change between each unedited image and its targets. The images are VAE-encoded and forward-noised with the same noise at steps 20, 28, 36 and 44, and the directions are pooled over the 8 scenes and both targets. This is a shared basis, but privileged: it is computed from these scenes' targets.
- **Time.** Steps 17-49 (σ 2.65 → 0.04), starting from the cached unedited latent at step 17. Round 13 found the layout is mostly set by then. There are two segments, 17-32 and 33-49, each with its own coefficients.
- **Loss** (pixels 0-255), computed on the actual decoded final image with exact gradients through every sampling step (re-run step by step) and the VAE decoder. No linear approximation from round 12 is used.

  MSE_bg(x, T) / MSE_bg(ref, T) + ((Δlog-sharpness_bg(x) − target) / 0.15)² + MSE_subject(x, T) / 3² + MSE_band(x, T) / max(MSE_band(ref, T), 3²)

- **Budget.** Adam with the same iteration count and learning rate for every fit, starting from zero. The lowest-loss iterate is kept. The count and rate are set from the probe's speed and edit-scale measurements before any fit is run, and recorded below.

### Controls

- **Determinism:**
  - regenerated unedited images must match the saved ones exactly;
  - the zero edit must equal no edit;
  - the cached run from step 17 must equal the full run;
  - round 13's sampler must reproduce the same image.
- **Gradient check:** autograd against a central finite difference, on both edit types (probe).
- **Optimization control:** on scenes 0 and 3, generate an image from a known random spatial edit, then fit the spatial edit to it from zero with the same budget. The control passes if the remaining MSE is at most 10% of the zero-edit MSE. If it fails, a NO-GO is inconclusive (an optimizer problem, not an architecture result).

## Frozen criteria

These are engineering thresholds, not established standards. They are evaluated per scene and per target.

| | requirement | passing condition |
|---|---|---|
| R | blur happens | background log-sharpness change within ±0.15 of the target |
| S | subject intact | mean absolute pixel change in the subject interior ≤ 3/255 |
| A | background like the target | background MAE to the target ≤ 0.5 × the unedited image's background MAE to the target (it removes at least half the gap rather than only lowering a sharpness score) |
| V | spatial adds value | spatial interior MAE ≤ 0.5 × that of the best feasible baseline (tuned masked RFM or fitted global, among those meeting R); met automatically if no baseline meets R |
| B | boundary usable | spatial band MAE vs the target ≤ that of the best matched baseline (the feasible baseline with the lowest subject error, or else the baseline closest to the target); doubled outlines are inspected on `boundary_crops_T2.png` |

**Decision:**

- **GO:** the spatial edit meets R, S, A, V and B on at least 7 of 8 scenes at both −0.5 and −2.0. Next step: test whether a shared controller can learn these edits and work on unseen scenes.
- **SIMPLER:** the fitted global edit meets R, S and A on at least 7 of 8 at both levels, but the spatial edit does not GO. Drop the spatial-controller claim and study the simpler mechanism.
- **NOT GO (no margin):** the spatial edit meets R, S and A on at least 7 of 8 at both levels, but not V or B.
- **NO-GO for this architecture:** otherwise. Failure modes are reported per scene: blur only by changing the subject; subject kept only by missing the blur; misses both; background unlike the target.

## Budget (from the probe, before fitting)

The first `gen` run (Nov 15, scene 0 with the original mask) measured:

| measurement | result |
|---|---|
| one fitting iteration (33 steps forward, re-run with gradient, VAE decoder) | 52 s, peak 8.2 GB on a GTX 1080 Ti |
| gradient check (autograd vs central difference) | global: relative error 3e-5; spatial: 3% |
| zero edit vs no edit; run cached from step 17 vs full run | identical (max difference 0) |
| regenerating an unedited image vs its saved copy | max difference 1 level of 255 (GPU nondeterminism); the check allows 1 |
| background change of a global edit along b_1 at a = -0.1, -0.2, -0.4 | -0.19, -0.42, -0.80 (subject interior MAE 0.44, 0.73, 1.46) |
| same along b_2 | under 0.08 in size |
| basis variance captured by b_1..b_8 | 49% (b_1 19%) |

Frozen from these:
- **Learning rate 0.08, 16 iterations, for every fit.** The −2.0 target needs coefficients of about 1, and Adam can move each coefficient up to 1.28 in 16 steps.
- **Optimization control:** the hidden edit's coefficients have spread 0.1.
- **Estimated total:** 34 fits × 16 × 52 s ≈ 7.9 h, plus about 1 h for the RFM baseline.

**Mask check (by eye, after the first `gen`):**
- Scenes 1, 2, 4, 6 and 7 are correct. Scene 4 got seed 44: the horse's centre is 25% of the width off centre.
- Scene 0's "woman" mask left out the red umbrella she holds. It is fixed as the union of the CLIPSeg masks for "woman" and "red umbrella" (`MASK_FIX`), and `gen` is re-run.
- Scene 3's "bird" is a small figure on a thin stalk. The mask (0.11% of the image) covers the figure, not the stalk; it is kept, as in round 13.
- Scene 5's mask covers the mug, its handle and most of the saucer; it is kept.

**Revision after the first optimization control failed (Nov 15, 22:20, before any scene had been fitted).** With step 0.08 and 16 iterations, the spatial fit could not recover a known random spatial edit:

| control scene | remaining MSE (pass ≤ 0.1) | loss by iteration |
|---|---|---|
| 0 | 0.84 | 1.0 → 7.3 → 3.7 → … → 0.84 |
| 3 | 1.00 | 1.0 → 19.0 → 5.6 → … → 1.64 |

The first Adam step moves all 4,096 coefficients by 0.08 at once and overshoots; the loss then falls only slowly. This is an optimizer setting, not a property of the edit, so the run was stopped before any scene was fitted (`control_v1_lr0.08.json` is kept). Changes, decided on the control alone:

- **Spatial parameterization.** The spatial edit is now fitted as one shared coefficient set (2 × 8, step 0.08, the same as the global edit) plus a per-cell residual (2 × 8 × 16 × 16). The set of edits it can express is unchanged; only the first steps are better conditioned.
- **Residual step size.** Chosen from {0.01, 0.02, 0.04} on control scene 0 only, by lowest remaining MSE. It must then also pass on control scene 3.
- **Iterations.** 24 for every fit, global and spatial.
- **Estimated time.** About 13.5 h.

## Results

Files: [results/fit_checks.md](results/fit_checks.md) (full tables and checks), [fit.json](results/fit.json), [fit_summary.json](results/fit_summary.json), [control.json](results/control.json), sheets [examples_T2.png](results/examples_T2.png), [examples_T0.5.png](results/examples_T0.5.png), [boundary_crops_T2.png](results/boundary_crops_T2.png) (columns: target, RFM, global, spatial).

### What ran

- **Optimization control (v2, after the revision above).** On scene 0 the residual-step grid gave remaining MSE 0.480, 0.487 and 0.574 for steps 0.01, 0.02 and 0.04. Step 0.01 was chosen; scene 3 then gave 0.632. The pass bar is 0.1, so the control **failed**. The first Adam step still overshoots: loss rises from 1 to 5.8 on scene 0 and to 12.6 on scene 3.
- **Scenes.** Each scene took about 90 minutes: 4 fits × 24 iterations × 52 s, plus the RFM search. My estimate had been 22 minutes. The pod was stopped at DSMLP's 6-hour limit after scenes 0-2 (person, dog, deer in a cluttered forest). Scenes 3-7 were not run. With 3 of 8 scenes failing at −2.0 for every method, GO, SIMPLER and NOT GO all need at least 7 of 8, so none is reachable. The remaining 5.5 GPU-hours could not change the decision.
- **Determinism checks** all passed: regenerated images, round 13's sampler, both endpoints, and finite measures.

### Numbers (scenes 0-2)

| | −0.5: reached | interior MAE | bg MAE vs target ÷ no edit's | −2.0: reached | interior MAE | bg MAE vs target ÷ no edit's |
|---|---|---|---|---|---|---|
| tuned masked RFM | 3/3 | 33.3 | 16.3 | 0/3 | 39.8 | 4.8 |
| fitted global | 0/3 | 0.87 | 1.55 | 0/3 | 9.8 | 1.98 |
| fitted spatial | 1/3 | 2.07 | 1.67 | 0/3 | 7.9 | 1.77 |

Per scene, the spatial edit's background change is:
- −0.5 target: −0.29, −0.39, −0.27
- −2.0 target: −1.46, −1.60, −1.50

Its subject-interior MAE at −2.0 is 5.0, 8.1 and 10.6 (the limit is 3).

### Decision (frozen rule)

- The spatial edit passes all five criteria in 0/3 scenes at both levels, and R + S + A in 0/3. The global edit passes R + S + A in 0/3.
- **NO-GO for this architecture, inconclusive: the optimizer failed its control.**
- Failure modes of the spatial edit:
  - at −0.5: subject kept only by missing the blur (scenes 0 and 2); background unlike the target (scene 1);
  - at −2.0: misses the blur and changes the subject (all 3).

### What the result does and does not show

1. **It is budget-limited, not a ceiling.** At −2.0, all six fitted edits reached their lowest loss at the last iteration (24). The spatial losses were still falling by about 2 per iteration (scene 0: 45.7 → 36.5 over the last 5). Together with the failed control, this means the test did not find out what this edit can express. Neither "it cannot blur the background while keeping the subject" nor "a better optimizer would pass" is established.
2. **Spatial beat global under equal conditions, but not by the frozen margin.** At −2.0 the spatial edit reached about the same blur with 19% lower interior error (7.9 vs 9.8) and lower band error (12.5 vs 15.7). Criterion V needs a 50% reduction. At −0.5 the global edit barely moved the background (−0.18 to 0.00, and coefficient 0 was best on scene 2). The spatial edit moved further but still missed ±0.15 on 2 of 3 scenes.
3. **The background change is not the target's blur.** In every fitted run, the background ended further from the rendered target than the unedited image was (ratio 1.4-2.5; criterion A needs at most 0.5). The edits lower the sharpness score partly by changing texture and colour. Scene 0's street goes grey, as with the global edit in `examples_T2.png`. They do not reproduce a Gaussian blur.
4. **Even with privileged fitting, the edits preserve the subject far better than the RFM steering of rounds 9-13.** Interior error is 8-10 vs 40 at −2.0, with layout IoU 0.96 vs 0.47. The boundary crops show no doubled outlines. A fitted, position-dependent edit at a decoder layer is much less destructive than one direction at the encoder, but it still does not meet a 3/255 subject bar.

### What would make this conclusive (not run)

- A fit that passes the control: smaller shared step or warm-up, and about 100 iterations. At 52 s per iteration, that is about 1.5 h per fit, so it must be split across pods by scene because of the 6-hour limit.
- If the control then passes and the spatial edit still misses R or S, that is a genuine NO-GO for this layer and basis. The basis captures only 49% of the target change, which is the first thing to vary.
