# Round 16: can a small spatial activation edit make the right depth-of-field edit at all?

**Status:** design frozen; probe pending.

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

Pending.

## Results

Pending.
