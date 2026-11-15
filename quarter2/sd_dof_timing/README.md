# Q2 round 13: does delaying the same edit preserve the scene at the same background blur? (SD 1.5 depth of field)

Branch `q2-sd-dof-timing`. Started Nov 15, 2026. The index of all Q2 rounds is in [../README.md](../README.md).
A bounded diagnostic with a stopping rule, not a method. The correction approach (round 12) is paused.

**Question.** Can delaying the same activation edit preserve scene content while reaching the same background blur?
In plain terms: wait until the scene is mostly formed, then change its appearance.

**Prior work that bounds this.** Delaying the edit is a baseline, not a contribution:
- Limited Interval Guidance (arXiv 2404.07724) shows guidance timing matters, but favours a middle interval, not
  "later is always better".
- SteeringDiffusion suppresses early activation modulation and ramps it up toward the end.
- Bokeh Diffusion reports that changing blur changes content even with the same prompt and seed, and fixes it with
  reference-grounded attention.

A contribution would need an advantage over that kind of baseline: less training, lower inference cost, better
preservation on difficult boundaries, or more reliable control across scenes.

## Design (frozen before the run)

- **Fixed:**
  - round 9's paper direction (`rfm_eig1`), round 4's layer, edit size and third-branch strength (w 0.7, s 4);
  - caption, seed 42, text CFG 7.5, 50-step DDIM;
  - the edit masked to the background as in round 12 (non-subject fraction per 16x16 block, from the unedited
    image's subject mask).
- **Varied: only when the edit is applied.**
  - `full`: round 4's window, every step with sigma in [0.041, 13.12].
  - `early`, `middle`, `late`: equal-length runs, each a third of that window, at its start, centre and end. Actual
    steps and sigma ranges are recorded.
- **Same starting point.** Each steered run starts from the unsteered trajectory's own cached latent at the window's
  first step, so the comparison is about when the edit begins.
- **Same achieved blur.** Strength is tuned separately for each window and target (secant search, at most 4
  generations, c up to 24).
  - Targets: -0.5 (round 12's mild target) and **-2.0**, chosen before testing. Round 10's 1 px post-hoc blur gave
    about -2.4, so -2.0 is a visible softening.
  - A window that cannot reach a target (within 0.15) counts as a control failure, not a preservation success.
- **Baseline:** post-hoc blur outside the subject, matched to each target.
- **Measures:** round 12's, with masks fixed from the original image:
  - subject pixel RMS, interior RMS, SSIM and DINOv2;
  - boundary error against an approximate rendered target;
  - layout: IoU between the original subject mask and the subject found again on the edited image (same prompt and
    threshold), plus the shift of its centre.
- **Scenes:** 7: round 12's six plus the small bird that round 12's masker lost.
  - The automatic masker's failure is reported.
  - For the bird, other prompts and thresholds are tried. The mask is then checked independently on `timing_masks.png`
    before its scene is used in any conclusion.
- **Sampler checks:**
  - the unsteered sampler run twice gives identical images, and so does the steered one;
  - the cached sampler reproduces round 4's sampler;
  - a zero edit with the third branch matches unsteered up to numerics.

**Decision**, for the later window (`middle` or `late`, whichever wins more scenes), at each target:

| result | rule | action |
|---|---|---|
| 1. Later editing reaches the target and preserves the subject better | reached in >= 5 of 7, subject RMS <= 0.8 x the full window's (both on target) in >= 5 of 7, mean IoU >= 0.8 | Timing fixes part of the problem; confirm on new scenes and seeds before anything more complex |
| 2. Preserves the subject but cannot reach the target | preserved in >= 5 of 7, reached in < 5 | No demonstrated improvement in the trade-off |
| 3. Reaches the target but still changes pose/layout | reached in >= 5, rule 1 fails | Early timing is not the whole explanation; stop this direction-and-layer approach |
| 4. Post-processing better at both blur levels | post-hoc subject RMS and boundary error <= every window that reached the target, at both targets | The task does not justify a generative steering method |

Seven scenes decide only whether to continue; any promising setting must then hold on untouched scenes and seeds.
FID is not used: it cannot tell whether the particular dog, person or scene survived the edit.

## Results (Nov 15, 2026; files in [results/](results/))

**Every check passed.**
- **Sampler:** run twice it gives identical images, both unsteered and steered, and it reproduces round 4's sampler
  exactly.
- **Zero edit:** adding the third branch with no edit changes a pixel by at most 1 level (mean 0.0006).
- **Windows:**

  | window | steps | sigma |
  |---|---|---|
  | early | 0-15 | 13.1 -> 3.1 |
  | middle | 17-32 | 2.7 -> 0.96 |
  | late | 34-49 | 0.84 -> 0.04 |

- **The bird:** the automatic masker failed on it again ("bird" at threshold 0.4 found nothing). "small bird" at 0.3
  found it (290 px), and the overlay sheet confirms the mask sits on the bird. Its layout IoU on edited images is
  unreliable (0 in most cells), so the bird is read only through subject RMS.

**Target -0.5, means over the scenes that reached the target:**

| window | reached | subject RMS | subject-mask IoU | centre shift (px) |
|---|---|---|---|---|
| full | 7/7 | 0.185 | 0.48 | 21 |
| early | 6/7 | 0.181 | 0.51 | 37 |
| middle | 5/7 | 0.144 | 0.58 | 82 |
| late | **1/7** | 0.087 | 0.81 | 8 |
| post-hoc | 7/7 | < 0.001 | 1.00 | 0 |

- **Target -2.0:** no window reached it in more than 2 of 7 scenes (full 2, early 1, middle 1, late 0).
- **The full-window images that did reach -2.0 are badly degraded:** smeared, with the scene dissolved
  (`timing_examples.png`).
- **The late window cannot produce the blur.** Even at c up to 24, its background change stayed between -0.02 and
  -0.88. At high strength it adds streaks and ghosting instead of blur.

**Post-hoc row: corrected after the run.** The frozen run matched post-hoc blur on a radius grid with steps of
0.1-0.25 px. That was too coarse: it reported 3/7 reached, because radius 0.4 gave -0.29 and 0.5 gave -0.72 for the
dog. Recomputed on a 0.02 px grid from the saved reference images and masks (`results/posthoc_fine.json`):
- target -0.5: post-hoc reaches the target in **7/7** scenes, at sigma 0.42-0.50 px;
- target -2.0: **7/7**, at sigma 0.66-1.32 px;
- subject RMS at most 0.003 in every case.

This correction only strengthens the row-4 outcome.

**Decision (frozen rule):**
- **Target -0.5: row 3.** The best later window (middle) reaches the target in 5/7 scenes but beats the full window
  by 20% subject RMS, while on target, in only 3/7. The subject still changes (IoU 0.58).
- **Target -2.0: row 2.** Later windows preserve the subject better in 6/7 scenes but reach the target in at most 1/7.
- **Row 4 at both targets.** Post-processing beats every window that reached the target, on subject change and on
  boundary error.
- **Action: stop this direction-and-layer approach.** The task, as posed, does not justify a generative steering
  method.

**What the timing test says about the hypothesis.** Supported in part, and not the whole story:
- **Supported:** the early third does almost all the layout damage. Its subject change and IoU match the full window's
  (0.181 / 0.51 vs 0.185 / 0.48), and delaying the edit does keep the layout (late IoU 0.81).
- **But the edit's background softening is produced in the same early and middle steps.** Once the layout is set,
  this edit can no longer blur. With this direction at this layer, softness and layout change come from the same phase
  of sampling, so timing cannot separate them.
- **Limits:** one seed, 7 scenes, one direction and layer. The middle window's large centre shift (82 px) comes
  mostly from masks that broke on edited images (IoU 0 for the woman and the bird), not from measured moves.

## Running

`bash quarter2/sd_dof_timing/dsmlp_timing.sh` on a DSMLP GPU pod. It needs round 9's `directions.pt`; round 12's
`diag.py` and its dependencies are copied onto this branch unchanged.

## Status

- Nov 15, 2026: design frozen, code written; launched as `q2-sd-dof-timing`.
- Nov 15, 2026: ran on a GTX 1080 Ti (06:50-07:42, 52 min); every check passed; results above. Closed.
