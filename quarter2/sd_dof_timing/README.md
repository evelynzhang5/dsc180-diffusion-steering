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

## Running

`bash quarter2/sd_dof_timing/dsmlp_timing.sh` on a DSMLP GPU pod. It needs round 9's `directions.pt`; round 12's
`diag.py` and its dependencies are copied onto this branch unchanged.

## Status

- Nov 15, 2026: design frozen, code written; launched as `q2-sd-dof-timing`.
