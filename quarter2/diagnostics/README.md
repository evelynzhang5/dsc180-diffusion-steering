# Q2 round 12: evaluation fixes, then idea 1's feasibility test (SD 1.5 depth of field)

Branch `q2-diagnostics`. Started Nov 14, 2026. The index of all Q2 rounds is in [../README.md](../README.md).
Diagnostics, not results: 5-6 scenes, one seed.

**Why.** Round 10 compared masked and global edits at the same steering strength. The masked edit blurred the
background far more, so that comparison was not fair. Round 11's response pilot showed the linear model it relied on
was badly wrong, with errors of 60-90% under scaling. Round 12 follows the mentor-style review of Nov 14:

- Fix the evaluation first.
- Then ask one question: do useful measured-response corrections exist at all, before anything is trained?

## Evaluation (every stage)

| question | measure |
|---|---|
| Did the requested background change happen? | Paired log Laplacian variance change in the background (> 12 px from the subject). **Achievement** = within 0.15 of the target. |
| Did the subject stay the same? | Pixel RMS vs the unedited reference in the subject and its interior; SSIM over the subject; DINOv2 of the subject crop. |
| Did the boundary behave? | Error in the band around the subject against an approximate rendered target: depth blur with the subject pasted back, at the blur level matching the method's background change. |
| Did the method just avoid the edit? | Achievement is reported next to every preservation number. |

- **Subject masks** come from CLIPSeg with the subject noun, applied to the unedited reference only. They are checked
  against the depth mask and on overlay sheets. The nearest-40% depth mask is a depth partition, not a subject mask.
- **Comparisons** are always made at the same achieved background change, never at the same steering strength.
- **The uniform-blur control** is a matched curve: a uniform blur reaching the same background change on the same
  scene. It is not a constant to subtract.
- **Copy-back baseline:** the edited image with the original subject pasted back through a feathered mask. It
  preserves the interior trivially, so any new method must beat it somewhere else (boundary, realism).

## Stages (`diag.py`, one pod: `dsmlp_diag.sh`)

| stage | what | answers |
|---|---|---|
| `evalfix` | Rescore round 10's smoke images (global and masked at s 2 and 4) with the fixed measures. Add matched curves: post-hoc subject blur at sigma 0.25-3 (sub-pixel included), uniform blur, and copy-back. | Masked vs global at equal achieved blur. How much the selectivity depends on the mask convention. |
| `vaecheck` | VAE round-trip loss on our generated references and on real photos, vs the subject damage from steering. | Idea 3: generated images are D(z) and were never encoded. If steering damage is many times the round-trip loss, residual reinjection cannot be the fix. |
| `geometry2` | Round 11's pilot 2, plus a clean-geometry baseline and a soft depth mask. The ensemble's subject cut is now drawn around the method's own (wrong) threshold. Round 11 drew it around the true one, which hid the threshold error. | Idea 2: does any rule beat plain rendering once the confound is removed? |
| `feasibility` | Idea 1, six scenes, target background change -0.5. (1) Tune masked RFM, masked RFM with a 16 px-dilated subject, and global RFM to the target (secant search, at most 3 generations each). (2) Around the tuned masked edit, measure the response of the signed subject pixels (vs the original reference, so existing damage counts) and of the background to 5 components (RFM / 2nd eigenvector on background / subject blocks, paired direction on background), with + and - perturbations. (3) Solve for a correction in a box of +/-1 that keeps the background at the target. (4) **Generate it**; take one more step. (5) Compare predicted with realised. | The decision table below. |

Feasibility controls:
- a random correction of the same size and the same predicted background change;
- a 10% edit (the noise floor: how much any small perturbation already moves the subject);
- copy-back;
- a post-hoc blur matched to the target.

| result | next action |
|---|---|
| Corrections do not beat tuned masking | Reconsider the basis or mechanism before training |
| Corrections work, but predictions are unreliable | Investigate locality, step size, nonlinear response |
| Corrections work reliably on individual scenes | Test whether a reusable predictor transfers to unseen scenes |
| A reusable predictor keeps the advantage at reasonable cost | Begin the method study |

Per-scene probing is a feasibility diagnostic. Success is not evidence of reusable steering and not a proven upper
bound.

## Results (Nov 15, 2026; files in [results/](results/))

All stages ran (GTX 1080 Ti, 03:29-04:26, 57 min). One check failed in `evalfix`: CLIPSeg found no bird in the
small-bird scene (mask area 0). That scene was skipped and the other four were scored. Small subjects need a
better segmenter.

**Evaluation fixes (`evalfix`, 4 scenes).**
- **Matched comparison at a background change of -0.28:**

  | method | subject sharpness | subject RMS | subject DINOv2 |
  |---|---|---|---|
  | masked | +0.01 | 0.139 | 0.87 |
  | global | -0.14 | 0.169 | 0.81 |
  | post-hoc blur (sigma 0.34 px) | 0 | 0.000 | 1.00 |

  - Masking does make the *sharpness* effect selective: +0.27 above a matched uniform blur, close to post-hoc blur's
    +0.28.
  - But masking does not preserve the subject, and post-hoc blur beats every steering method on every preservation
    measure.
- **The global edit reaches the target less often:** -0.5 in 2 of 4 scenes, against 3 of 4 for masked.
- **Copy-back does not fix moved subjects.** It keeps the interior exactly by construction, but its boundary error is
  10x post-hoc blur's (0.059 vs 0.006 at -0.5). The sheet shows why: the steered subject has moved, so pasting the
  original produces ghosts and double outlines (two deer, a doubled dog).
- **The mask convention matters a lot.** The sign of the selectivity score flips between the depth mask and the
  CLIPSeg subject mask in 3 of 8 scene-method cells. IoU between the two masks is only 0.22-0.48: the depth mask
  covers ground and tree trunks.

**Idea 3 (`vaecheck`).** Steering moves the subject 6-7x more than a VAE round trip does (subject RMS 0.19 vs 0.03).
Round trip: 28.6 dB on our generated images, 30.4 dB on real photos. Residual reinjection cannot be the fix for
steering damage. It remains a separate, known problem for editing real photos.

**Idea 2 (`geometry2`).** With the confound removed, nothing beats plain rendering.
- Ensemble boundary error under the threshold error: 0.0159 vs 0.0175 at sigma 8. Round 11 reported 0.0039 because
  of the confound.
- The soft mask and the sigma-wide feather match plain rendering under every error.
- With clean depth, every protection rule only adds error.

**Idea 1 feasibility (`feasibility`, 6 scenes, target -0.5). Outcome: the first row of the decision table.**
Measured-response corrections do not beat tuned masking.
- A correction beat the best tuned masking by at least 10% subject RMS in **2 of 6** scenes, and beat the random
  control in **3 of 6**. The correction shows no demonstrated advantage over a random one (six scenes cannot
  establish that it performs at chance).
- **The local linear approximation was unreliable at the tested perturbation sizes (+/- 0.5).** The second
  difference of the subject pixels was 0.5-2.3 times the first difference for every component, mostly above 1.
  This does not show the response is non-smooth: a smooth but strongly curved function gives the same pattern.
  Two things were not checked: that identical inputs and sampler states reproduce identical outputs (round 13 checks
  this), and smaller steps.
  - Linear predictions of the corrected subject RMS were within 25% in 4 of 6 scenes, but only because the
    corrections barely moved it.
  - The background often missed: -1.15 and -0.04 against a predicted -0.5.
- **The noise floor is large.** A 10% edit already moves the subject by 0.076 RMS on average, 38% of the full tuned
  edit's 0.20 (0.16 of 0.28 for the cat). In the sheet the dog, deer and cat change pose or position under every
  edit.
- **No correction beats copy-back on the boundary (0 of 6), and copy-back itself loses to post-hoc blur.**
- **Cost:** 22 generations (about 8 min) per scene, against 3 or fewer for a tuned mask.

**Reading (Nov 15, revised after review).** Established, for these settings and small scene sets:
- masking improves where the softness appears but does not preserve the subject;
- the tested linear correction does not improve on tuned masking, costs far more, and has no demonstrated advantage
  over a random correction;
- for the mild blur tested, post-processing is the strongest preservation baseline.

This is enough to pause the correction approach. It does not show that all activation steering or response-based
methods fail.

A hypothesis, not yet tested: the window starts at sigma 13, where the layout is still being decided, so early
steering re-lays-out the scene (pose and position change). Round 13 (`q2-sd-dof-timing`) tests this.

## Status

- Nov 14, 2026: code written after round 11's pilots finished. Launched as `q2-diagnostics`.
- Nov 15, 2026: ran; results above. Closed.
