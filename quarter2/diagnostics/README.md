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

## Status

- Nov 14, 2026: code written after round 11's pilots finished. Launched as `q2-diagnostics` (waiting for a GPU).
