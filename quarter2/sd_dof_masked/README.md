# Q2 round 10: does restricting the RFM edit to the background make depth of field selective? (SD 1.5)

Branch `q2-sd-dof-masked`. Started Nov 14, 2026. The index of all Q2 rounds is in [../README.md](../README.md).

**Why.** Round 9 (`q2-sd-dof-direction`) found:
- **The edit is global.** The paper's RFM direction lowers background sharpness (-0.28) but foreground sharpness more
  (-0.39), with pixel change about equal inside and outside the subject.
- **No other direction did better.**
- **A post-hoc background blur scores far higher** (log FG/BG ratio +0.65).

The simplest fix to test before learning anything: apply the same edit only at background positions of the steered
layer. This is the first comparison in [the proposal](../../proposal.md) ("tuned RFM vs. background-masked RFM").
If it already works at matched blur, it becomes the baseline that any learned correction has to beat.

**Status.**
- Nov 14, 2026: smoke tests 1-3 written; one pod (`q2-sd-dof-masked-smoke`) runs them one after another. No
  design frozen yet; the smoke images are a pipeline check and are not used for any decision. The dev/test design is
  written and frozen before any dev image.
- Nov 14, 2026: all three smoke tests passed every check (one pod, 17 min on the GPU after a 3.5 h wait; pod
  deleted). Results: `results/smoke/`. See "Smoke results" below.

## Smoke test 1: masked steering and the measures

`python quarter2/sd_dof_masked/sd_dof_masked.py smoke` (on a GPU pod: `dsmlp_sd_dof_masked.sh smoke`).

**Captions** (6, seed 42), hand-written to cover:
- 2 ordinary scenes;
- 4 hard cases: no subject, two subjects at different depths, a small subject, and a request for a sharp background.

**Conditions:**

| condition | what it is |
|---|---|
| `none` | unsteered, text CFG 7.5 |
| `rfm_global_s{2,4}` | round 4/9's edit at `down_blocks[2].resnets[-1]`, everywhere |
| `rfm_masked_s{2,4}` | the same edit, multiplied by the unsteered image's depth background mask |
| `posthoc_r{1,3}` | the unsteered image, blurred outside its depth foreground (round 9's oracle) |
| `uniform_r2` | the unsteered image, blurred everywhere |

**The background mask.** The farthest 60% of pixels by Depth Anything V2. Each of the layer's 16 x 16 positions gets
the fraction of background pixels in its 32 x 32 block. The edit's per-image scale (0.7 |h|) is unchanged; the edit
is only zeroed or weakened where the subject is.

**Checks** (pass or fail, written to `smoke_checks.md`; the script exits with an error if any fails):

| check | expected |
|---|---|
| unsteered image vs itself | every change exactly 0 |
| masked hook | layer output unchanged where the mask is 0, changed where it is > 0 |
| post-hoc blur | background drops, foreground change under a quarter of it |
| post-hoc vs uniform blur | ratio at least 0.3 higher: the measure rewards selective blur, not any blur |
| fixed-mask and own-mask ratios | finite for every generated condition |
| matched-blur matching | a strength, or "out of range", per method (target -0.28) |
| metrics | all finite; runtime and peak memory recorded |
| contact sheet | 24 images or fewer |

The two blur checks test the measure itself: a uniform blur and "no edit" should not look selective.

Open choices for the design: soft vs. hard mask; which mask (depth vs. subject segmentation); which steps; the
mask source (the unsteered image costs a second generation).

## Smoke test 2: paired data (`smoke_paired.py paired`)

- **Images:** 8 sharp SD 1.5 images (the 6 captions above plus 2 ordinary ones, seed 42).
- **Rendered blur:** Depth Anything V2 inverse depth gives a depth-dependent Gaussian blur at 3 levels (maximum sigma
  2, 4, 8 px). The nearest 40% of pixels, which is the meters' foreground, is left exactly sharp.
- **Paired activations:** round 4's layer and noise level, with the same noise and the caption for the original and
  each edit.
- **Paired direction:** the mean of the per-pair changes at the strongest level. This equals the difference of the
  two group means, so it can only differ from round 9's mean-difference direction through the data.
- **Steered once:** the paired direction is run at s 4.

**Checks:**
- rendered foreground unchanged, and background change growing with level;
- the same noise reproduces the same activations;
- activation change growing with level.

**Reported for information:**
- cosine of the paired direction with round 9's directions;
- how much the per-pair changes share one direction;
- where in the 16 x 16 grid the paired change lands (background vs foreground positions).

## Smoke test 3: reference methods

**LoRA on the same pairs (`smoke_paired.py lora`).** A rank-4 LoRA on the UNet attention, trained for 50 steps
(plain diffusion loss) on the 8 strongest-level edited images with their captions. It is then saved, reloaded into a
fresh pipeline, and sampled at adapter weights 0, 0.5 and 1.

**Checks:**
- the fixed evaluation loss goes down;
- weight 0 reproduces the plain model;
- weight 1 changes the image.

This tests the training, saving and loading path only; it is not a trained baseline.

**Bokeh Diffusion (checked Nov 14, 2026, no GPU needed).**
- **Released:** the official repo `atfortes/BokehDiffusion` has inference code and one checkpoint,
  `atfortes/BokehDiffusion` on Hugging Face, for **FLUX only**. The dataset is `atfortes/BokehDiffusion`.
- **Not released:** training code, and any SD 1.5 weights.
- **Requirements:** PyTorch 2.6, CUDA 12.6, flash-attn. FLUX.1-dev is a 12B-parameter gated model, too large for the
  course's small-GPU pods.
- **License:** NTU S-Lab License 1.0 (FLUX.1-dev has its own non-commercial license).

So it cannot run on our SD 1.5 setup or our pods. Any comparison with it is against its released FLUX model on a
larger GPU, labelled as a different backbone. Without training code, it cannot be retrained on our pairs.

## Smoke results (Nov 14, 2026; pipeline checks on 6-8 images, not results)

Every pass/fail check passed in all three tests: `results/smoke/{smoke,paired,lora}_checks.md`.

**Things the smoke test showed that the design has to handle:**
- **The masked edit still changes the subject.**
  - Sharpness says it is selective: at s 4, background -0.93, foreground -0.16, ratio +0.78, against -0.24 for the
    global edit.
  - Pixels say it is not: pixel change is about equal inside and outside the subject (0.079 / 0.090), and DINOv2
    similarity is 0.69.
  - In `smoke_examples.png` (columns: none, global s 4, masked s 4, post-hoc blur), the masked images have a
    different dog pose, a different street, a different deer.
  - Zeroing the edit at foreground positions of one layer does not keep the foreground: later layers spread the
    change. The design needs a subject-preservation measure (pixel or DINOv2 change inside the mask) next to the
    sharpness ratio, and the ratio alone must not decide.
- **A uniform blur scores slightly selective** (ratio +0.20). The measure has a small bias, so the design needs a
  uniform-blur control at matched background change, not a comparison with 0.
- **Post-hoc blur radii are too coarse.** Even radius 1 drops background sharpness by -2.4, so matching at round 9's
  -0.28 needs radii below 1 px (the "0.11" in the checks is a linear guess between 0 and 1).
- **The depth foreground can be the wrong region.** For the small bird on a beach, the nearest 40% is the sand, so
  the rendered pair blurs the bird. Hard cases need a subject mask, not only "nearest 40%".
- **Per-pair changes carry spatial information.**
  - Their energy is 7x higher at background positions than at foreground ones (median over 8 images).
  - They barely share a direction (mean cosine 0.12).
  - Their mean has cosine 0.13 with round 9's RFM direction and 0.16 with its mean difference.
  - This supports the proposal's point that a single averaged direction throws away what the pairs contain.
- **The LoRA path works** (797k parameters, 66 s for 50 steps, peak 4.9 GB). At weight 1, 50 steps on 8 images
  mostly changed the composition, which is expected from so little training.
- **Cost:** a steered generation takes 21.9 s per image against 14.9 s unsteered (third CFG branch), with the same
  peak memory (6.8 GB). Masking adds nothing beyond the unsteered pass that makes the mask.
