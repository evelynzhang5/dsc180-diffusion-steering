# Proposal: scene-preserving depth-of-field control by activation steering

Draft, Nov 14, 2026. A three-person project for next quarter, building on the Q2 rounds in
[quarter2/README.md](quarter2/README.md).

## Question

**Can paired examples teach a compact activation-steering method to control where and how much an image is defocused,
while keeping the scene the same?**

Target behaviour: generate "a cat beside a fence", then increase background blur over five levels. The cat, the
fence, the framing and the foreground detail stay the same.

## Why: what our rounds showed

We replicated NA-RFM's Stable Diffusion 1.5 depth-of-field experiment (paper Appendix G.2) and tested it in two
pre-registered rounds:

- **Round 4.** Background sharpness fell 35.6% (paper: 27.6%), but foreground sharpness fell about as much. The paired
  change in log FG/BG sharpness ratio was +0.02 (95% CI -0.22 to +0.24).
- **Round 9**, on test captions with 256 images per condition, using the RFM direction at s 4:
  - **Global edit:** background sharpness fell by -0.28 (log) and foreground sharpness by -0.39. Pixel change was
    about equal inside and outside the subject mask (0.127 / 0.119).
  - **Ratio direction depends on the mask:**
    - With the depth mask held fixed to the unsteered image, the FG/BG ratio *fell* by -0.11.
    - With the mask recomputed on each steered image, as the paper does, it *rose* by +0.15. This matches the paper's
      2.25 -> 2.58 (log +0.14).
    - The steered images change layout and content (DINOv2 cosine with the unsteered image 0.68), so the mask moves
      with them.
  - **Better scores than the RFM direction, with no generation:**
    - A post-hoc blur outside the depth mask scored +0.65.
    - No alternative direction (2nd and 3rd RFM eigenvectors, mean difference) was more selective.

Two lessons carry into this project:

1. **Steering direction is not enough.** A single global direction learned from unpaired "blurry vs sharp" photos
   does not make the background blur selective. Changing which global direction we use does not fix this.
2. **The measurement must be paired.** Preservation must be measured against the unedited image with a fixed region,
   otherwise layout change looks like selective blur.

## Related work and what we cannot claim

| Source | What it gives us | What is not new |
|---|---|---|
| NA-RFM | Our reference-trained steering baseline and code | Activation steering on SD |
| Bokeh Diffusion | Blur conditioning plus attention anchored to a reference generation to keep scene structure | Continuous, scene-preserving bokeh control itself |
| CamEdit / CamEdit50K (NeurIPS 2025) | Paired camera edits with numeric parameters; learns focal plane, aperture and shutter controls with parameter embeddings, feature modulation and LoRA | Learning continuous camera controls from paired images |

So adding attention localization or a background mask is a **baseline**, not a contribution. "We freeze the
backbone" is not enough either, because small adapters already do that. The Bokeh Diffusion and CamEdit descriptions
come from our reading notes and must be checked against the papers before the proposal is final.

**Candidate contribution.** A compact steering representation learned from paired edits that transfers across
scenes. It has to show a measurable advantage over a LoRA trained on the same pairs in at least one of:

- preservation;
- the amount of training data needed;
- inference cost.

This is a hypothesis to test, not a result.

## Data

**CamEdit50K (checked on Nov 14, 2026, Hugging Face `wangzike/CamEdit50K`):**
- **Focal plane only.** The public release is titled "CamEdit50K - Focal Plane". Its `parameter` values describe
  which depth stays sharp, not how strong the blur is.
- **Contents:** 26,620 rows in the parquet index, with columns original/edited image path, instruction, task and
  parameter.
- **Files:** 9 parquet files but only 8 image tar files (`CamEdit_Focal_009.tar` is not listed). Some rows may have no
  images, so we need to count usable pairs.
- **License:** CC-BY-NC-4.0, so non-commercial use only. That is fine for coursework, but must be stated.
- **Not enough on its own.** This subset alone cannot supervise blur *strength*. We first look for aperture or
  blur-strength pairs from the authors.

**If no blur-strength pairs exist: a documented synthetic set.**
1. Generate a sharp SD 1.5 image.
2. Estimate depth with Depth Anything V2.
3. Render depth-dependent blur at known levels.

This gives exact paired targets and known strengths. The risk is that a method learns to imitate our renderer, and
the renderer is then trivially best on synthetic metrics. Synthetic pairs are therefore for training and controlled
experiments only. Claims are made on a separate real-image evaluation set: CamEdit pairs and real shallow-depth-of-field
photos.

**Splits.** Splits are made by scene, not by pair, so no scene appears in both training and test.

## Method

The generator (SD 1.5) stays frozen. We learn a steering basis and a small controller, so the method is **not**
training-free.

1. **Learn the change from pairs.** Collect activations for the original and the edited image under the same noise
   and conditioning.
2. **Apply it according to scene geometry.** A small controller takes depth or a subject mask and the requested
   level. It decides which spatial positions get which edit, and how strongly.
3. **Train for both outcomes.** Reward the requested defocus pattern, and penalize changes to the protected subject
   and to scene structure.

**Pairing alone is not a new estimator.** Averaging paired differences gives the difference of means:

$$\frac{1}{N}\sum_i \left(h_i^{\text{edited}} - h_i^{\text{original}}\right) = \bar h^{\text{edited}} - \bar h^{\text{original}}$$

That is the mean-difference direction we already tested. Pairs only add information when the method keeps one of
these:

- **Per-pair spatial changes:** each pair's change at each position, not just the average.
- **Depth and level dependence:** how the change depends on depth and on the requested level.
- **Per-pair preservation losses:** losses computed against each pair's own unedited image.

## Experiments

Each comparison separates one explanation:

| Comparison | What it tells us |
|---|---|
| Unpaired references (NA-RFM) vs. paired references | Whether better supervision alone fixes selectivity |
| Global edit vs. masked edit vs. learned spatial edit | Whether localization needs a learned mechanism |
| Fixed strength vs. calibrated control | Whether gains are ordinary strength tuning |

**Baselines**, all at matched background blur where possible:

- NA-RFM global direction (rounds 4 and 9);
- **masked steering**: the same direction applied only where the depth mask is background (round 10);
- **post-hoc depth-based blur**, round 9's oracle generalized to several levels. A generative method has to justify
  its extra cost against it, for example with better boundaries, realistic bokeh, or difficult structure;
- **LoRA on SD 1.5 trained on the same pairs**, the key comparison for the "compact steering" claim;
- **released Bokeh Diffusion and CamEdit models**, as references, labelled with their different backbones and
  training data.

## Evaluation

Three outcomes are reported separately, not folded into one score:

- **Control:** requested vs. achieved defocus, and monotonic change across levels. Both are tested at held-out levels
  and on new scenes and subject categories.
- **Preservation:** foreground sharpness, subject identity and layout (DINOv2 against the unedited image), and
  artifacts at boundaries. Every measure is paired with the unedited image and uses its fixed mask (lesson 2).
  Difficult boundaries get their own subset: hair, fur, thin objects.
- **Quality and cost:** FID/KID against suitable targets with the same sample counts, plus runtime and memory.

**Checking the measures.** Preservation measures can reward doing nothing. Before any comparison, we check on known
pairs that each measure gives the intended blur a good score and penalizes no edit and global blur.

## Team

| Owner | Research question | Deliverables |
|---|---|---|
| Person 1: paired data and representation | Do pairs give cleaner, more transferable edits than group-level discrimination? | Dataset audit; scene-level splits; paired activation extraction; RFM and paired-direction baselines; learning curves at several data budgets |
| Person 2: spatial control and preservation | Where should steering enter the model to change the requested region without changing the subject? | Round 10 masking baselines; spatial controller; comparison with attention anchoring or latent preservation; boundary leakage analysis |
| Person 3: continuous control and evaluation | Does a requested level mean the same thing across scenes? | Calibration and monotonicity experiments; held-out levels; the fixed evaluation pipeline; quality, preservation and runtime comparisons |

Each person owns a question and its experiments. Together they build one method and write one report.

## Scope

**Committed:**
- **Task:** background defocus with the selected subject preserved.
- **Model:** SD 1.5, using our existing code.
- **Study:** several blur levels; new scenes and subject categories; difficult boundaries.

**Extensions, only after the main task works:**
- focal-plane movement using CamEdit pairs;
- one more backbone (rather than more attributes).

## Timeline (10 weeks)

| Weeks | Work | Ends with |
|---|---|---|
| 1-2 | Data audit, splits, evaluation pipeline and its validation, baselines (global, masked, post-hoc, LoRA) | **Decision point** (below) |
| 3-5 | Core method: paired representation, spatial controller, calibration | Frozen design for held-out tests |
| 6-8 | Held-out evaluation and ablations | Results tables |
| 9-10 | Report, website, demo | Deliverables |

**Decision point after week 2.** If paired data plus simple masking already solves the task, that is the baseline the
learned controller has to beat. The project then focuses on where it falls short: boundaries, calibration, or
transfer.

As in Q2, each experiment's design and selection rules are frozen before its test images are generated. Every
experiment runs on a DSMLP GPU.

## Risks and open questions

- **Blur-strength supervision may not exist publicly.** Then the claims rest on synthetic training plus real-image
  evaluation, and the risk of imitating our own renderer has to be addressed directly.
- **LoRA may match the steering method.** Then the contribution is analysis (what pairs add, where to steer, how to
  calibrate), not a new method. We say so in advance.
- **Laplacian sharpness falls when texture is removed, not only when it is blurred.** We need a defocus measure that
  can tell the two apart, or a check on pairs where the blur is known.
- **Related work is unconfirmed.** The Bokeh Diffusion and CamEdit claims above still have to be checked against the
  papers and released code.
