# Quarter 2: What Makes a Steering Intervention Control the Final Image?

**Status (Nov 7, 2026):** the text-to-image pilots (SD 1.5, `sd_pilots/`) are finished and archived as a
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
| readable: held-out probe R^2 | 0.57 | 0.82 | 0.87 |
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

- **No attenuation after release:** once the edit is off, the thickness difference stays flat to the end (kept
  0.85-1.13 across layers, starts, signs and both step counts). The earlier "one-step edits fade" was the edit's
  direct effect on the network's guess disappearing when switched off, not later steps undoing it.
- **Step count does not matter at matched duration:** 16 and 64 steps agree within 0.02 SD everywhere.
- **What shrinks late is sensitivity:** the same relative activation edit moves the guess about 7x less at t0 = 0.75
  than at 0.25, while the sampler's per-step weight on the guess only falls about 2x (DDIM: 1 - tan(pi t/2) /
  tan(pi s/2) for a step t -> s). In the local model dz_(k+1) = A_k dz_k + B_k u_k, thickness behaves as if A_k
  keeps it (about 1) and B_k falls with t.
- **The state change is thickness-aligned:** per unit of state change at release, the learned direction gives
  4-6 SD of thickness, random directions about 0. The state difference itself keeps growing after release (x1.5-3,
  for random directions too), so state norm is not the right persistence measure; the attribute is.

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
| `mnist_release.py` | Release-and-follow: edit on for a fixed time window, then off; follows both trajectories at 16 and 64 steps |

```bash
python quarter2/mnist/mnist_thickness.py --out outputs/q2_mnist_thickness/steps16 --steps 16 --n 1000
python quarter2/mnist/mnist_release.py --out outputs/q2_mnist_release --n 1000
```

Both use `generator.pt`, `evaluator.pt` and `mnist_guidance_lab.py` from the repository root and run on a laptop
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
