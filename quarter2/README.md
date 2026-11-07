# Quarter 2: Predicting When a Steering Intervention Will Help

**Status:** first smoke test done (Nov 5, 2026, 2 h 45 min on a GTX 1080 Ti);
it was inconclusive because the task was too hard (below). Quarter 1 results and the earlier MNIST
attribute-preservation idea are in [../notes.md](../notes.md).

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

## Files and how to run

| File | What it does |
| --- | --- |
| `prompts.json`, `dev_prompts.json` | 24 test prompts; 4 separate prompts for choosing pulse strengths |
| `sd_branch.py` | SD 1.5 baselines (saving latents after steps 6, 15, 24) and branched continuations |
| `evaluate.py` | Automated checklist scores: OWLv2 boxes (objects, left–right), CLIP ViT-L/14 colors |
| `analyze.py` | Pass/fail checks, success mix, strength pick, and the repair/damage/oracle report |
| `dsmlp_smoke.sh` | The whole smoke test in one GPU pod |

From the repository root, in a GPU pod:

```bash
bash quarter2/dsmlp_smoke.sh      # about 3 hours on a GTX 1080 Ti; outputs in outputs/q2_smoke/
```

Submitted from dsmlp-login as a 12-hour batch pod:

```bash
K8S_TIMEOUT_SECONDS=43200 launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 8 -m 32 -B -N q2-smoke -- \
    bash -c "cd ~/dsc180-diffusion-steering && bash quarter2/dsmlp_smoke.sh"
```

Settings: DDIM (eta 0), 30 steps, CFG 7.5, float32, 512×512, seeds 0–7 (dev
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
