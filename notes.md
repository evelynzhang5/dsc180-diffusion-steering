# Experiment notes

What we found in Quarter 1, part by part, and the Quarter 2 project proposal
that follows from it. Reproduction commands are in [README.md](README.md); the
dated run log is at the end.

**Contents**
1. [Quarter 1 results and insights](#quarter-1-results-and-insights), lab parts 1–6, full-data EDM training, MNIST
2. [What the results add up to](#what-the-results-add-up-to)
3. [Quarter 2 proposal](#quarter-2-proposal-guide-toward-the-condition-while-protecting-what-it-leaves-open)
4. [Questions for the mentor](#questions-for-the-mentor)
5. [Run log](#run-log)

---

## Quarter 1 results and insights

### Lab part 1: unconditional sampling (pretrained EDM, CIFAR-10)
- **What we did:** sampled 30 images with EDM's 18-step sampler and recorded how
  3 images form as noise is removed (`figures/parts1_3/01`, `02`).
- **What we saw:** varied classes and backgrounds; most images are recognizable
  objects at 32×32.
- **Insight:** the pretrained model is a solid starting point, so guidance
  effects can be studied without training our own model.

### Lab part 2: class-conditional sampling
- **What we did:** 8 samples per class from the conditional model (`03`).
- **What we saw:** each row reads as its class, and variety within a class is
  still large (pose, color, background).
- **Insight:** conditioning alone (w = 1) already gives the right class most of
  the time while keeping variety.

### Lab part 3: classifier-free guidance (CFG)
**Prediction before looking:** as w goes from 1 to 4, samples look more like
cats, with stronger contrast and saturation and less variety.

**CFG grid** (`04`, "cat", w = 0, 1, 1.5, 2, 3, 4, same noise per column):
- w = 0 gives unrelated classes (bird, plane, car, deer, ship). From w = 1 every
  column is a cat, but layout and colors carry over from the w = 0 image with
  the same noise (the ship's teal background stays in column 7).
- From w = 1 to 4 the pose barely changes; contrast, saturation and edge
  sharpness rise, and faces become more frontal.
- At w = 3–4 some samples are oversaturated or posterized (column 8 turns bright
  orange; column 1 loses its cat shape at w = 4).

**Trajectories** (`05`, `06`, cat, 3 samples, w = 1 vs. w = 4; σ = 80 → 0.002):
1. *When does the class appear?* At w = 1 the estimate D(x; σ) is a brown blur
   at σ = 80–28, a blob at 8.4, a coarse layout at 3.3, and an animal only at
   1.1, and two of three are still ambiguous. At w = 4 a cat face shows at
   σ = 3.3, and all three are clearly cats by 1.1.
2. *When does the image stop changing?* For both weights the layout is fixed by
   about σ = 1.1; below 0.3 the estimates barely change.
3. *Low-noise differences:* w = 4 gives higher contrast, saturated orange, dark
   backgrounds and sharp edges, giving more prototypical cats but sometimes garish
   colors or exaggerated stripes. w = 1 is more natural but less clearly a cat.

**Insight:** the class is decided between σ ≈ 8 and 1. Strong guidance makes
the class commit earlier, but at low noise it mostly adds saturation. This is
what first suggested that w should depend on the noise level.

### Lab part 4: measuring with label accuracy and FID
- **Label accuracy** uses a ResNet-56 classifier (94.4% on real test images).
- **FID** compares the mean and spread of Inception features. The 2D toy
  (`figures/lab/07`) shows the two terms separately: a shifted mean gives
  FID 2.27, all from the mean term, and too little variety gives 0.69, all
  from the spread term.
- **Insight:** FID penalizes lost variety, not only bad images, which is
  exactly what strong guidance tends to cause.

### Lab part 5: CFG sweep (two networks, 1,000 images per w)

| w | 0 | 1 | 1.5 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| label accuracy | 11.5% | 95.5% | 98.7% | 98.8% | 99.5% | 98.9% |
| FID | 31.19 | 31.33 | 31.44 | 33.05 | 36.03 | 39.03 |
| FID_class | – | 112.1 | 107.6 | 108.9 | 112.4 | 117.5 |

- **What we saw:** accuracy is nearly maxed by w = 1.5; beyond that FID keeps
  rising. FID_class is best around w = 1.5.
- **Insight:** extra guidance stops helping the class but keeps hurting
  quality and variety. These FIDs use 1,000 images, so compare them only with
  each other.

### Lab part 6: training on 100 and 1,000 images, with CFG
7.0M-parameter EDM network, 10% label dropout, 20,000 updates per data size.

| update | 4,000 | 8,000 | 12,000 | 16,000 | 20,000 |
| --- | --- | --- | --- | --- | --- |
| FID, 100 images | 75.8 | 92.2 | 104.8 | 106.6 | 108.6 |
| FID, 1,000 images | 93.9 | 111.3 | 124.9 | 124.9 | 130.7 |

- **Memorization:** the 100-image model reproduces its training images (median
  distance to the nearest one 1.3, vs. 23.7 for real test images). The
  1,000-image model's samples are blurry rather than copies (14.0 vs. 21.3; blur
  alone lowers pixel distance, so this does not prove memorization).
- **CFG with one network** (accuracy at w = 0, 1, 2, 4): 100 images 8%, 93%,
  100%, 100%; 1,000 images 6%, 49%, 76%, 89%.
- **Insight:** loss keeps falling while FID gets worse; with little data the
  model copies instead of generalizing. The weaker 1,000-image model needs
  much stronger guidance to hit the class.

### Full-data EDM training (optional part)
EDM's 55.7M-parameter network, no labels, all 50,000 images, two DSMLP jobs
(3.3 h + 10.6 h on a GTX 1080 Ti), 4.77 million images seen in total.

| million images | 0 | 0.50 | 1.00 | 2.01 | 3.01 | 4.02 | 4.52 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FID (1,000 samples) | 412.4 | 105.5 | 76.5 | 67.7 | 56.2 | 52.2 | 50.2 |

- **What we saw:** FID falls at every snapshot, while loss barely moves
  (0.195 → 0.182 in run 2).
- **Insight:** with full data, FID improves steadily, unlike part 6, but the
  model is far from the pretrained one (FID 31.2), which saw 200 million
  images. A falling FID does not by itself prove the model can't memorize.
  Our GPU budget is better spent on pretrained models than on more training.

### MNIST CFG network (trained from scratch)
The lab's U-Net (1.9M parameters), 60,000 digits, 10% of labels replaced by
"no label", 30 epochs (22 min on Apple MPS); loss 0.22 → 0.060, flat after
epoch 10. Evaluated with 5,000 samples per w; the digit classifier scores 99.3%
on real test digits, and real-vs-real FID is 8.4 (a reference value, not a
lower bound).

| w | label accuracy | FID | FID_class | pixel spread |
| --- | --- | --- | --- | --- |
| 0 | 10.2% | 127.1 | – | 0.239 |
| 1 | 97.6% | 16.0 | 48.3 | 0.191 |
| 2 | 100% | 61.2 | 145.2 | 0.176 |
| 4 | 100% | 158.6 | 308.1 | 0.172 |

- **What we saw:** w = 0 is at chance, as expected. Accuracy hits 100% at w = 2,
  yet FID gets ten times worse from w = 1 to 4. Strokes become thicker and more
  uniform (`figures/mnist/cfg_grid.png`).
- **Insight:** the class is right, but guidance keeps changing attributes the
  label never asked for (thickness, uniformity). On MNIST, w = 1 is already
  near-perfect, so the only room to improve is quality and variety.
- **Caveat:** this model's unconditional mode is weak (FID 127 at w = 0; the
  course's generator.pt gave 57 with 1,000 samples, a rough comparison).

---

## What the results add up to

| Evidence | What it suggests | Still unproven |
| --- | --- | --- |
| CIFAR-10: accuracy flat after w = 1.5; FID 31.3 → 39.0 | Extra guidance stops helping the class but keeps hurting quality | Which noise levels or components cause the damage |
| MNIST: 100% accuracy from w = 2 to 4; FID 61 → 159 | Right class and realistic images can diverge sharply | Whether the cause is the guidance direction, model error, or the sampler |
| MNIST strokes thicken as w grows | Guidance changes attributes the label doesn't specify | Whether that can be blocked without losing accuracy |
| CIFAR cats keep layout but get oversaturated | Some variation survives; other aspects are over-amplified | How to tell useful changes from unwanted ones |
| Cats commit at σ ≈ 3.3 (w = 4) vs. 1.1 (w = 1) | Guidance moves when the class commits | Whether committing earlier is needed for a better final image |

**The central question:** why does "make this more like the requested class"
keep changing stroke width, color or pose after the class is already correct,
and can we stop that selectively?

**Confounds to keep in mind.**
- On MNIST, the unconditional branch is much worse than the conditional one
  (FID 127 vs. 16), so the guidance direction Δ = D_c − D_u may carry model
  error, not only class information.
- On CIFAR-10, the two branches are separately trained networks.

**Report wording to soften.**
- Real-vs-real FID (8.4) is a reference value, not a lower bound.
- A falling FID on 50,000 images does not prove the model cannot memorize.

---

## Quarter 2 proposal: guide toward the condition while protecting what it leaves open

### Why not just noise-dependent schedules
The original plan compared constant, early-heavy (3 − 2τ) and late-heavy (1 + 2τ)
schedules. That alone is not a new contribution, and the obvious extensions are
already published:

| Idea | Existing work | What it means for us |
| --- | --- | --- |
| Guide only in the middle noise levels | Limited-interval guidance (Kynkäänniemi et al., NeurIPS 2024) | Required baseline |
| A different schedule per sample | Annealing guidance (SIGGRAPH Asia 2025) | A small scheduler on D_c − D_u is not enough |
| Adapt w from online quality/alignment feedback | Dynamic CFG (ICLR 2026) | "Use confidence to set w" is crowded |
| Learn state-dependent schedules from a distribution objective | Adversarial schedule learning (2026 preprint) | Learning the schedule is not itself new |
| Project out oversaturating components | APG (ICLR 2025) | Simple parallel/orthogonal split exists |
| Separate semantic change from structural correction | SCALE (ECCV 2026) | A generic "semantic projection" needs a sharper difference |
| Theory of early mode loss and late contraction | Stage-wise dynamics; "Towards Understanding the Mechanisms of CFG" | Build on these; don't re-claim "guidance reduces diversity" |
| Model error in the guidance direction; numerical error | Autoguidance, CFG-OEC; ERK-Guid | Don't assume error-based guidance is new |

### Research question
**Can guidance improve target adherence while preserving the distribution of
attributes the target does not specify?**

- MNIST: requested = digit identity; unspecified = stroke thickness, slant,
  position, style.
- Natural images: requested = object class; possibly unspecified = viewpoint,
  background, lighting, pose.

What counts as unspecified depends on the condition: color is free for "a car"
but specified for "a red car".

### Method sketch
Write CFG around the conditional prediction:

$$D_{\text{CFG}} = D_c + (w-1)\,\Delta, \qquad \Delta = D_c - D_u .$$

Ordinary CFG scales every part of Δ by the same number. Hypothesis: that ties
the gain in class adherence to unwanted changes in unspecified attributes.
Replace it with a selective correction:

$$D_{\text{new}} = D_c + \delta, \qquad
\min_\delta \|\delta - \alpha\Delta\|^2
\ \text{ s.t. }\ \|J_n\delta\| \le \epsilon,\ \ J_a\delta \ge m,$$

where J_a is the local sensitivity of the requested attribute (e.g. the digit
classifier's logit) and J_n that of protected attributes (thickness, slant). If
the constraints conflict, use a soft penalty or a smaller margin m. If it works,
learn an offline low-rank approximation so sampling doesn't need Jacobians.

The contribution must be how the protected directions are identified (from
controlled, condition-preserving variation) and evidence of what they preserve,
not "we use a projection". Unlike APG and SCALE, we don't assume "orthogonal"
means "irrelevant".

### First decisive experiment (MNIST)
Build an MNIST variant where digit identity is independent of stroke width and
slant. Compare, at matched accuracy:
1. Plain conditional sampling (w = 1)
2. Tuned constant CFG
3. Tuned limited-interval CFG
4. APG
5. The constrained correction

Measure digit accuracy and the full within-digit distributions of thickness and
slant (not just their means). If APG or interval guidance already preserves
them, the extra machinery isn't justified and we pivot. The first goal is to
**show a failure that tuned interval CFG and APG do not fix**.

### Discovery phase (before committing to a method)

| Experiment | What it settles | Decision rule |
| --- | --- | --- |
| Reproduce tuned interval CFG and APG | Do known methods explain most of the gain? | If yes, move beyond scalar schedules |
| Measure identity, thickness, slant, saturation, feature diversity separately | Can unwanted changes be isolated? | Pursue attribute protection only if the split is reproducible |
| Branch trajectories: extra guidance vs. w = 1 over contiguous σ intervals | When guidance causes final benefit or damage | Don't infer cause from intermediate images |
| 18-step vs. finer sampling grids | Are artifacts numerical error? | Claims must hold with accurate integration |
| Shared-network vs. separate-network CFG | Does branch quality mismatch drive the effect? | Treat model error separately |
| Synthetic class-conditional mixtures with known modes | Do aggregate metrics hide mode loss? | Pursue backup 1 if a clear failure appears |

### Backup directions
1. **Protect rare valid modes within a class.** Guidance can reach high accuracy
   by producing only typical, frontal cats. Start from a synthetic mixture with
   known, unequal modes per class; measure which modes vanish, shift or shrink;
   test a controller that penalizes distance from a reference within-class
   distribution. Risk: overlaps with stage-wise theory, MMD guidance and
   determinant-based diverse sampling; modes are hard to define on natural images.
2. **Predict the remaining benefit of guidance** ("if we stop extra CFG now, is
   the final image still correct?"). Close prior work: Commitment Before
   Realization, online-feedback CFG, lookahead reward guidance. Use it as a
   diagnostic, not the main claim.

### What a strong final result needs
- A precise failure mechanism, e.g. CFG amplifies nuisance-sensitive directions
  after class adherence saturates.
- A method derived from that mechanism, with ablations separating "changed
  direction" from "smaller magnitude".
- Pareto curves at matched target adherence, not one favorable w; baselines:
  tuned constant CFG, interval CFG, APG and the closest competitors. Equal
  average w is a diagnostic, not a fair comparison by itself.
- Independent final evaluators, multiple seeds, uncertainty, larger evaluation
  sets, and full compute accounting (extra denoiser calls, offline training).
- Validation beyond MNIST/CIFAR-10 on a stronger pretrained conditional model.
- Compute: use pretrained generators and spend GPU time on interventions and
  evaluation, not on more full-data EDM training.

---

## Questions for the mentor
- Is "strengthen the condition while protecting unspecified attributes" a good
  fit for Quarter 2, or is the rare-mode direction preferable?
- Is a controlled MNIST variant (identity independent of thickness and slant) a
  convincing first test, and which stronger pretrained model should we use for
  validation?
- Can we get a larger GPU than the DSC180A workspace's 1080 Ti for evaluation runs?

---

## Run log

One entry per GPU run. Machine and package versions are recorded in
`figures/parts1_3/run_environment.json`.

### Lab parts 1–6 and EDM training run 1
- Date: November 1, 2026 (batch pod steer-run, GTX 1080 Ti); commit e4aff85
- Time: parts 1–3 in 1 min; full lab (parts 1–6) in 135 min; EDM training for
  197 min (1,205 kimg). All exit 0; training stopped at its time limit as planned.
- Problems: the DSC180A workspace only schedules small GPUs (H100/A100 requests
  are overridden). GPU quota is 1, and a pending pod already holds it. Pods stop
  after 6 hours by default (K8S_TIMEOUT_SECONDS=43200 allows 12).

### EDM training run 2
- Date: November 1–2, 2026 (batch pod edm-train2, GTX 1080 Ti, 12-hour limit);
  commit 84addec (`scripts/dsmlp_train_edm.sh 12 run2`)
- Time: started from run 1's 1,004-kimg snapshot with `--transfer`; trained
  10.6 h (76 ticks, 3,764 kimg, 10.2 s/kimg), then snapshot FID for 8 snapshots
  in 21 min. FID 76.5 → 50.2; loss 0.195 → 0.182. Results in commit 0587a04.
- Problems: the laptop's SSH session hit the 8-hour Duo limit overnight, so the
  status watcher stopped; the pod was unaffected. With `--dump=10`,
  training-state-003512.pt on DSMLP can be continued exactly with `--resume`.

### MNIST CFG network
- Date: November 1, 2026, locally on Apple MPS; 22 min training, evaluation
  with 5,000 samples per w (`python train_mnist.py train`, then `eval`).
