# Diffusion Steering Capstone — Handoff

**Last updated:** November 1, 2026

**Purpose:** Transfer the project context, assignment baseline, concepts already discussed, candidate capstone directions, and open decisions.

## 1. Project summary

The project studies **steering diffusion models**: how to influence what a model generates and understand the trade-offs among target adherence, sample quality, diversity, and (when editing an existing image) content preservation.

The course checkpoint establishes baselines on CIFAR-10 and MNIST. The capstone should build on those baselines and ask a focused research question about steering. It should not drift into a standalone comparison of U-Net and DiT unless the course explicitly requests that.

## 2. Source materials and grounding

The conversation referred to:

- The October 28 checkpoint assignment (`Oct_28_note.pptx`, also linked in the conversation).
- The October 28 steering lecture notes.
- Earlier course material on diffusion, U-Net, DiT, samplers, and latent diffusion.

The assignment and lecture were reviewed earlier in the conversation. The summary below records what was extracted during that review. Before implementation, re-open the original assignment and notebooks to verify exact filenames, API signatures, metric definitions, sample counts, and deadlines. The lecture's descriptions of RFM and noise alignment are high-level; they are not complete implementation specifications.

## 3. Assignment baseline

### 3.1 CIFAR-10: pretrained EDM and CFG

**Dataset:** CIFAR-10, 10 classes, 32×32 RGB images. The earlier assignment review described 50,000 training images.

**Provided models:**

- A pretrained unconditional EDM model.
- A pretrained class-conditional EDM model.

For this experiment, the assignment uses separate unconditional and conditional pretrained models. This differs from the common single-network CFG setup trained with condition/label dropout.

**EDM denoiser CFG formula:**

$$
D_{\mathrm{guided}}(x;\sigma,c) = D_{\mathrm{uncond}}(x;\sigma) + w\left[D_{\mathrm{cond}}(x;\sigma,c) - D_{\mathrm{uncond}}(x;\sigma)\right]
$$

Guidance values reported in the assignment:

$$
w \in \{0, 1, 1.5, 2, 3, 4\}
$$

Under this convention:

- $w = 0$: unconditional prediction.
- $w = 1$: ordinary conditional prediction.
- $w > 1$: amplify the conditional direction relative to the unconditional prediction.

**Evaluation described in the assignment:**

- Generate 1,000 images per guidance setting.
- Label accuracy: does the generated image get classified as the requested class?
- FID: compare real and generated image feature distributions.
- `FID_class`: class-aware/feature-based FID as implemented in the supplied notebook. Verify its exact feature extractor and definition from the notebook; do not assume more than the source supports.
- Visualize sampling trajectories, including noisy states and denoiser estimates at noise levels.

### 3.2 CIFAR-10: limited-data EDM training

Train EDM models using:

- 100 CIFAR-10 examples.
- 1,000 CIFAR-10 examples.
- 10% label dropout, according to the assignment summary.

Save snapshots/checkpoints and compare:

- FID across training snapshots.
- Generated images against nearest training images.
- CFG-generated image grids.

The assignment warns that small training sets can lead to memorization. Nearest-neighbor comparisons are therefore important for examining whether generated samples resemble training examples too closely.

### 3.3 MNIST: train a CFG-capable U-Net

**Dataset/model:**

- MNIST, 28×28 grayscale handwritten digits, 10 classes.
- Train a U-Net from scratch.
- The earlier assignment review described 60,000 training images.

**Condition dropout:**

- Class labels 0–9.
- Label 10 represents the null/unconditional condition.
- Randomly drop the condition 10% of the time, according to the assignment summary.

**Guidance values:**

$$
w \in \{0, 1, 2, 4\}
$$

**Evaluation described:**

- Label accuracy using the provided frozen digit classifier.
- FID computed using that classifier's features.

> **Important:** the frozen classifier is an evaluation tool for recognizing generated digits and extracting features. Its presence does **not** mean classifier guidance is being implemented. Classifier guidance would use the classifier's gradient with respect to the current noisy image/latent during sampling.

### 3.4 Checkpoint purpose

The checkpoint is primarily a baseline/reproduction exercise:

1. Run pretrained CIFAR-10 unconditional/conditional EDM and CFG.
2. Train limited-data CIFAR-10 EDM models and inspect quality/memorization.
3. Train a conditional-dropout MNIST U-Net and sample with CFG.
4. Produce quantitative metrics and qualitative sample/trajectory visualizations.

The prior conversation mentioned a November 6, 2026 checkpoint deadline. Verify this against the original course instructions before relying on it.

## 4. What "steering" means here

Steering is the broader problem of influencing a diffusion trajectory so that the generated sample satisfies a target.

The lecture distinguishes interventions at different points:

| Approach | Intervention | Generator retraining? |
| --- | --- | --- |
| Conditional generation | Give the denoiser a condition during training/inference | Normally requires a condition-trained model |
| Classifier-free guidance (CFG) | Combine conditional and unconditional denoiser predictions during sampling | Requires a CFG-capable model, commonly trained with condition dropout |
| Classifier guidance | Add a classifier input-gradient to the score during sampling | No generator retraining; requires a suitable noisy-image classifier |
| RFM steering | Lecture describes modifying internal denoiser features to steer a frozen model | Intended as frozen-model steering |
| Noise alignment | Lecture describes correcting early predictions during sampling | Intended as frozen-model steering |

### CFG

For a noise-predicting model:

$$
\epsilon_{\mathrm{guided}} = \epsilon_{\mathrm{uncond}} + w\left(\epsilon_{\mathrm{cond}} - \epsilon_{\mathrm{uncond}}\right)
$$

For the assignment's EDM denoiser, use the denoiser-output form:

$$
D_{\mathrm{guided}} = D_{\mathrm{uncond}} + w\left(D_{\mathrm{cond}} - D_{\mathrm{uncond}}\right)
$$

Do not interchange noise-prediction and EDM denoiser formulas without checking the model parameterization.

### Classifier guidance: $p$ and $s$

- $p_t(x)$: probability density of noisy samples at time/noise level $t$.
- $p_\phi(c \mid x_t, t)$: classifier probability of class $c$, given the noisy sample and noise level.
- $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$: score, the gradient of log density with respect to the current noisy input.

Bayes' rule gives:

$$
\nabla_x \log p_t(x \mid c) = \nabla_x \log p_t(x) + \nabla_x \log p_t(c \mid x)
$$

Thus classifier guidance adds a classifier gradient to the unconditional score:

$$
s_{\mathrm{guided}} = s_{\mathrm{uncond}} + w\,\nabla_x \log p_\phi(c \mid x_t, t)
$$

This differs from CFG, which combines two denoiser predictions rather than differentiating a separate classifier.

### RFM and noise alignment caveat

The October 28 lecture presents RFM and noise alignment as frozen-model steering approaches:

- RFM is described at the denoiser-feature level.
- Noise alignment is described at the sampler/prediction level.

The earlier review associated the lecture's reference with Wang, Belkin, and Wang (2026). **Verify the exact bibliographic reference and obtain the paper/code before promising an implementation.** The slides alone do not establish all algorithmic details.

## 5. Evaluation: what should be compared?

The lecture frames steering evaluation around:

1. **Target accuracy:** does the sample satisfy the requested class/attribute?
2. **Diversity:** are samples varied, or do they collapse toward similar outputs?
3. **Sample quality:** do generated images resemble the real data distribution?
4. **Content preservation:** if editing an existing image, is content outside the intended change preserved?

The checkpoint already includes label accuracy, FID, and `FID_class` in its described experiments. Possible additional capstone metrics:

- **Diversity:** pairwise LPIPS or feature-space distances among generated samples.
- **Memorization:** nearest-neighbor distance between generated samples and training images, especially for the 100- and 1,000-example experiments.
- **Content preservation:** similarity of non-target features before/after an edit, only if the project includes editing an input image.

Do not force a content-preservation metric onto ordinary class generation when there is no source image to preserve.

## 6. Candidate capstone directions

### A — Adaptive CFG across the sampling trajectory

**Working title:** *Beyond Fixed Classifier-Free Guidance: Adaptive Steering Across the Diffusion Trajectory.*

**Research question:** Can noise-dependent guidance schedules improve target-class accuracy while preserving more diversity and sample quality than constant CFG?

The checkpoint tests fixed $w$. An extension is to make $w$ depend on normalized sampling progress $\tau$, where $\tau = 0$ is the high-noise start and $\tau = 1$ is the final low-noise state.

Candidate schedules:

- Constant baseline: $w(\tau) = 2$
- Early-heavy: $w(\tau) = 3 - 2\tau$
- Late-heavy: $w(\tau) = 1 + 2\tau$

Both variable schedules average to 2 over $[0, 1]$. These are proposed experimental schedules, not known optimal schedules.

**Setup:**

- Use pretrained CIFAR-10 unconditional and conditional EDM models.
- Keep both networks frozen.
- Use identical initial noise seeds across schedules.
- Keep sampler, noise schedule, number of steps, and requested classes fixed.
- Compare constant and adaptive guidance.
- Track label accuracy, FID, `FID_class`, diversity, and trajectory visualizations.

**Why attractive:**

- Direct extension of the assigned CFG baseline.
- No extra model training.
- Clear sampler-level intervention.
- Feasible and easy to explain.

**Risks:**

- The proposed schedules may not outperform fixed CFG.
- Normalize $\tau$ consistently with the actual sampler's noise-level ordering.
- Compare schedules with matched seeds and sampling budgets.
- Check related work before making novelty claims.

### B — CFG, diversity, and memorization under limited data

**Working title:** *The Trade-off Between Guidance, Diversity, and Memorization in Data-Limited Diffusion Models.*

**Research question:** Does stronger CFG exacerbate memorization or reduce diversity when a diffusion model is trained on very small datasets?

**Setup:**

- Use the assigned CIFAR-10 models trained on 100 and 1,000 examples.
- Use the assigned 10% label dropout.
- Evaluate multiple checkpoints and guidance scales.
- Optionally compare against a full-data reference if resources permit.

**Measurements:**

- Label accuracy.
- FID and `FID_class` as implemented in the assignment.
- Nearest-neighbor distance to training examples.
- Diversity between generated samples.
- Visual inspection of generated images and nearest training neighbors.

**Why attractive:**

- Uses a required assignment experiment.
- Connects CFG, data scarcity, memorization, and generalization.

**Risks:**

- Training/checkpoint management costs more time and compute.
- FID can be noisy with small sample sets; report sample counts and uncertainty.
- A close nearest neighbor is evidence to investigate, not automatic proof of memorization.

### C — Example-based steering of a frozen diffusion model

**Working title:** *Example-Based Steering of Frozen Diffusion Models: Feature-Level versus Sampling-Level Interventions.*

**Research question:** Can a pretrained unconditional model be steered toward a target class using a small set of examples, without retraining the generator?

**Potential comparisons:**

- Unconditional EDM.
- Conditional EDM / CFG baseline.
- RFM feature-level steering.
- Noise-alignment sampling-level steering.

**Setup:**

- Start with a target CIFAR-10 class, such as cats.
- Use a small, explicitly defined reference set of target-class examples.
- Keep the base unconditional generator frozen.
- Compare target accuracy, FID/feature quality, diversity, and (where appropriate) similarity to reference/training images.

**Why attractive:**

- Closely aligned with the lecture's frozen-model steering theme.
- Compares different intervention points: denoiser features versus sampler predictions.

**Risks:**

- Highest implementation risk.
- Need the full cited paper and preferably official code.
- Slides provide only a high-level description; do not invent missing algorithmic steps.
- Need fair baselines and controls for reference examples.

## 7. Recommended strategy

**Recommended primary direction: A — Adaptive CFG**, with **C — frozen-model steering** as an optional extension if the paper/code is accessible and time allows.

Rationale:

- First complete the course baselines.
- Use pretrained CIFAR-10 EDM models to isolate sampler-level guidance changes.
- Adaptive CFG is a manageable extension that does not require training a new model.
- If time permits, compare it with RFM/noise alignment as a more ambitious frozen-model intervention.
- Keep Direction B as a secondary analysis using the limited-data training work already required.

### Suggested sequence

**Stage 1 — Reproduce the assignment**

1. Run CIFAR-10 pretrained unconditional/conditional EDM.
2. Reproduce the assigned fixed CFG scales.
3. Generate required sample counts and metrics.
4. Visualize trajectories.
5. Train 100- and 1,000-example CIFAR-10 models and inspect nearest neighbors.
6. Train the MNIST U-Net with null-label dropout and reproduce CFG results.

**Stage 2 — Capstone experiment**

1. Choose one target class or a balanced set of classes.
2. Establish a fixed-CFG baseline.
3. Implement early-heavy and late-heavy guidance schedules.
4. Use matched initial seeds and identical sampler settings.
5. Compare accuracy, FID, `FID_class`, diversity, and trajectories.
6. Run ablations if results warrant them.
7. Add RFM/noise alignment only if implementation is feasible.

### Experimental controls

- Same pretrained checkpoints.
- Same sampler and noise schedule.
- Same number of sampling steps.
- Same initial noise seeds for paired comparisons.
- Same class distribution and number of generated samples.
- Same evaluation classifier/feature extractor.
- Same preprocessing and metric implementation.
- Record guidance schedules and code/config versions.

## 8. Diffusion concepts already discussed

### Forward and reverse processes

A DDPM forward step adds noise:

$$
q(x_k \mid x_{k-1}) = \mathcal{N}\left(\sqrt{1-\beta_k}\,x_{k-1},\ \beta_k I\right)
$$

A common forward marginal is:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

The reverse model estimates a distribution for a cleaner previous state:

$$
p_\theta(x_{k-1} \mid x_k) = \mathcal{N}\left(\mu_\theta(x_k, k),\ \Sigma_\theta(x_k, k)\right)
$$

Sampling can be written:

$$
x_{k-1} = \mu_\theta(x_k, k) + \Sigma_\theta(x_k, k)^{1/2} z
$$

The forward process adds noise to training data. The reverse sampler starts from noise and repeatedly updates the current state toward a cleaner sample. The sampler does not feed a less-noisy state into the forward process; it uses the trained denoiser to run the reverse generative process.

### $x_t$, $\hat{x}_0$, and the denoiser

- $x_t$: current noisy state at noise/time $t$.
- $\hat{x}_0$: model's estimate of the clean data corresponding to that noisy state.
- Denoiser: learned function mapping noisy input (and noise level/condition) to a prediction such as clean image, noise, velocity, or score.

Toy example:

$$
x_t = 0.6(8) + 0.4(-2) = 4
$$

Here 4 is the noisy/current interpolated state in the toy example; it is not the denoiser. The denoiser is the learned function that takes the current state and noise level and predicts a target. The coefficients' exact interpretation depends on the corruption path/parameterization.

### Prediction parameterizations

Common network targets:

- Clean-image prediction: predict $x_0$ directly.
- Noise prediction: predict $\epsilon$.
- Velocity prediction: predict a path-dependent $v$.
- Score prediction: predict $\nabla_{x_t} \log p_t(x_t)$.

The sampler converts the network output into the quantity required for its update. Keep the parameterization consistent.

### DDPM, DDIM, ODE, SDE

- **DDPM:** discrete-time diffusion formulation with a stochastic reverse Markov chain.
- **DDIM:** sampling procedure using the same trained diffusion model, often allowing fewer steps and deterministic sampling under a particular setting.
- **SDE:** stochastic differential equation describing a continuous-time noisy process.
- **ODE:** ordinary differential equation describing deterministic evolution; diffusion models have probability-flow ODE formulations related to their SDEs.
- **Sampler:** numerical procedure that calls the denoiser and updates the current state. DDPM/DDIM/SDE/ODE are process formulations or sampling choices, not denoiser architectures.

### U-Net and DiT

Both can serve as the denoiser:

- **U-Net:** convolutional encoder–bottleneck–decoder, downsampling/upsampling, and skip connections. Time and condition embeddings can enter its blocks.
- **DiT:** transformer-based denoiser; converts image/latent patches into tokens and processes them using transformer blocks and attention. Timestep and condition can modulate blocks or enter through attention.
- Neither replaces the sampler or diffusion objective.
- The sampler repeatedly calls whichever denoiser architecture is used.

U-Net/DiT comparison is background, not the assignment's main comparison.

### VAE and latent diffusion

- **VAE:** variational autoencoder with encoder and decoder. The encoder maps an image to a learned latent distribution/representation; the decoder reconstructs an image from a latent.
- **Latent space:** learned compressed representation, not simply a smaller RGB image.
- **Latent diffusion:** runs diffusion on VAE latents rather than directly on pixels.
- In Stable Diffusion v1, a 512×512 RGB image is represented by a 64×64×4 latent. The denoiser operates on noisy latents; the VAE decoder is used after sampling to produce the final pixel image.
- The VAE encoder/decoder, diffusion denoiser, and sampler have distinct roles. The VAE decoder is not used after every denoising step.

These are useful background concepts but Stable Diffusion/VAE is not the assignment's primary dataset or required baseline.

### FID

Fréchet Inception Distance compares feature distributions of real and generated images, commonly using Gaussian approximations to feature embeddings:

$$
\mathrm{FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

Lower FID generally indicates closer feature-distribution statistics. FID is sensitive to sample count, preprocessing, and feature extractor. Use the assignment's prescribed implementation for course comparisons. FID is not the same as class accuracy or diversity.

## 9. Text-to-image conditioning

For text-conditioned diffusion:

- A text encoder maps a prompt into text embeddings.
- The denoiser receives the noisy image/latent and text conditioning.
- Cross-attention or another conditioning mechanism lets visual features use the text representation.
- CFG can combine conditional and unconditional predictions to strengthen prompt adherence.

This is useful conceptual context, but the assignment described here is class-conditioned CIFAR-10/MNIST, not a Stable Diffusion text-to-image experiment.

## 10. Open questions to verify

Before finalizing the proposal, verify:

1. Exact assignment wording, deadline, notebook names, commands, and output formats.
2. Exact pretrained EDM checkpoint variants and model parameterization.
3. Exact `FID_class` implementation and feature extractor.
4. Whether exactly 1,000 images are required per CIFAR-10 CFG setting and how classes are balanced.
5. Exact sampler/noise schedule used by the assignment.
6. Exact MNIST FID implementation and frozen classifier role.
7. Full bibliographic details for the lecture's RFM/noise-alignment reference.
8. Whether official RFM/noise-alignment code exists and is compatible with the supplied EDM implementation.
9. Whether the capstone may use external datasets/models or must stay within course-provided materials.

Do not silently fill these gaps. Check the source notebook/assignment or ask course staff.

## 11. Current proposal framing

**Problem:** Fixed classifier-free guidance is a simple way to steer conditional diffusion models, but a single guidance scale is applied throughout the sampling trajectory.

**Research question:** Can guidance that varies with noise level improve the trade-off between target-class adherence, sample quality, and diversity compared with fixed CFG?

**Baseline:** Pretrained CIFAR-10 unconditional and conditional EDM models, with the assignment's fixed guidance scales.

**Proposed method:** Compare constant guidance with matched-average early-heavy and late-heavy guidance schedules.

**Evaluation:** Target-class accuracy, FID, `FID_class`, sample diversity, and sampling-trajectory visualizations.

**Controls:** Frozen checkpoints, matched seeds, same sampler/noise schedule, same number of steps, same class distribution, and same metric implementation.

**Optional extension:** Compare adaptive CFG with example-based frozen-model steering (RFM or noise alignment), only after confirming the algorithm and implementation from the primary paper or official code.

**Expected contribution:** An empirical characterization of how the timing of guidance affects class adherence, image quality, and diversity. Do not assume in advance that adaptive guidance will win.

## 12. Short handoff for the next assistant

The user is learning diffusion models while developing a course capstone on steering. Explain concepts from first principles and connect equations to the actual project.

Assignment baseline:

- **CIFAR-10 pretrained EDM:** unconditional + class-conditional models, CFG scales 0/1/1.5/2/3/4, label accuracy/FID/`FID_class`/trajectory visualization.
- **CIFAR-10 limited-data EDM training:** 100 and 1,000 images, 10% label dropout, snapshots and nearest-neighbor/memorization analysis.
- **MNIST U-Net training:** 28×28 grayscale, labels 0–9 plus null label 10, 10% dropout, CFG scales 0/1/2/4, frozen-classifier accuracy and feature FID.

The October 28 lecture expands the topic to steering a frozen model, including high-level RFM feature steering and noise-alignment sampler steering. Implementation details and the exact paper reference need verification.

Best current capstone recommendation: **adaptive CFG across the sampling trajectory**, comparing constant, early-heavy, and late-heavy schedules under controlled conditions. An ambitious optional extension is RFM/noise alignment. Another viable direction is studying whether CFG exacerbates memorization in 100/1,000-example models.

Avoid redirecting the main project into U-Net-vs-DiT or VAE/Stable Diffusion comparisons; those are background topics, not the core assignment.
