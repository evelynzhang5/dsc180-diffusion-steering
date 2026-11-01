# Diffusion concepts

Study notes on the background behind the project. Not needed to reproduce anything.

> **Notation warning.** The course and the lab write $x_1$ for the clean image and $x_0$ for Gaussian noise, with a noisy image $x = x_1 + \sigma x_0$ and EDM's clean-image estimate $D(x;\sigma)$. The DDPM section below uses the standard DDPM papers' notation instead, where $x_0$ is the clean image. Don't mix the two.

## Ways to steer a diffusion model

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

For EDM's denoiser, which returns a clean-image estimate, the lab uses:

$$
D_w(x;\sigma,c) = D(x;\sigma) + w\left(D(x;\sigma,c) - D(x;\sigma)\right)
$$

$w = 0$ is unconditional, $w = 1$ is ordinary conditional, and $w > 1$ pushes further along the difference the class makes. Don't interchange the two formulas without checking the model's parameterization.

### Classifier guidance

- $p_t(x)$: density of noisy samples at noise level $t$.
- $p_\phi(c \mid x_t, t)$: classifier probability of class $c$ given the noisy sample.
- $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$: the score.

Bayes' rule gives:

$$
\nabla_x \log p_t(x \mid c) = \nabla_x \log p_t(x) + \nabla_x \log p_t(c \mid x)
$$

so classifier guidance adds a classifier gradient to the unconditional score:

$$
s_{\mathrm{guided}} = s_{\mathrm{uncond}} + w\,\nabla_x \log p_\phi(c \mid x_t, t)
$$

CFG instead combines two denoiser predictions without differentiating a classifier.

A frozen classifier used only to score generated images (as in the lab's label accuracy) is **not** classifier guidance.

## Evaluating steering

1. **Target accuracy:** does the sample show the requested class?
2. **Diversity:** are samples varied, or do they collapse toward similar outputs?
3. **Sample quality:** do generated images resemble the real data distribution?
4. **Content preservation:** only when editing an existing image.

### FID

$$
\mathrm{FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \operatorname{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

Lower is better. FID depends on sample count, preprocessing, and feature extractor, so only compare runs measured the same way. It is not the same as class accuracy or diversity.

## DDPM forward and reverse processes

Forward step:

$$
q(x_k \mid x_{k-1}) = \mathcal{N}\left(\sqrt{1-\beta_k}\,x_{k-1},\ \beta_k I\right)
$$

Forward marginal:

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Reverse model and sampling step:

$$
p_\theta(x_{k-1} \mid x_k) = \mathcal{N}\left(\mu_\theta(x_k, k),\ \Sigma_\theta(x_k, k)\right), \qquad
x_{k-1} = \mu_\theta(x_k, k) + \Sigma_\theta(x_k, k)^{1/2} z
$$

The forward process adds noise to training data. The reverse sampler starts from noise and repeatedly moves toward a cleaner sample using the trained denoiser.

## The denoiser and its outputs

- $x_t$: the current noisy state.
- $\hat{x}_0$ (DDPM notation; the lab's $D(x;\sigma)$): the model's estimate of the clean image.
- Denoiser: the learned function from noisy input (plus noise level and condition) to a prediction.

Common prediction targets: clean image, noise $\epsilon$, velocity $v$, or score $\nabla_{x_t} \log p_t(x_t)$. The sampler converts whichever one the network outputs into what its update needs.

Toy example: $x_t = 0.6(8) + 0.4(-2) = 4$. Here 4 is the noisy state, not the denoiser.

## Samplers: DDPM, DDIM, SDE, ODE

- **DDPM:** discrete-time formulation with a stochastic reverse Markov chain.
- **DDIM:** samples from the same trained model, often in fewer steps and deterministically.
- **SDE / ODE:** continuous-time stochastic and deterministic (probability-flow) formulations.
- **Sampler:** the numerical procedure that calls the denoiser and updates the state. The lab uses EDM's deterministic Heun sampler, 18 noise levels from σ = 80 to 0.

These are sampling choices, not denoiser architectures.

## U-Net and DiT

Both can be the denoiser.

- **U-Net:** convolutional encoder–bottleneck–decoder with skip connections; time and condition embeddings enter its blocks. EDM's CIFAR-10 network is a U-Net.
- **DiT:** transformer over image/latent patches; timestep and condition modulate blocks or enter through attention.

Neither replaces the sampler or the training objective.

## VAE and latent diffusion

- **VAE:** encoder maps an image to a learned latent; decoder reconstructs an image from it.
- **Latent diffusion:** runs diffusion on VAE latents instead of pixels. In Stable Diffusion v1 a 512×512 RGB image becomes a 64×64×4 latent, and the decoder runs once after sampling.

## Text-to-image conditioning

A text encoder turns the prompt into embeddings; the denoiser attends to them (usually by cross-attention); CFG strengthens prompt adherence. Background only: this project is class-conditioned CIFAR-10/MNIST.
