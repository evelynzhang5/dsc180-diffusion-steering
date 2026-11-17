# Q2 round 27: is the patch-to-patch blur pattern real, or a measurement effect?

**Question.** Rounds 24–26 found a stable per-patch pattern in delivered blur:
- astronaut and chelsea read about 15 % low;
- coins and night lights read about 10–20 % high.

Steering PISA's attention did not change that pattern. Round 24 already warned that the disk-fit estimator reads known Gaussian blur 3–11 % differently across these same patches. If the pattern is a measurement effect, downstream steering has no target there. If it is real, it does.

**Status.** Run Nov 17 (pod `q2-kernel`, 2 min, no new BokehDiff generations). It re-measures round 25's saved baseline outputs: fp32 PISA + original gate + radius curve; 12 held-out patches × 3 positions × 7 targets, 3–15.5 px. Plan and rules were committed (79571c7) before the run.

**Frozen outcome: V.** The shape-agnostic estimator failed its validation at small kernels, so the rule allows no conclusion.

**The step that does not depend on that estimator points clearly to a real model effect.** Step B uses only the validated disk-fit estimator: BokehDiff's own median kernel, applied identically to every patch, reads almost the same everywhere (across-patch CV 0.5–1.6 %, against 10 % for the real outputs), and does not reproduce the real ranking (Spearman −0.19). **The per-patch pattern is therefore not a measurement artefact: BokehDiff renders differently shaped blur kernels on different content.**

## Method (`check27.py`)

**A second, shape-agnostic estimator.** Per cell, the blur kernel is a non-negative radial profile, rings of 1 px width up to 34 px, fitted by bounded least squares on luminance with a free offset. Its size is the second moment r2, which is valid for any kernel shape: a disk of radius R has r2 = R/√2, a Gaussian of σ has r2 = σ√2.

| step | what |
|---|---|
| V, validation | known disks (4, 8, 12 px) and Gaussians (σ 2, 4, 6) applied to these exact patches, degraded as before. Pass: median r2 error ≤ 10 % per kernel **and** across-patch CV ≤ 3 % |
| A, real outputs | per-patch pattern (each patch's measure ÷ the median of all patches at that target) for the disk fit and for r2; across-patch CV of each |
| B, measurement only | the median real kernel profile (targets 5, 8, 11.6 px) is applied identically to every patch and read with the disk-fit estimator. Does measurement alone reproduce the real per-patch pattern? |

## Rules (fixed now)

| outcome | rule | meaning for steering |
|---|---|---|
| V | validation fails | no conclusion |
| K1 | median CV with r2 ≤ ½ × with the disk fit, **and** Spearman(real disk pattern, identical-kernel disk pattern) ≥ 0.7 | the pattern is mostly measurement; downstream steering has no target in it |
| K2 | median CV with r2 ≥ 0.8 × with the disk fit, **and** the identical-kernel CV ≤ ½ × the real disk-fit CV | the pattern is a real model effect; downstream steering has a target |
| K3 | otherwise | mixed |

**Also reported:** the radial kernel profiles of the high-contrast patches (coins, night lights) vs typical ones (gravel, chelsea), i.e. whether BokehDiff renders a disk-, Gaussian- or ring-like (bokeh) kernel there.

## Results

Files:
- `results/kernel_check.json`;
- `results/kernel_rows.json`;
- `results/kernel_check.png`: kernel profiles at target 8 px, and the per-patch patterns.

### V: validation of the shape-agnostic estimator (failed)

| known kernel | disk 4 | disk 8 | disk 12 | Gaussian σ 2 | σ 4 | σ 6 |
|---|---|---|---|---|---|---|
| median error of r2 | +5.0 % | +1.4 % | +0.9 % | +7.8 % | +0.6 % | −0.1 % |
| across-patch CV | **4.9 %** | 1.9 % | 0.7 % | **6.9 %** | 1.9 % | 1.0 % |

It fails the ≤ 3 % CV bar only for the two smallest kernels (r2 ≈ 2.8 px). From disk 8 / σ 4 upwards (r2 ≥ 5.7 px), it is accurate and consistent across patches. By the frozen rule the outcome is **V**.

### A: real outputs (reported; the r2 column is only trustworthy for larger kernels)

| target (px) | 3 | 4 | 5 | 6 | 8 | 11.6 | 15.5 |
|---|---|---|---|---|---|---|---|
| across-patch CV, disk fit | 0.117 | 0.097 | 0.103 | 0.109 | 0.103 | 0.100 | 0.108 |
| across-patch CV, second moment | 0.45 | 0.46 | 0.42 | 0.38 | **0.30** | **0.20** | **0.14** |

**Per-patch ratio** (each patch ÷ the median of all patches at the same target):

| measure | chelsea_1 | chelsea_0 | astronaut_0 | rocket_0 | rocket_1 | gravel_1 | text_1 | text_0 | coins_0 | coins_1 |
|---|---|---|---|---|---|---|---|---|---|---|
| disk fit | 0.83 | 0.87 | 0.87 | 0.99 | 1.06 | 1.01 | 1.01 | 1.04 | 1.14 | 1.09 |
| second moment | 0.61 | 0.73 | 0.84 | 0.74 | 0.93 | 1.17 | 1.25 | 1.41 | 1.63 | 1.84 |

The two measures rank patches alike (Spearman 0.85). The kernel-shape-agnostic size *amplifies* the pattern rather than removing it. This holds even at targets 8–15.5 px, where the estimator is validated (CV 0.14–0.30, against the disk fit's 0.10).

### B: measurement-only reproduction (uses only the validated disk-fit estimator)

BokehDiff's median kernel profile (at 5, 8 and 11.6 px) was applied identically to all 12 patches:
- the disk-fit readings vary by CV 1.6 / 1.2 / 0.5 %; the real outputs vary by 10 %;
- the per-patch ratios are 0.98–1.01;
- Spearman with the real pattern is −0.19.

**The measurement does not create the pattern.**

### Kernel shapes (target 8 px, median over positions)

| patch | radius holding half the mass | mass beyond 15 px | peak ring |
|---|---|---|---|
| astronaut_0/1, chelsea_0/1 | 5 px | 0.00–0.01 | 4–5 px |
| rocket_0 / rocket_1 | 6 / 5 px | 0.00 | 7 / 3 px |
| gravel_0/1, text_0/1 | 6–7 px | 0.02–0.09 | 4–6 px |
| **coins_0 / coins_1** | **8 / 7 px** | **0.11 / 0.18** | **8 / 7 px** |

- On soft, low-contrast content (astronaut, chelsea), BokehDiff renders a compact, centre-weighted kernel.
- On high-contrast content (coins, some text), it renders a disk-edge-weighted kernel, peaked near the rim, with 10–18 % of the mass beyond 15 px. That looks like a halo or bokeh-like spread.

## Interpretation

- **Frozen:** V (no conclusion from the shape-agnostic estimator).
- **Supported by the validated parts:** the patch-to-patch pattern is a real property of BokehDiff's output, not of our measurement. The model renders content-dependent kernel shapes: compact on soft texture, rim-weighted with long tails on high-contrast texture.
- **Correction to round 24's M flag.** M used Gaussian test kernels, which the disk fit reads inconsistently across patches. With BokehDiff's *actual* kernel shape, the disk fit is consistent (CV ≤ 1.6 %). So M overstated the measurement share.
- **For steering:**
  - this is a real, repeatable, content-dependent failure of a specified photographic control: the same requested blur yields differently shaped kernels on different content;
  - it is **not** set by the amount of attention PISA spreads (round 26);
  - the next step is to localize where kernel shape is decided: the UNet layers after PISA, or the fine-tuned VAE decoder. That is where a steering intervention would act.

**Not shown:** whether rim-weighted kernels on high-contrast content are physically right (real lens bokeh is rim-weighted for bright highlights), which would need a reference render; real photos; boundaries.
