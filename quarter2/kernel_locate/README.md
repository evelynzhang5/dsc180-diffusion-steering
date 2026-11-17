# Q2 round 28: where is BokehDiff's content-dependent kernel shape decided, and is it physical bokeh?

**Question.** Round 27 found that the same blur request gives differently shaped kernels on different content: compact on soft texture, rim-weighted with long tails on high-contrast texture. Two questions follow:
1. **Physics.** Real lens blur acts in *linear light*; BokehDiff's own dataset code applies gamma 1/2.2 to linear renders. A linear-light disk makes bright, high-contrast content spread wider in sRGB. Is the pattern what physical bokeh would produce? If so, the issue is our target definition (an sRGB-space disk), not the model.
2. **Localization.** Is the pattern already in the latents (decided in the UNet), or created by BokehDiff's fine-tuned VAE decoder? That is where a steering intervention would act.

**Status.** Run Nov 17 (pod `q2-locate`, 6 min). The test's own checks all passed: the outputs reproduce round 25 exactly, and the linear-light estimator is valid (across-patch CV ≤ 0.2 %). Plan and rules were committed (cca099d) before the run.
- **Physics: NOTPHYS.** Linear-light lens blur does not explain the per-patch pattern.
- **Localization: UNET.** The pattern is in the latents. Note that **BokehDiff did not fine-tune the VAE decoder at all** (only 30 of 108 encoder tensors changed; 0 of 140 decoder tensors), so the decoder swap could not differ. The decoder-only pathway is what rules the decoder out.

## Design (`run28.py`, `analyze28.py`)

**Generations.** Round 25's baseline configuration (fp32 PISA, original gate, frozen radius curve): corrected runs on the 12 held-out patches × positions 0, 2, 4 × targets 3–15.5 px, 42 generations. They must reproduce round 25's images exactly. The final latents are captured and decoded twice:

| decode | decoder |
|---|---|
| ft | BokehDiff's fine-tuned VAE: the normal output |
| orig | the original RealVisXL V5.0 VAE |

The run also logs which VAE tensors BokehDiff changed.

**Decoder-only pathway.** Known pixel-space blurs (an sRGB-space disk and a linear-light disk, at 5, 8 and 11.6 px) are encoded with BokehDiff's encoder and decoded with both decoders.

**Estimators:**
- sRGB-space disk fit (rounds 18–27);
- a new **linear-light disk fit**: blur in linear light (gamma 2.2), convert back, per-channel gain and offset.
- Validation: known linear-light disks (5, 8, 12 px) recovered within max(1 px, 10 %) on every test patch, with across-patch CV ≤ 3 %.

## Rules (fixed now)

Decisions use targets 5–15.5 px; 3–4 px are reported only.

**Gate:**
- the ft outputs reproduce round 25 (≤ 1/255);
- the linear-light estimator passes validation.

| question | outcome | rule |
|---|---|---|
| physics | PHYS | across-patch CV of real outputs with the linear-light estimator ≤ ½ × with the sRGB estimator, **and** Spearman(real per-patch pattern, pattern of a pure pixel-space linear-light disk) ≥ 0.7. **The pattern is physical; fix the target definition, not the model** |
| physics | NOTPHYS | CV ratio ≥ 0.8 **and** Spearman < 0.4 |
| physics | PARTIAL | otherwise |
| localization | DEC | CV with the original decoder ≤ ½ × with the fine-tuned decoder: the fine-tuned decoder creates most of the pattern |
| localization | UNET | CV ratio ≥ 0.8: the pattern is already in the latents |
| localization | MIXED | otherwise |

**Also reported:**
- Spearman of the patterns from the ft and orig decoders;
- the decoder-only pathways (does a decoder turn an identical known blur into a patch-dependent one?);
- delivered/target ratios under both estimators.

## Results

Files:
- `results/locate.json`;
- `results/locate_rows.json`.

Images are in `outputs/q2_kernel_locate/`.

### What BokehDiff's VAE checkpoint changes

| part | tensors changed |
|---|---|
| encoder | 30 of 108 |
| decoder | **0 of 140** |

The fine-tuned VAE affects how the input is encoded, not how blur is drawn. Decoding the same latents with the original decoder therefore gives identical results: across-patch CV 0.103 vs 0.103, Spearman 1.00.

### Across-patch CV of the measured radius (targets 5–15.5 px; median over targets)

| set | sRGB disk fit | linear-light disk fit |
|---|---|---|
| **BokehDiff outputs** | **0.103** | **0.098** |
| pure pixel-space sRGB disk (no model) | 0.000 | 0.060 |
| pure pixel-space linear-light disk (no model) | 0.042 | 0.000 |
| sRGB disk → VAE encode → decode | 0.049 | 0.046 |
| linear-light disk → VAE encode → decode | 0.087 | 0.031 |

**Per-patch ratio (sRGB disk fit), BokehDiff vs physics:**

| patch | chelsea_1 | chelsea_0 | astronaut_0 | astronaut_1 | rocket_0 | gravel_0 | coins_1 | coins_0 |
|---|---|---|---|---|---|---|---|---|
| BokehDiff | 0.83 | 0.84 | 0.90 | 0.95 | 0.98 | 1.05 | 1.11 | 1.17 |
| linear-light disk | 0.96 | 0.98 | 1.08 | 1.09 | 0.99 | 0.98 | 1.08 | 1.07 |

Spearman of BokehDiff's per-patch pattern with each reference: pixel linear-light 0.32; VAE round trip of an sRGB disk 0.13; of a linear-light disk 0.24.

### Decisions

- **Physics: NOTPHYS** (CV ratio 0.95, Spearman 0.32).
  - Measuring in linear light barely reduces the spread.
  - Physics predicts the coins' over-read only in part (+7–8 % of BokehDiff's +11–17 %). It does not predict chelsea's under-read, and it predicts astronaut should read *high*, where BokehDiff reads it *low*.
  - So BokehDiff is not simply rendering physical bokeh.
- **Localization: UNET.**
  - The decoder is the original one, and identical latents decode identically.
  - The VAE round trip adds some patch dependence of its own (CV 0.05), but with an unrelated ranking (Spearman 0.13).
  - **BokehDiff's per-patch pattern is produced by the UNet, in the latents.**

**Accuracy is unaffected:** delivered/target in linear light is 0.96–1.04 across targets (sRGB: 1.01–1.09). The baseline calibration is fine on average; the issue is consistency.

## Interpretation

- **The failure stands.** The same requested blur gives differently sized and shaped kernels on different content (about 10 % spread; soft content compact and under, high-contrast content rim-weighted and over). It is:
  - not a measurement artefact (round 27);
  - not physical bokeh (this round);
  - not set by the decoder (this round);
  - not set by the amount of attention PISA spreads (round 26).
- **Where steering must act:** the UNet's features *after* PISA (`down_blocks.1.attentions.0`). Those are the later down, mid and up blocks of BokehDiff's single denoising step, which turn the attention spread into a latent kernel.
- **Next steering step:**
  1. A per-layer latent probe: measure kernel shape in the latent itself, and how it changes block by block after PISA.
  2. Then a per-region feature intervention at the block where the content dependence appears, tested against the calibration baseline on held-out patches.

**Not shown:**
- that a UNet-feature intervention can fix it;
- real photos;
- boundaries.
