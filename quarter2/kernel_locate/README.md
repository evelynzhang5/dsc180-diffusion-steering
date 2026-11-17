# Q2 round 28: where is BokehDiff's content-dependent kernel shape decided, and is it physical bokeh?

**Question.** Round 27 found that the same blur request gives differently shaped kernels on different content: compact on soft texture, rim-weighted with long tails on high-contrast texture. Two questions follow:
1. **Physics.** Real lens blur acts in *linear light*; BokehDiff's own dataset code applies gamma 1/2.2 to linear renders. A linear-light disk makes bright, high-contrast content spread wider in sRGB. Is the pattern what physical bokeh would produce? If so, the issue is our target definition (an sRGB-space disk), not the model.
2. **Localization.** Is the pattern already in the latents (decided in the UNet), or created by BokehDiff's fine-tuned VAE decoder? That is where a steering intervention would act.

**Status.** Plan and rules fixed before running.

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

Pending.
