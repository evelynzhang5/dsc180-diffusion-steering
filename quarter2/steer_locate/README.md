# Q2 round 29: where can a small feature edit correct BokehDiff's output?

**Question.** Rounds 26–28 showed:
- the same blur request gives different blur on different content (soft content under-blurs, high-contrast content over-blurs);
- this is produced in the UNet after the blur attention (PISA), not by the attention spread or the decoder.

This round asks a practical steering question instead of a mechanistic one: **at which location, if any, can one small, restricted edit move the output closer to the requested blur, beyond what changing the blur request itself achieves?** It does not try to prove where the error begins.

**This is privileged feasibility testing.** Every edit is chosen per case using the known target. A positive result would not yet be a deployable steering method.

**Status.** Plan and rules fixed before running (this commit).

## Design (`run29.py`, `metrics29.py`, `analyze29.py`)

**Fixed configuration.** Round 25's baseline:
- full-precision blur attention, original gate, frozen radius curve;
- same image (held-out group 0, position 0), geometry and random state.

BokehDiff runs the UNet once, untiled, at 512² (checked in the pipeline code), so there is no sampling-time axis.

**Eight cases.** Four patches × targets 8 and 11.625 px. These cover under-blurring (chelsea_0), about right (astronaut_0), and over-blurring (gravel_0, coins_0), and avoid the small-kernel range where the estimator is weak.

**Reference.** The rendered target: an sRGB-space disk blur of the input at the target radius.

**Edit family.** One location at a time, applied only inside the selected patch's cell:

  d = G_s ∗ h − h,  h′ = h + α · M · d · rms_M(h) / rms_M(d)

- h is the feature map;
- G_s is a per-channel Gaussian;
- M is the patch's cell, area-downsampled to the feature grid;
- the residual is rescaled to the region's feature RMS, so α has the same meaning at every location.

Positive α smooths the features and negative α sharpens them. What either does to the image is measured, not assumed.

**Search grid:**

| | values |
|---|---|
| Gaussian σ | 16, 32 image pixels |
| α | −0.10, −0.03, +0.03, +0.10 |

**Locations** (hooks; actual shapes are logged):

| id | location |
|---|---|
| pre_pisa | input to `unet.down_blocks.1.attentions.0` (the block containing PISA) |
| post_pisa | output of that block |
| mid | output of `unet.mid_block` |
| up_late | output of the last up block (`unet.up_blocks.2`) |
| latent | final denoised latent, input to `vae.decode` |
| dec_early | output of `vae.decoder.mid_block` |

That gives 6 × 8 cases × 2 σ × 4 α = **384** edited generations.

**Input-only correction (the strong comparison).**
- Only that patch's blur request changes.
- It is searched against the same rendered target, with the same error: factors 0.3–2.0 × the baseline request in steps of 0.1, then 3 halving refinements around the best (24 generations per case).

## Measurements

- **Primary error:** mean absolute difference from the rendered target inside the patch interior, after a per-channel gain and offset fit. Colour offsets are not blur; checked on round 25's outputs, they are up to half the raw error (gravel 7.5 raw vs 3.4 adjusted) and neither method can fix them. Raw error is reported.
- **Blur attained:** the validated disk-radius fit (rounds 18–28); tolerance max(2 px, 20 %).
- **Preservation:**
  - focused-cell MAE against the source, as added over the baseline;
  - change in all other cells against the baseline output.

## Rules (fixed now)

**Gate:**
- the baseline reproduces round 25 (≤ 1/255);
- repeated runs are identical;
- α = 0 at every location reproduces the baseline exactly;
- each hooked module runs once per generation.

**Per case and location:** the best edit is the one with the lowest primary error among edits adding ≤ 0.5/255 focused MAE, or the baseline if none is better.

**A location passes if all of these hold:**
- median (best edit error / best input-only error) ≤ 0.80 over the 8 cases;
- the edit beats input-only correction on ≥ 6 of the 8 cases;
- at least as many cases are within blur tolerance as with the baseline;
- the focused-MAE guard holds (by construction).

| outcome | rule |
|---|---|
| **LOC(x)** | at least one location passes; x is the one with the lowest median ratio. "Small edits at this location correct output errors that changing the blur request does not." |
| **NONE** | no location passes: "this smoothing/sharpening edit family did not demonstrate useful control at these locations". A failing location where ≥ 4 of 8 best edits sit at \|α\| = 0.10 is marked **unresolved** (search range too small), not ineffective. |

The thresholds are resource decisions, not publication standards. The paired crops (`results/crops_t11.6.png`) are inspected for structural damage before any claim.

**If LOC:** freeze the location, edit family, search budget and selection rule, then confirm on four fresh source images (next round).
