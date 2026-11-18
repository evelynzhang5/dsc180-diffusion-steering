# Q2 round 29: where can a small feature edit correct BokehDiff's output?

**Question.** Rounds 26–28 showed:
- the same blur request gives different blur on different content (soft content under-blurs, high-contrast content over-blurs);
- this is produced in the UNet after the blur attention (PISA), not by the attention spread or the decoder.

This round asks a practical steering question instead of a mechanistic one: **at which location, if any, can one small, restricted edit move the output closer to the requested blur, beyond what changing the blur request itself achieves?** It does not try to prove where the error begins.

**This is privileged feasibility testing.** Every edit is chosen per case using the known target. A positive result would not yet be a deployable steering method.

**Status.** Run Nov 17 (pod `q2-steer`, 592 generations at 1.6 s, GTX 1080 Ti). Plan and rules were committed (f0aba56) before the run. The gate passed exactly: the baseline matches round 25, repeats are identical, and α = 0 changes nothing at all six locations (max difference 0).
- **Outcome: NONE.** No location passes. Five of six are formally **unresolved**: the best edits sit at |α| = 0.10.
- **But the edits are not content-specific.** At almost every location the best edit is the same for every patch: maximum feature smoothing (σ = 32 px, α = +0.10). That holds for under-blurring chelsea and over-blurring coins and gravel alike. It barely changes the measured radius (≤ 1 px) and lowers the error by washing out texture. A larger α would most likely give more of the same, not a per-patch correction.
- **The primary error can be gamed by over-blurring.** Coins' best input-only request is 2.09× (the grid top), which delivers a 25 px blur for an 8 px target.

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

## Results

Files:
- `results/steer_locate.json`;
- `results/steer_locate_cases.json`;
- `results/crops_t11.6.png`.

Images are in `outputs/q2_steer_locate/`.

**Module shapes:**

| location | shape |
|---|---|
| pre_pisa, post_pisa | [1, 640, 32, 32] |
| mid | [1, 1280, 16, 16] |
| up_late | [1, 320, 64, 64] (UpBlock2D) |
| latent | [1, 4, 64, 64] |
| dec_early | [1, 512, 64, 64] |

**Primary error per case** (gain/offset-adjusted MAE vs the disk target; * = best edit at |α| = 0.10):

| case | baseline | input-only (factor) | pre_pisa | post_pisa | mid | up_late | latent | dec_early |
|---|---|---|---|---|---|---|---|---|
| gravel_0 @ 8 | 3.42 | 2.64 (0.89) | 3.25* | 3.27* | 3.42* | 3.31 | 3.01* | 2.84* |
| gravel_0 @ 11.6 | 2.38 | 1.96 (0.91) | 2.20* | 2.28* | 2.37* | 2.38 | 2.12* | 2.03* |
| astronaut_0 @ 8 | 7.39 | 7.37 (1.01) | 6.79* | 6.85* | 7.36* | 7.39 | 6.05* | 5.93* |
| astronaut_0 @ 11.6 | 6.87 | 6.74 (1.04) | 6.18* | 6.24* | 6.84* | 6.87 | 5.51* | 5.34* |
| chelsea_0 @ 8 | 2.69 | 2.33 (1.13) | 2.38* | 2.54* | 2.68 | 2.56 | 1.93* | 1.82* |
| chelsea_0 @ 11.6 | 3.03 | 2.09 (1.16) | 2.59* | 2.88* | 3.03* | 2.76 | 2.11* | 2.01* |
| coins_0 @ 8 | 10.84 | 8.50 (2.09) | 9.10* | 9.08* | 10.82* | 8.42* | 9.20* | 8.71* |
| coins_0 @ 11.6 | 10.46 | 7.10 (2.09) | 8.54* | 8.71* | 10.43* | 8.41* | 8.75* | 8.30* |

**Per location** (rule: median ratio ≤ 0.80 and ≥ 6 wins vs input-only):

| location | median ratio vs input-only | wins | median ratio vs baseline | within blur tolerance (baseline 4/8) | best at \|α\| = 0.10 | other cells changed (MAE) |
|---|---|---|---|---|---|---|
| pre_pisa | 1.10 | 2 | 0.89 | 7 | 8/8 | 0.11 |
| post_pisa | 1.13 | 2 | 0.94 | 5 | 8/8 | 0.10 |
| mid | 1.24 | 1 | 1.00 | 4 | 7/8 | 0.02 |
| up_late | 1.14 | 1 | 0.96 | 6 | 2/8 | 0.08 |
| latent | 1.05 | 3 | 0.83 | 4 | 8/8 | 0.08 |
| dec_early | 0.99 | 4 | 0.80 | 4 | 8/8 | 0.08 |

Input-only correction alone lowers the error by a median 20 % vs the baseline. All edits stay well inside the focused-region guard (≤ 0.03/255 added).

**Post hoc (not a frozen rule):** restricting input-only correction to requests whose delivered blur is within tolerance removes coins' over-blur "win" (coins best becomes 0.80× request). Against that, dec_early reaches a median ratio of 0.89 with 6/8 wins and latent 0.93 with 4/8. Neither reaches 0.80.

## Interpretation

- **No useful steering location was found with this edit family**, at this strength.
- **The "unresolved" flags should not be read as "promising".** The best edits all push in the same direction, maximum feature smoothing, regardless of whether the patch under- or over-blurs. They reduce the error by lowering texture contrast (visible in the crops: gravel and coins look washed out), not by moving each patch's blur toward its target; measured radii change by ≤ 1 px except at up_late. A content-independent smoothing edit cannot remove a content-dependent error. A larger α range would test "more smoothing", not steering.
- **The error measure needs fixing before the next round.** Coins' unconstrained input-only optimum is a gross over-blur (25 px for 8 px) with the lowest score, and global smoothing scores well everywhere. Any next round should score error only among outputs within blur tolerance, or use a measure that penalizes lost texture contrast.
- **What this suggests for steering:**
  - an edit whose *direction* depends on the patch (smooth where the model under-blurs, sharpen where it over-blurs), chosen from a measured signal rather than fixed;
  - or an edit family that changes kernel *shape* (rim and tails) rather than overall smoothness.

  The latent and early-decoder locations gave the largest error reductions vs baseline (0.83 and 0.80 median). They are the cheapest places to test this, and the crops show no structural damage there.

## Exploratory reanalysis: same constraints on every method (`reanalyze29.py`, saved outputs only)

**Rule.** Round 29's frozen outcome (NONE) is unchanged. A reviewer pointed out that blur tolerance should constrain selection, not only be checked afterwards. So for every method, the best output is chosen only among candidates that are:
- within blur tolerance (max(2 px, 20 %));
- within the focused-region guard.

A case with no valid candidate is an **attainment failure** and is kept, not dropped.

The candidate sets are:
- input-only: 24 requests, including the baseline;
- each feature location: 8 edits plus the baseline.

Results: `results/reanalysis.json`.

| location | cases attained (input-only: 8/8) | median ratio vs input-only | wins |
|---|---|---|---|
| pre_pisa | 7 | 1.03 | 2/7 |
| post_pisa | 6 | 1.06 | 2/6 |
| mid | 4 | 1.08 | 1/4 |
| up_late | 8 | 1.06 | 1/8 |
| latent | 7 | 1.01 | 3/7 |
| dec_early | 7 | **0.96** | 4/7 |

- **Correction to the post-hoc note above.** The "dec_early 0.89, 6/8" figure compared unconstrained feature edits against constrained input-only correction. Constraining both gives 0.96 with 4/7 wins, plus one attainment failure. **No location shows a reliable advantage.**
- **Feature edits cannot bring coins @ 11.6 px into tolerance.** The baseline delivers 14.8 px against a 13.95 px limit; every location except up_late fails. Input-only correction attains all 8 cases.
- **Sign choice was already covered.** Positive and negative α were searched per case, and selections do differ in sign (e.g. gravel @ 8 picks sharpening at the latent and decoder). A rule like "smooth under-blurred patches, sharpen over-blurred ones" picks from the same candidates, so it cannot beat this search.

**Size vs shape.** For each selected output with measured radius r̂, the error is split into two parts:
- the *size* part: error of an ideal disk at r̂ against the target;
- the *shape* part: error of the output against that ideal disk. This part covers everything other than size, including any non-blur change.

After the best valid request adjustment (all 8 cases), the median size part is **0.69** and the median shape part is **4.64**. Once the request is tuned, nearly all remaining error is the kernel's shape, not its size.

**Decision** (reviewer's table):
- no reliable advantage at any location → **close this single-width smoothing/sharpening edit family**;
- correct radius but persistent shape mismatch → the next test is a **two-scale correction**, run on top of the best valid request adjustment, at the final latent and early decoder:

  h′ = h + M[a(G_s1 ∗ h − h) + b(G_s2 ∗ h − h)]

  It asks whether internal steering adds anything after the ordinary control is tuned. It must be run, not composed from saved outputs.

**Not shown:** that any location is ineffective in general (only this edit family and range); fresh images; real photos.

**If LOC:** freeze the location, edit family, search budget and selection rule, then confirm on four fresh source images (next round).
