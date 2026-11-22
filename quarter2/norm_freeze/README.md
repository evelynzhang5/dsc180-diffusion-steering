# Round 37: does freezing downstream normalisation statistics keep the regional correction local?

**Why this round.** In round 35, the optimised edit at up_blocks.1 (converged) matched the teacher inside the four case cells:
- in-region error 0.9–2.3, against 2.2–10.2 for request-only;
- but it changed the rest of the image by about 2/255;
- that change was **flat with distance**: 1.6–2.0/255 from 16 to 200 px away, against 0.23–0.41 for the regional-paste control.

A flat change suggests a path that mixes information across the whole image. Downstream of up_blocks.1, the SDXL UNet has only:
- up_blocks.2: resnets, with no attention (checked at run time);
- conv_norm_out and conv_out.

So the only image-wide paths are the **GroupNorm statistics** (the per-group mean and variance over all positions). Editing 4 cells shifts those statistics and therefore every pixel. This round tests that mechanism directly and cheaply.

**Intervention.** The GroupNorms in up_blocks.2, plus conv_norm_out, can run with the mean and variance recorded from the *unedited* pass of the same image ("frozen"). Nothing else changes. The VAE decoder is untouched: the regional paste shows it adds little spread.

**Arms (up_blocks.1, both targets, all 4 case cells jointly, one combined image each).**

| Arm | What | Purpose |
|---|---|---|
| free | Round 35's optimisation, unfrozen → edit D35 | Reproduces round 35 |
| zero_frz | Zero edit, statistics frozen | Control: freezing alone must change nothing |
| repro | D35 re-applied, unfrozen | Control: reproduces *free* |
| d35_frz | **The same D35, statistics frozen** | Mechanism test |
| opt_frz | Edit re-optimised with statistics frozen (same loss, length rule and learning rate as round 35) | Can a preserving correction be found? |

Image change is measured directly against the baseline, separately in each protected region:
- the focus;
- rocket_0 and text_0;
- the two anchors;
- distance bands from the edited cells.

**Rules (fixed before the run; `analyze37.py`).**
- **Controls, at both targets:**
  - K1: zero_frz changes every protected region by ≤ 0.1/255;
  - K2: repro is within 1/255 (max) of free.
- **MECHANISM SUPPORTED** if, at both targets:
  - the mean protected-region change of d35_frz ≤ 0.5 × free's;
  - the median in-region error ≤ 1.2 × free's.

  If not supported, move on from this mechanism.
- **PRESERVING CORRECTION FOUND** if opt_frz passes round 34's FEASIBLE rule:
  - valid outputs, contrast guard;
  - median ratio to request-only ≤ 0.80, wins ≥ 6/8;
  - unedited cells within request-only's change + 0.5/255.
- **Learning a reusable steering vector** comes only after a preserving correction is found.

Files: `run37.py`, `analyze37.py`, `dsmlp_freeze.sh` (smoke: t = 11.6 only, 5 steps), launched as one chained pod `q2-r37` (smoke, then full).

## Run 1 (Nov 20, 2026): controls failed because of a bug in the test setup; fixed and re-run

Run 1 failed K1 (zero edit with statistics frozen) and K2 (re-applying the edit unfrozen). Both showed the same discrepancy:
- mean |Δ| 0.67–0.86/255, max 13–33, with about 7–11 % of pixels > 2;
- this held even for the **unfrozen** re-application, where nothing should change.

The whole-pipeline repeat remained exact.

**Cause.** BokehDiff's PISA attention adds random draws at inference:
- the occlusion map supersamples with `torch.rand` jitter;
- the attention applies **dropout p = 0.1 with `train=True`**.

Each generation is reproducible only because the seed is reset before it. Every *extra* UNet pass inside a generation (the optimisation steps, the replay passes) drew new random numbers, which by itself moves the output by about 0.7/255.

**Consequences for earlier rounds.**
- Rounds 34–35 optimised a stochastic objective, with a new dropout mask at every step. Their final images include this resampling noise: about 0.3–0.7/255 in the protected regions, judging by run 1's zero-edit arm.
- The leak they report (focus 2.1–3.0/255, flat with distance) is well above that, so their qualitative conclusion stands. Its size will be re-measured by the *free* arm here.
- Rounds 33 and earlier compared separate, seeded generations and are unaffected.

**Fix.**
- The RNG state of the real UNet call is captured, and restored before every inner pass, so all passes see the same dropout mask and occlusion jitter as the real call.
- New control **K0**: a zero edit, unfrozen, must reproduce the baseline to ≤ 1/255 (max).
- Every other rule is unchanged.

Run 1's per-arm numbers are kept in `results/run1_rng_bug/arms_run1.json` and are not interpreted. The re-run writes to `outputs/q2_norm_freeze_v2`.

## Results (re-run with the RNG fix; Nov 20–21, 2026; pod `q2-r37b`)

**Controls pass.**
- G1–G3, C1 and C2 pass as before.
- **K0:** the zero edit, unfrozen, reproduces the baseline exactly (max 0).
- **K1:** the zero edit with frozen statistics changes protected regions by ≤ 0.02/255 (max 1/255).
- **K2:** re-applying D35 reproduces *free* exactly (max 0).

| Arm, t = 8 / 11.6 | In-region error (4 cells) | Focus | rocket / text | Anchors | Distance 16–48 / 96–200 px |
|---|---|---|---|---|---|
| free (round 35's edit, RNG-fixed) | 1.09–2.24 / 0.83–1.75 | 2.66 / 2.48 | 1.0–1.1 / 1.5 | 1.4–1.5 / 1.3–1.4 | 1.87 / 1.42–1.54 |
| zero_frz | baseline | 0.02 | ≤ 0.02 | 0.01 | 0.02 / 0.01 |
| **d35_frz** (same edit, statistics frozen) | **3.9–8.5 / 2.9–5.8** | 0.49 / 0.44 | 0.2–0.6 | 0.6–0.8 | 0.65–0.75 / 0.43–0.54 |
| **opt_frz** (re-optimised, frozen) | **1.02–2.19 / 0.82–1.63** | **0.43 / 0.47** | 0.24–0.30 | 0.13–0.17 | 0.43 / 0.23 |

**Decisions (fixed rules).**
- **Controls: OK.**
- **MECHANISM SUPPORTED: NO.** Freezing the statistics removes 70–80 % of the image-wide change from the *same* edit (focus 2.5–2.7 → 0.44–0.49). But that edit also loses most of its in-region effect: error 1.3 → 6.4 and 1.1 → 3.4. Its regional correction partly travelled *through* the shifted statistics.
- **PRESERVING CORRECTION FOUND: YES.**
  - Re-optimised with frozen statistics, the edit reaches the teacher in-region and keeps the focus at 0.43–0.47/255, within the 0.5 guard.
  - Other cells change by 0.24–0.30/255 and anchors by 0.13–0.17/255.
  - Median ratio to request-only 0.31, wins 8/8, unedited-cell change 0.27 (limit 0.84).
  - The t = 11.6 optimisation ran its full 1000 steps.

**Reading.**
1. **The image-wide leak runs through the downstream GroupNorm statistics.** These are the only image-wide path after up_blocks.1, and holding them at baseline cuts the far-field change from about 1.5 to about 0.25/255.
2. **A local edit plus frozen statistics can hit a regional latent target while preserving the rest of the image.** This is the first preserving correction in rounds 34–37, but it is a per-image, privileged optimum. As in rounds 34–35 it is high-dimensional, so it is not a steering vector.
3. **The leak measured in rounds 34–35 was real.** Under the RNG fix, *free* still changes the focus by 2.5–2.7/255. Their qualitative conclusion stands.

**Major caveat: the target.** Round 38 (D2b–c) found that BokehDiff's outputs match a **linear-light** disk blur far better than the **sRGB** disk blur used to build this teacher. Rescored against the linear reference (`results/r37_linear_rescore.json`), the preserving correction is **worse than request-only**:
- ratios 0.97–1.55 for gravel, astronaut and chelsea, and 3.4–3.5 for coins;
- it reproduces the sRGB teacher faithfully, and that teacher is the wrong target.

So the method result (preservation through frozen statistics) is reference-independent. The specific "correction" is not useful.

Files: `results/norm_freeze.json` (decision), `results/r37_linear_rescore.json`, and `results/run1_rng_bug/`.
