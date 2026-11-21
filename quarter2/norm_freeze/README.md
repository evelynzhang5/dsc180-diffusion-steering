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
