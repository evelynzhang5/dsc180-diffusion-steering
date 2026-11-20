# Round 34: does any activation edit at a location reach the target? (teacher with fixed controls)

**Why this round.** A Nov 20 review of rounds 32–33 found four problems, each addressed here.

1. **The round 29 error score rewards contrast loss.** It fits the target's gain and offset to the output, so halving an output's contrast halves its score without changing the measured radius.
   - *Fix:* a revised error, `metrics34.off_mae`. It fits a per-channel offset only, so global brightness and colour offsets are ignored while lost contrast and detail are penalised.
   - *Validation (`validate34.py`, CPU, 8 cases):* the error behaves correctly under contrast reduction, extra smoothing, shifts, radius changes, Gaussian and ring kernels, and offsets. **80/80 checks pass.**
   - On BokehDiff's own baseline outputs, halving the contrast changes the scores as follows:

     | Score | Original | Contrast halved |
     |---|---|---|
     | Old | 3.42 | 1.71 |
     | New | 5.01 | 8.82 |

   - The offset check initially failed only because of float32 rounding (3.5e-7); its tolerance was set to 1e-3.
   - The teacher, an essentially correct output, has fitted gain 1.00–1.03. BokehDiff therefore does not shift colour gain, so ignoring only the offset is justified.
2. **Rounds 31–32 rescored with the new error (exploratory; frozen decisions unchanged; `rescore34.py`):**
   - Round 31's LoRA result weakens from 7/8 wins (median 0.95) to **5/8 (median 0.97)**, and its selected outputs had slightly lower contrast than request-only in 6 of 8 cases. Part of that small gain was contrast loss.
   - Round 32 is unchanged: all arms are 0.97–1.04.
3. **Round 33's teacher signal is input-relative.**
   - BokehDiff takes one DDPM step from the *clean* input latent x: z = (x − √(1−ᾱ)·ε)/√ᾱ. Checked here: this formula matches the scheduler to 2.4e-7.
   - The teacher's UNet output is therefore relative to its own blurred input. Copying it, or its activations, into the student's sharp input removes the blur (round 33: the ceiling gave 1.2–1.7 px).
   - *Fix:* the teacher's **final denoised latent** Z_T is used as the signal. The edit is **optimised for the student's own input**.
4. **Checks ran after the scan, and per-cell choices came from different images.**
   - *Fix:* every check and control now runs first, and the run aborts without optimising if one fails.
   - All 4 case cells are edited jointly, and the score comes from one combined image.

**Question.** Even with the target known, does an activation edit at a given location exist that makes BokehDiff's output match the target better than adjusting the request? This is a privileged upper bound for steering at that location.
- **Not feasible** (and converged): no edit at that location of this form helps, so steering there is closed.
- **Feasible:** the next question is whether that edit is predictable without the target. The reported size and single-direction share of the optimised edit indicate how hard that would be.

**Setup.**
- **Configuration:** round 25's baseline: fp32 PISA, original gate, frozen radius curve, seed 0, held-out group 0 / position 0.
- **Cases:** round 29's 8 cases.
- **Teacher:** the case cells already blurred to the target, with a zero blur request there.
- **Request-only comparison:** round 31's request search (192 generations), re-selected with the revised error.

**Checks and controls (`run34.py`, before any optimisation).** A smoke run on DSMLP on Nov 20 passed all of them.

| Check | Requirement | Smoke |
|---|---|---|
| G1 | Baseline reproduces round 25 exactly; repeat identical | 0, 0 |
| G2 | Hand x0 formula equals the scheduler step (< 1e-3) | 2.4e-7 |
| G3 | Pass-through: teacher valid, and error ≤ 0.5 × request-only in ≥ 6/8 | 8/8 (e.g. coins 1.04 vs 10.23) |
| C1 | Positive control: replace the whole final latent with Z_T → the teacher image exactly | max diff 0 |
| C2 | Regional control: paste Z_T into the case cells only → valid, error ≤ 0.5 × request-only in ≥ 6/8 | 8/8 (region-paste matches the teacher, e.g. 0.74 vs 0.75) |

**Optimised edit.**
- **Form:** at location L, h′ = h + m · D · rms_m(h), with D a free per-position, per-channel tensor confined to the 4 case cells (m = area fraction).
- **Optimisation:** Adam (lr 0.05, 150 steps) minimises the masked MSE between the student's final latent and Z_T. The loss is in latent space, so no decoder gradient is needed.
- **Locations:** PISA block output (640×32²), mid_block (1280×16²), up_blocks.0 (1280×32²), up_blocks.1 (640×64²).
- **Reported per location:** the edit's size ‖δ‖/‖h‖ and the share of its energy in its top 1 and top 5 channel directions. A large top-1 share would mean a single steering vector could represent it.
- **Smoke:** 5 steps at the PISA output took 9 s, with peak memory 9.95 GB of the GTX 1080 Ti's 11 GB.

**Rules (fixed before the full run; `analyze34.py`).**
- **Valid:** |r̂ − t| ≤ max(2 px, 20 %) and focus guard ≤ 0.5/255.
- **Contrast guard:** the edit's contrast ratio ≥ request-only's − 0.05.
- **Failures:** an invalid output or a failed guard counts as ratio ∞.
- **A location is FEASIBLE if all three hold:**
  - median (edit / request-only) ≤ **0.80** on the revised error;
  - wins ≥ **6 of 8**;
  - the unedited cells (rocket_0, text_0) change ≤ the largest request-only other-cell change + 0.5/255.
- **Convergence:** a location whose loss still fell by > 5 % over the last 10 % of steps is reported "not converged". A non-feasible result there is unresolved, not negative.
- **After a FEASIBLE result:** a fresh-image confirmation, and a test of whether the edit can be predicted without the target.

Files: `metrics34.py`, `validate34.py`, `rescore34.py`, `run34.py`, `analyze34.py`, `dsmlp_topt.sh` (`SMOKE=1` for the smoke run). Shared modules are copied from round 33.
