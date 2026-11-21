# Round 35: can the optimised edit correct the regions *without* disturbing the scene?

**Why this round.** In round 34, an activation edit optimised against the teacher reached the target inside the case regions:
- at up_blocks.1, the in-region error fell to the teacher's level, 0.8–2.2 against 2.2–10.2 for request-only;
- but the edit changed the in-focus centre by 2.3/255 and the other cells by 1.0–1.5/255.

Its loss constrained only the case cells, so the later layers were free to spread the edit, and every location failed on preservation. Three of the four locations had also not converged. This round asks whether a correction that leaves the rest of the scene intact exists at all.

**Change from round 34 (only this).**
- **Target.** The target is the *regional-paste latent* over the whole image:
  - Z\* = Z0 + m·(Z_T − Z0);
  - Z0 is the student's own unedited final latent, Z_T the teacher's, and m the case-cell area mask.
- **Loss.** It is the whole-latent MSE to Z\*. Inside the cells it pulls toward the teacher; outside it penalises any change from the unedited output.
  - Control C2 (regional paste, 8/8 in round 34) already shows that Z\* decodes to a valid, preserved image, so the target is attainable at the latent.
- **Edit.** As in round 34, the edit is confined to the case cells at the location.
- **Length.** The optimisation runs at least 200 steps and at most 1000. It stops once the loss fell < 1 % over the last 10 % of steps, checked every 50.
- **Locations.** PISA block output, up_blocks.0, up_blocks.1. mid_block is dropped: it converged in round 34 and could not express the correction.
- **Unchanged from round 34:**
  - the checks (G1–G3, C1, C2), which must pass before any optimisation;
  - the revised error;
  - the request-only comparison;
  - the edit form, h′ = h + m·D·rms_m(h), optimised with Adam at lr 0.05.

**Rules (round 34's, unchanged; `analyze35.py`).** A location is **FEASIBLE** if all of the following hold:
- median (edit / request-only) ≤ 0.80 on the revised error;
- wins ≥ 6/8;
- outputs are valid (|r̂ − t| ≤ max(2 px, 20 %) and focus guard ≤ 0.5/255);
- the contrast guard holds;
- the unedited cells change ≤ the largest request-only change + 0.5/255.

If the loss still fell by > 5 % over the last 10 % of steps, the location is reported "not converged".

**Reading.**
- **FEASIBLE at a location:** a scene-preserving correction exists there. It is still a privileged, per-image optimum. Whether it can be predicted without the target is a separate question, and the reported size and top-1 direction share show how hard that would be.
- **Not feasible and converged:** at this budget, no edit of this form at that location both reaches the target and preserves the scene.

**Smoke (DSMLP, Nov 20).** All checks passed again: round 25 reproduced exactly, C1 exact, C2 8/8. Five steps at the PISA output took 9 s, with peak memory 9.9 GB. The in-cell and out-of-cell loss terms are both reported.

Files: `run35.py`, `analyze35.py`, `dsmlp_pres.sh` (`SMOKE=1`: 5 steps at one location). Shared modules are copied from round 34. The run is chained with round 36 in one pod (`q2-r3536`); each full run starts only if its own smoke passed.

## Results (Nov 20, 2026; pod `q2-r3536`, 87 min for this round)

**All checks and controls passed again before optimisation.** Round 25 was reproduced exactly, the x0 formula matched to 2.4e-7, and C1, G3 and C2 all passed (C1 exact, G3 8/8, C2 8/8).

**Decision: no location is FEASIBLE.**
- **up_blocks.1 converged** (stopped at 200–250 steps), so its result is **negative**.
- **The PISA output and up_blocks.0 ran the full 1000 steps without converging**, so they are **unresolved**.
- **Every case at every location fails on preservation.** The focus change was 1.6–2.6/255 against a guard of 0.5, so every ratio counts as ∞.

| Location | Steps | Final loss in / out of the cells (t = 8; 11.6) | Error in the regions vs request-only | Focus change (guard 0.5) | Other cells (request 0.34) | ‖δ‖/‖h‖ | Top-1 share |
|---|---|---|---|---|---|---|---|
| PISA output | 1000 (not converged) | 0.0068 / 0.0054; 0.0036 / 0.0042 | 1.2–3.3 vs 2.2–10.2 | 2.3–2.6 | 2.0 | 3.3 | 0.16 |
| up_blocks.0 | 1000 (not converged) | 0.017 / 0.0050; 0.013 / 0.0041 | 1.7–4.4 | 1.6–1.8 | 1.8 | 1.8–2.2 | 0.16–0.17 |
| up_blocks.1 | 250 / 200 (converged) | 0.0005 / 0.0039; 0.0004 / 0.0030 | **0.9–2.3** (teacher level) | 2.1–2.3 | 1.6 | 0.7–0.9 | 0.14–0.16 |

**Reading.**
1. **The penalty did not buy preservation.**
   - The out-of-cell term was the *larger* part of the converged loss at up_blocks.1 (0.003–0.004 against 0.0004 inside the cells), so the optimiser was pushing against it. Yet it stopped at a change outside the regions as large as round 34's (focus about 2.2/255, other cells 1.6).
   - At the PISA output, extra steps improved the in-region error a lot (now 1.2–3.3, against 2.6–7.7 in round 34), but preservation got worse (focus 2.3–2.6).
2. **The leak is global, not local spread** (computed from the saved images, mean |Δ| against the baseline by distance from the edited cells):

   | Distance from the edited cells | 16–48 px | 48–96 px | 96–200 px |
   |---|---|---|---|
   | Regional paste | 0.39–0.41 | 0.26–0.27 | 0.23–0.24 |
   | up_blocks.1 edit | 1.9–2.0 | 1.7–1.8 | 1.6–1.7 |
   | PISA-output edit | 2.6 | 2.3–2.4 | 2.3–2.4 |

   - The change does not decay with distance.
   - At up_blocks.1, part of it is a uniform colour shift of +0.8 to +1.9/255 per channel.
   - A change that is flat across the image is what you would expect if editing 4 cells shifts statistics shared across the whole image, such as GroupNorm statistics or global attention. A local edit then moves every pixel. This mechanism is a hypothesis; it has not been tested.
3. **Edits stay high-dimensional:** top-1 direction share 0.14–0.17, with ‖δ‖/‖h‖ up to 3.3 at the PISA output.

**Consequence for the project question.**
- At up_blocks.1, the last location with a result, the privileged optimum still cannot reproduce the target in the regions while keeping the rest of the image within 0.5/255. The cost is about 2/255 everywhere, roughly 18× the regional paste's 0.12 in the focus.
- An image-wide guard of 0.5/255 is strict. Whether a ~2/255 global shift is acceptable is a judgement call; these rules fixed it in advance as "no".
- **Not tested:**
  - a lower learning rate after convergence (Adam at lr 0.05 may sit on a noise floor);
  - letting the edit act outside the cells, where it could cancel the global shift.

Files: `results/preserve_opt.json`, `results/preserve_opt_cases.json`. Images are in `outputs/q2_preserve_opt/img/` (gitignored).
