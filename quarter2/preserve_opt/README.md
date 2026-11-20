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
