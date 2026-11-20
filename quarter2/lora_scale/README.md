# Round 31: can scaling BokehDiff's LoRA in one block correct the output beyond adjusting the request?

**Question.** Even when the desired output is known, can an internal intervention improve it beyond adjusting the blur request? Round 30 tested prediction (a probe of the error); this round tests control. It is a search for **useful control, not proof of where the error originates**: a block can compensate for an error created elsewhere.

**Corrections to earlier framing (carried into this round).**
- PISA is not a guaranteed-correct lens renderer: it constrains learned attention with blur and depth information, and content still affects the mixing.
- The error has not been conclusively located after PISA. Equal attention spread does not imply equal attention outputs, and swapping identical decoders (round 28) cannot exclude the decoder's response to different latents.
- The scene-8 figure comparison changed the corner blur controls and depth configuration as well as the surrounding texture, so it suggests a context effect without isolating its cause.
- Automatic per-region correction is not a guaranteed fallback: round 30 found no reliable predictor.

**Intervention.** BokehDiff's LoRA (diffusers 0.32.1 + PEFT 0.14; 1576 tensors) covers seven UNet blocks: down_blocks.0–2, mid_block, up_blocks.0–2. down_blocks.1 contains the PISA attention. For one block at a time, every LoRA layer's scaling is multiplied by s ∈ {0.5, 0.75, 1.25, 1.5}. The change acts on the whole image, so one generation gives all four cases at a target. In total: 28 variants.

**Fixed configuration and cases.** Round 25's baseline (fp32 PISA, original gate, frozen radius curve, seed 0, untiled 512²), held-out group 0 / position 0. Round 29's eight development cases: gravel_0, astronaut_0, chelsea_0, coins_0 × targets 8 and 11.625 px. Reference: sRGB disk blur of the input at the target; primary error: per-channel gain/offset-adjusted MAE in the cell interior (round 29).

**Stages (`run31.py`).**

| Step | What | Establishes |
|---|---|---|
| Gate (smoke stage; aborts the run on failure) | Baseline reproduces round 25 exactly; repeat identical; s = 1 in all blocks identical; s = 1.5 in all blocks changes the output; LoRA layers found in every block and none elsewhere | Faithful implementation, working hook |
| Scan | 7 blocks × 4 scales × 2 targets at the baseline request (56 generations) | Which interventions move the output |
| Request-only search | Per case, that cell's request only: factors 0.3–2.0 (step 0.1) × baseline request, then 3 refinements (±0.05, ±0.025, ±0.0125) around the best valid candidate, or, if none is valid, around the candidate with delivered radius closest to the target (192 generations) | The strong comparison |
| Retune | Per case, the 3 scan variants with the lowest error among those passing the focus guard, each with the same request search (576 generations) | Improvement survives correcting blur strength |
| Preservation | Focused-region MAE added vs the baseline (guard ≤ 0.5/255); MAE change in the other free cells vs the baseline (reported) | Improvement is not from erasing detail |

**Rules (fixed before running; `analyze31.py`).**
- Same selection for both methods: keep candidates with delivered blur within tolerance (|r̂ − t| ≤ max(2 px, 20 %)) and focused-region guard ≤ 0.5/255; select the lowest primary error.
- Request-only candidates: its search (including the baseline request). LoRA candidates: scan and retuned outputs only, every one with a block scaled, so a win requires the LoRA change.
- Unattainable cases are failures: LoRA unattainable → loss (ratio ∞); request-only unattainable and LoRA attained → win (ratio 0).
- **PASS** if median (LoRA error / request-only error) ≤ **0.80** over the 8 cases **and** LoRA wins ≥ **6 of 8**. Otherwise **FAIL**: "single-block LoRA scaling in 0.5–1.5 did not demonstrate useful control beyond request adjustment on these cases".
- The crop sheet (`results/crops_t11.6.png`, 16 tiles) is inspected for structural damage before any claim.
- These reused cases establish feasibility only. A PASS leads to a separate confirmation round: freeze the selected intervention and test fresh source images.
- Reported, not part of the decision: each single (block, scale) setting across all cases; size vs shape split of the selected outputs; changes in other cells.

Files: `run31.py`, `analyze31.py`, `metrics29.py` (round 29's cases and error), `dsmlp_lora.sh`.

## Results (Nov 19, 2026; 829 generations, pod `q2-lora`)

**Gate passed exactly.** Round 25's baseline reproduced at both targets (max abs diff 0). The repeat run was identical, and s = 1 in all blocks was identical. s = 1.5 in all blocks changed the output (max diff 64). LoRA layers found per block: down0 7, down1 52, down2 211, mid 108, up0 319, up1 79, up2 12. None were found outside these blocks.

**Decision: FAIL.**
- Median ratio (LoRA / request-only) was **0.95**; the bar was ≤ 0.80.
- LoRA won **7 of 8** cases, which clears the ≥ 6 requirement.
- Both methods attained all 8 cases. The unadjusted baseline was within tolerance on only 4 of 8.

Single-block LoRA scaling in 0.5–1.5 did not demonstrate useful control beyond request adjustment on these cases.

| Case | Baseline err (r̂) | Request-only err (r̂, factor) | LoRA err (r̂, setting, factor) | Ratio |
|---|---|---|---|---|
| gravel_0 @ 8 | 3.42 (10.3) | 2.64 (8.4, 0.888) | 2.68 (8.4, up0×0.75, 0.95) | 1.01 |
| gravel_0 @ 11.6 | 2.38 (13.7) | 1.96 (12.2, 0.913) | 1.82 (12.1, down2×0.5, 0.913) | 0.93 |
| astronaut_0 @ 8 | 7.39 (8.0) | 7.37 (8.2, 1.013) | 6.99 (8.2, up2×1.25, 1.063) | 0.95 |
| astronaut_0 @ 11.6 | 6.87 (11.3) | 6.74 (12.0, 1.038) | 6.13 (11.9, up1×0.75, 1.075) | 0.91 |
| chelsea_0 @ 8 | 2.69 (6.8) | 2.33 (8.0, 1.125) | 2.25 (8.1, up0×1.5, 1.125) | 0.97 |
| chelsea_0 @ 11.6 | 3.03 (9.2) | 2.09 (11.9, 1.163) | 2.07 (11.4, up0×1.5, 1.163) | 0.99 |
| coins_0 @ 8 | 10.84 (10.1) | 8.83 (6.5, 0.788) | 8.36 (6.7, up2×1.25, 0.825) | 0.95 |
| coins_0 @ 11.6 | 10.46 (14.8) | 10.20 (10.6, 0.775) | 8.56 (10.9, up1×0.75, 0.825) | 0.84 |

**Reported, not part of the decision.**
- **Where the gains come from.** The LoRA gains are small and consistent, and almost all of them are in shape, not size. Shape error (output vs an ideal disk at the delivered radius) is lower in 7 of 8 cases. The largest drop is coins @ 11.6, 10.16 → 8.53.
- **No consistent block.** The selected setting differs across cases: up0, up1 and up2 each appear twice, plus down2 and up0×0.75. No single setting attains all 8 cases and wins. The best single setting was up1×0.75: 3 cases attained, median ratio 0.91.
- **Preservation.** Both methods pass the focus guard. LoRA changes the *other* free cells much more than request-only does: MAE vs the baseline was 0.97–2.50 for LoRA against 0.04–0.47 for request-only. A whole-image weight change costs scene preservation elsewhere.
- **Crop sheet.** `results/crops_t11.6.png` shows no structural damage, and request-only and LoRA outputs look nearly identical.
- **Caveat.** The ~5% gain is the minimum over about 100 candidates per case, against about 25 for request-only, so part of it may be selection on noise. Tested at 7 of 8 cases, it does not need a fresh-image round.

**Reading.** Even with the target known, per-block LoRA scaling buys about 5% beyond tuning the request, and it costs more change in other cells. The content-dependent error is not removable by turning one block's learned adjustment up or down.
