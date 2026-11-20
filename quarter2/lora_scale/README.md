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
