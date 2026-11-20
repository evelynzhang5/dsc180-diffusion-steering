# Round 30: does a linear readout of BokehDiff's activations predict its blur error?

**Question.** Rounds 24–29 showed the same blur request gives different blur on different content (about ±10 %), produced in the UNet after PISA. Simple image statistics predict this error only weakly (rank correlation 0.46 for contrast, 0.33 for edge strength, 0.0 for fine detail; 12 held-out patches, round 25 outputs). If a linear readout ⟨w, h⟩ of the model's own activations predicts it, that readout can set an adaptive steering strength per region: h' = h − M · e · w/‖w‖², with e the predicted error. This round only tests the readout, not steering.

**Setup.** Round 25's baseline (fp32 PISA, original gate, frozen radius curve C1 G0, seed 0, untiled 512²), round 24/25 3×3 layouts (centre focus, corner anchors, 6 free cells at the same |gap|).
- Generations: {dev, test} patches × 2 groups × 6 positions × 6 targets (4, 6, 8, 10, 11.625, 14 px) = 144, plus the gate. Dev patches come from the six sources used to fit the radius curve; test patches from the six held-out sources.
- Recorded per generation: each free cell's activations, mean-pooled over the cell interior, at 8 points: PISA block output, down_blocks.1, down_blocks.2, mid_block, up_blocks.0/1/2, final latent.
- Label per cell: e = log(delivered / target), delivered = disk-equivalent radius (measure19.fit_fast). **Content part**: e minus the mean e over that generation's 6 cells, which removes the per-scene offset a global curve already handles. Predictions are centred the same way.

**Rules (fixed before the full run).**
1. Gate: the gate generation reproduces round 25's `cor_g0_r0_G0_t11.6.png` exactly (max difference 0), and every recording point fires once per generation.
2. Probe per layer: standardised features, ridge regression on dev cells only (432). Ridge strength (10⁻²…10⁵) and the layer are chosen by leave-one-source-out cross-validation on dev. The test set (432 cells) is used once, for the chosen layer; other layers' test numbers are reported as exploratory.
3. Baselines with the same procedure: contrast only (patch log std), and all three image statistics.
4. **PASS** if the chosen layer's held-out Spearman ≥ 0.70 **and** ≥ contrast baseline + 0.20. Otherwise FAIL: at this pooling, the activations do not carry a usable error signal beyond image statistics.
5. Uncertainty: 95 % bootstrap interval over the 6 held-out sources.

**Smoke test (Nov 19, DSMLP GTX 1080 Ti).** Gate exact (max diff 0); all 8 recording points fired once; pooled features finite at every layer, shapes post_pisa 640×32², down1 640×16², down2 1280×16², mid 1280×16², up0 1280×32², up1 640×64², up2 320×64², latent 4×64²; 2.1 s per generation, peak 9.6 GB. Dev cells already vary widely at the same target (8 px: 6.0–8.4; 14 px: 12.1–15.3).

**Caveats known in advance.** 12 sources in total, so label variation is mostly between sources; a readout can predict without being something the model uses (steerability is a separate test); mean pooling discards spatial structure, so a FAIL here does not rule out spatial readouts.

Files: `run30.py` (driver), `smoke_check30.py`, `analyze30.py` (frozen decision), `dsmlp_probe.sh` (`SMOKE=0` for the full run).

## Results (Nov 19, full run on DSMLP)

145 generations, 1.9 s each. Gate passed: round 25 reproduced exactly (max diff 0), every recording point fired once.

**Decision: FAIL.** The layer chosen by dev cross-validation (down_blocks.2) predicts the held-out content-specific error with Spearman **0.37** (95 % interval over sources −0.28 to 0.66), below the 0.70 bar. It does beat the contrast baseline (−0.14) by more than 0.20, so the margin part passed.

| Predictor (fit on dev, tested on held-out) | Dev CV ρ | Held-out ρ | 95 % interval |
|---|---|---|---|
| **down_blocks.2 (chosen)** | 0.33 | **0.37** | −0.28 – 0.66 |
| contrast only | −0.28 | −0.14 | −0.77 – 0.60 |
| three image statistics | −0.28 | −0.08 | −0.66 – 0.62 |
| *exploratory:* mid_block | 0.29 | 0.59 | −0.07 – 0.75 |
| *exploratory:* up_blocks.0 / up_blocks.2 / latent | 0.25 / 0.24 / −0.06 | 0.47 / 0.46 / 0.42 | |
| *exploratory:* PISA output / down_blocks.1 / up_blocks.1 | 0.27 / −0.06 / 0.10 | 0.36 / 0.22 / 0.16 | |

**Ceiling check (exploratory, from the saved labels).** The labels are reliable: per patch and target, even vs odd positions agree at Spearman 0.82 (dev) and 0.90 (held-out). Knowing only which patch a cell holds predicts its centred error at 0.83 on held-out cells (0.71 on dev). So 0.70 was reachable in principle; the probe recovers less than half of the patch-level signal.

**Interpretation.**
- Activations carry some content-error signal (0.36–0.59 across post-PISA layers), and more than image statistics, which do not transfer from the curve-fitting sources to the held-out sources at all (the contrast relationship flips sign between them; the 0.46 quoted earlier was on held-out patch averages without this separation).
- It is too weak and too uncertain to set a per-region steering strength. With only 6 sources per side, intervals are wide; dev sources are also less consistent than held-out ones (page_0 +0.11 vs page_1 −0.09), which limits what a dev-trained probe can learn.
- Not ruled out: spatial (non-pooled) readouts, nonlinear readouts, more training sources. A FAIL here closes mean-pooled linear readouts as the adaptive-strength signal at this data size.

**Next.** Do not build adaptive steering on this readout. Options: (a) more sources for training (more crops and scikit-image images) before retrying the probe; (b) use the measured error from a first pass (two-pass correction) as the strength signal, which is exact but costs a second generation; (c) proceed with the layer diagnosis (option A), which does not need a predictor.
