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
