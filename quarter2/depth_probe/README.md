# Round 36: does an inner product read depth, does an edit control blur, and does adapting its strength help?

**Why this round.** The mentor proposed reading depth from the activations with an inner product and using it to set a local steering strength:

  d̂_p = w_d·h_p + b  (read depth at location p),  h′_p = h_p + α_p·u  (local edit with strength α_p)

These are **three separate claims**, and passing the first does not establish the other two. w_d (reads depth) and u (changes behaviour) need not be the same direction. Round 30 does not answer claim A: its labels were blur errors, not depth.

**A structural fact to keep in mind.**
- In BokehDiff, disparity reaches the UNet *only* through PISA's occlusion map. The blur amount comes from the separate CoC (blur request) channel.
- A probe that predicts "depth" may therefore be reading the blur request instead. The design below separates the two.
- Control D0 measures how much the activations and output change when *only* the disparity changes.

## Design

**Scenes (`scenes36.py`).** Round 24's 3×3 layout is used: the centre is the focus cell; the corners are anchors at disparity 1 and 0, so the map spans [0, 1]; six free cells hold the textures.
- **Development:** 12 scenes from 6 development source images (grass, brick, camera, coffee, immunohistochemistry, page).
- **Test:** 12 scenes from 6 fresh source images (gravel, astronaut, chelsea, coins, rocket, text).
- Both crops of a source stay in the same split.
- **Factors:**
  - focus disparity d_f ∈ {0.35, 0.5, 0.65};
  - each scene holds all six signed gaps {±0.15, ±0.25, ±0.35};
  - disparity d = d_f + gap.
- **Coverage:**
  - every texture appears at all 6 gaps and all 6 positions;
  - every scene has the same requested blur at two depths (front and behind);
  - different textures share each depth.
- **Correlation:** corr(d, requested blur) = 0.00 across the design. d_f varies, so absolute depth is not just the signed gap (corr 0.91).
- **Label:** the assigned disparity, the inverse-depth quantity BokehDiff is given.
- **Request:** calibrated with round 25's frozen curve to the nominal radius r\* = NOM·K·|gap| ∈ {5.8, 9.7, 13.6} px.

**Location.** up_blocks.0 output (1280 × 32²), preselected because round 32's blur-amount direction lives there. The configuration is round 25's baseline (fp32 PISA, seed 0).

**Order (`run36.py`).** The probe and the strength rule are fitted on development scenes only and **frozen before any test generation**.
1. **Checks.**
   - G1: round 25's baseline is reproduced exactly with the hook attached.
   - G2: repeats are identical.
2. **Development grid.**
   - Each dev scene is run with u = round 32's up0 size direction at a ∈ {−4, −2, −1, 0, 1, 2, 4}, with the same a in all 6 free cells.
   - The a = 0 runs save the activations.
3. **Fit and freeze.**
   - **Probe:** ridge regression from interior tokens to disparity, with λ chosen by leave-one-source-out cross-validation. Probes for the requested blur and the signed gap are fitted for comparison only.
   - **Strength rule:** g(signed gap) is piecewise-linear through the median best a per signed gap. "Best" means the lowest revised error among valid outputs; failing that, the radius closest to r\*.
   - **Fixed strength:** the grid value with the lowest dev median ratio.
4. **D0 (disparity-only control).** Two test scenes are re-run with every gap's sign flipped: the blur map is identical and the depth is mirrored. Reported: pixel change, activation change, and the probe's change.
5. **Test grid.** Each test scene is run at every a. This gives the calibrated request (a = 0), the fixed strength, and a per-cell oracle used as a ceiling. The a = 0 runs save the activations used to test the probe.
6. **Step 3, causal edit.**
   - Test scenes 0–5 are run with w_d (the probe's weight direction, scaled to |u|) and with a random direction of the same norm, at a ∈ {−2, −1, 1, 2}.
   - u itself is covered by the test grid.
7. **Step 4, adaptive strength.** Each test scene is run with three strength maps:
   - **depth map:** a_k = g(gap_k);
   - **proxy:** a_p = g(d̂_p − d_f) per token, with d̂ read from the unedited activation in the same forward pass;
   - **shuffled proxy:** the proxy's per-cell mean strengths permuted across the free cells by a fixed derangement (same strengths, wrong places).

Edits touch the free cells only. Focus and anchors are never edited.

About 255 generations in total.

## Rules (fixed before the run; `analyze36.py`; test scenes only)

**Preservation and validity.** Preservation is the direct pixel change of the focus and anchor interiors relative to the unedited (a = 0) output, and must be ≤ 0.5/255. Further conditions:
- **valid:** |r̂ − r\*| ≤ max(2 px, 20 %);
- **contrast guard:** contrast ≥ the calibrated request's − 0.05;
- **error:** round 34's revised error (offset-only), against a disk blur of the source at r\*.

**A. Does the probe read depth?** Measured on cell means over the 72 test cells.
- **A1:** r(d̂, d) ≥ 0.7 and MAE ≤ 0.10.
- **A2:** residual variance explained by texture ≤ 0.2, and by position ≤ 0.2.
- **A3:** front/behind pairs with the *same requested blur* are ordered correctly in ≥ 29/36 (p < 0.001 under chance).

Reported alongside:
- partial correlations given the requested blur and given the signed gap;
- the calibration slope;
- the blur-request and signed-gap probes' test R².

**B. Does the edit control blur?** Tested for u, w_d and random, at equal norm, on 36 cells.
- **B1:** the sign of r̂(+2) − r̂(−2) agrees in ≥ 80 % of cells.
- **B2:** median |r̂(+2) − r̂(−2)| ≥ 1 px and ≥ 2× the random direction's.
- **B3:** preservation ≤ 0.5/255 at |a| = 2.

The probe's own output is *not* used here, because adding along w_d moves it by construction.

**C. Does adapting the strength help?** Each method is compared with the calibrated request on 72 cells. The ratio is the method's error / the calibrated request's error.
- If the method's output is invalid, or fails the contrast or preservation guard, the ratio counts as ∞.
- If the calibrated request's output is invalid and the method's is valid, the ratio is 0.
- **Gate (smoke-test level, as rounds 29–35):** median ratio ≤ 0.80 and wins ≥ 54/72.
- **"Adaptation helps"** requires all three:
  - the proxy passes the gate;
  - the proxy beats the fixed strength, i.e. it wins > 36/72 cells head-to-head and has the lower median ratio;
  - the proxy beats the shuffled proxy in the same way.

A pass would still need fresh-image confirmation.

**Expected difficulty.** Depth alone is not the correction signal. Distance from the focus plane sets the *desired* blur, which the calibrated request already uses. It does not tell whether BokehDiff over- or under-blurred a region, so correcting that error needs an estimate of the delivered blur too. The oracle row shows how much headroom per-cell strength has at all.

**Smoke (DSMLP, Nov 20; 2 dev scenes, 1 test scene, a ∈ {−2, 0, 2}).**
- G1: round 25 reproduced exactly (0). G2: repeats identical (0).
- 21 generations took about 4 s each, with peak memory 9.5 GB.
- The analysis ran end to end.
- The numbers are not interpreted: there are too few scenes.

Files: `scenes36.py`, `run36.py`, `analyze36.py`, `dsmlp_depth.sh` (`SMOKE=1`: 2 dev scenes, 1 test scene, reduced grid). Shared modules are copied from round 34. Round 32's directions are in `results/directions32.*`.
