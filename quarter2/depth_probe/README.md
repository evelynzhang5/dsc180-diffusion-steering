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
5. **Test grid.** Each test scene is run at every a. This gives the calibrated request (a = 0), the fixed strength, and a per-cell best-of-sweep reference (each cell's best outcome among the uniform-strength images; not a bound on jointly chosen strengths). The a = 0 runs save the activations used to test the probe.
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

**Expected difficulty.** Depth alone is not the correction signal. Distance from the focus plane sets the *desired* blur, which the calibrated request already uses. It does not tell whether BokehDiff over- or under-blurred a region, so correcting that error needs an estimate of the delivered blur too. The best-of-sweep row shows how much the tested uniform-strength sweep can offer per cell (not a bound on adaptive rules).

**Smoke (DSMLP, Nov 20; 2 dev scenes, 1 test scene, a ∈ {−2, 0, 2}).**
- G1: round 25 reproduced exactly (0). G2: repeats identical (0).
- 21 generations took about 4 s each, with peak memory 9.5 GB.
- The analysis ran end to end.
- The numbers are not interpreted: there are too few scenes.

Files: `scenes36.py`, `run36.py`, `analyze36.py`, `dsmlp_depth.sh` (`SMOKE=1`: 2 dev scenes, 1 test scene, reduced grid). Shared modules are copied from round 34. Round 32's directions are in `results/directions32.*`.

## Results (Nov 20, 2026; pod `q2-r3536`, 11 min, 255 generations)

**Checks passed.** G1: round 25 reproduced exactly (0). G2: repeats identical (0, dev and test).

**Decisions: A (reads depth) NO · B (controls blur) YES for u and for w_d · C (adaptation helps) NO.**

### A. The inner product does not read depth

| Test-cell measure | Value | Rule |
|---|---|---|
| r(d̂, d) / MAE | 0.11 / 0.25 | ≥ 0.7 / ≤ 0.10 → **fail** |
| Same-blur front/behind pairs ordered correctly | 25 / 36 | ≥ 29 → **fail** |
| Partial r given requested blur / given signed gap | 0.16 / 0.02 | reported |
| Residual variance by texture / position | 0.00 / 0.08 | ≤ 0.2 → pass, trivially: the predictions barely vary |
| Probe for the *requested blur* (|gap|), test R² | **0.93** | reported |
| Probe for the signed gap, test R² | 0.006 | reported |

- **No generalisable disparity signal even on development data.** Cross-validation chose the strongest shrinkage on the grid (λ = 10⁶), where the CV error equals the label variance (0.083 vs 0.083). The probe effectively predicts a constant.
- **The blur request, by contrast, is read almost perfectly** (λ = 10³, CV error 0.0006 vs variance 0.0067).
- **D0 (disparity flipped, blur map identical):**
  - up0 features change by 4–7 %;
  - the output changes by 0.1–0.9/255;
  - the delivered radius changes by ≤ 0.5 px;
  - the probe's output moves by ≤ 0.002.

  Disparity reaches the network only through PISA's occlusion map, and the effect is small.
- **Reading:** at up_blocks.0 the activations carry the requested blur, not depth. A "depth" readout here would be a blur-request readout. The 25/36 pair ordering (one-sided binomial p ≈ 0.014) is a weak trace of occlusion, far below the rule.

### B. Edits along u and along w_d both control blur

| Direction (equal norm, a = −2 → +2) | Cells with consistent sign | Median radius change | Max preservation change at |a| = 2 | Median contrast change |
|---|---|---|---|---|
| u (round 32 size) | 36/36 | +1.85 px | 0.50 (at the guard) | 0.05 |
| w_d (probe weight) | 35/36 | −1.13 px | 0.28 | 0.02 |
| random | 23/36 | 0.08 px | 0.20 | 0.00 |

- **w_d controls blur** even though the probe does not read depth. cos(w_d, u) = −0.25, so w_d partly overlaps the blur-amount direction. Its effect (−1.13 px) is larger than that overlap alone would give (about 0.46 px), so other blur-related components also contribute.
- **Conclusion:** passing B says the direction is a blur knob. It says nothing about depth.
- **Strength limits:** at a = ±4, u changes the focus and anchors by 0.49–1.03/255, past the guard in 23 of 24 runs. Strength is limited to about |a| ≤ 2 (≈ ±1 px).

### C. Adapting the strength does not help

Each method's error is compared with the calibrated request's on 72 test cells. The calibrated request is valid in 66 of them.

| Method | Median ratio | Wins | Invalid | Gate (≤ 0.80, ≥ 54) |
|---|---|---|---|---|
| Fixed (dev chose a = 0, i.e. the calibrated request itself) | 1.000 | 0 | 0 | fail |
| Depth map g(gap) | 0.999 | 36 | 16 (7 radius, 9 contrast) | fail |
| Proxy g(d̂ − d_f) | 1.026 | 27 | 4 | fail |
| Shuffled proxy | 1.025 | 27 | 4 | fail |
| *Best of the uniform-strength sweep, per cell (reference only)* | *0.954* | *61* | 0 | *fail* |

Head-to-head:
- proxy vs fixed: 27–45;
- proxy vs shuffled: 30–35 (7 ties);
- depth map vs fixed: 36–36.

**Reading.**
- **The dev-fitted rule is noise.** The best dev strengths per signed gap scatter across −4…+4: medians −1.5…+2, with no monotone pattern. The fixed strength that won on dev was a = 0.
- **The proxy behaves like the shuffled proxy.** Its depth readout is nearly constant, so its strengths depend only on the focus setting, and shuffling them across cells changes almost nothing.
- **The tested strength sweep offered only a small improvement:** picking each cell's best outcome among the uniform-strength images improves the median error by 4.6 %. This is not a bound on adaptive rules: those images apply one strength to all cells at once, and strengths chosen jointly per cell were not swept. (Corrected Nov 21; an earlier version called this an oracle ceiling.)
- **The remaining error is not a radius error the knob can fix.** The calibrated request already gets the radius right in 66/72 cells. As rounds 32–35 found, what remains is shape and detail, which a one-dimensional radius knob cannot correct.

### Overall
1. At the preselected layer, the activations encode the blur request, not depth. Depth is nearly invisible to the UNet apart from occlusion.
2. Directions that change blur exist, including the probe's own weight direction, so B holds. That does not make them depth readouts.
3. Depth-based or proxy-based strength adds nothing over the calibrated request. As expected, depth sets the *desired* blur, which the request already supplies; it does not tell how BokehDiff missed it.

Files:
- `results/depth_probe.json`: all measures and per-cell ratios;
- `results/frozen36.json`: probe λ and CV, and the strength rule;
- `results/directions32.npz` (force-added);
- images, features and probe weights in `outputs/q2_depth_probe/` (gitignored).
