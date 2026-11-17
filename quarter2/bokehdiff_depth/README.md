# Q2 round 18: does BokehDiff blur each depth by the amount it was asked for?

**Question.** When we supply the correct depth and the requested focus, does the editor apply the right amount of blur to each region while preserving the scene?

**Status.** Plan and decision rules fixed before any BokehDiff output exists.

Setup is the same as round 17 (`bokehdiff_smoke/`):
- BokehDiff at commit `38ff937`, unchanged;
- fp16 on the GTX 1080 Ti, through the same input cast;
- 512 × 512 images, one-step inference.

## What BokehDiff is asked for (from its code)

- **Disparity.** The loader rescales each image's disparity to 0–1 (`dataset.py`): d_norm = (d − d_min)/(d_max − d_min). The blur requested therefore depends on the scene's own disparity range, not on absolute distance.
- **Focus.** d_focus = median d_norm inside the subject mask.
- **Blur map.** coc = |d_norm − d_focus| · K/10, sent with the disparity as `disp_coc` (`inference_hf.py`).
- **Where it is used.** Only in the self-attention of `down_blocks.1.attentions.0` (`PISA_attn_processor.py`), on a 32 × 32 token grid (16 px per token at 512²).
  - A source token j reaches a target token i when dist(i, j) · 51.2 < coc_j. Distance is in units where the grid spans 0 to 1, so 1 unit = 31 tokens = 496 px.
  - This is a sharp sigmoid (hardness ln 10⁷ ≈ 16).
- **Nominal radius.** So the attention spreads each source over a disk of radius

  **r_nom = 496/51.2 · coc = 0.969 · K · |d_norm − d_focus| px**

  This is the radius the internal constraint imposes. It is not guaranteed to be the pixel blur of the output, and that gap is what this round measures.
- **Coarse grid.** Neighbouring tokens are 16 px apart, so when r_nom < 16 px the constraint lets a token attend only to itself. The internal constraint cannot express blur that small; any such blur must come from the rest of the network.
- **Randomness.** PISA keeps dropout on at inference (p = 0.1, `train=True`) and samples occlusion rays with `torch.rand`. The driver reseeds before every call.

## Inputs (`scenes.py`)

Three rendered, all-in-focus 512 × 512 scenes with exact depth: fronto-parallel textured planes, i.e. piecewise-constant depth.
- **Textures:** scikit-image's grass, gravel and brick photos, tinted, at native resolution (no resampling blur).
- **Repeats:** each texture appears at two or more depths, so texture cannot explain a blur difference.

| scene | layout | depths z (texture) | focus "near" / "mid" |
|---|---|---|---|
| A | 4 quadrants | 1.0 grass, 1.6 gravel, 3.0 gravel, 8.0 grass | 1.0 / 3.0 |
| B | 2 × 3 grid | 1.2 brick + gravel, 2.5 gravel + grass, 6.0 grass + brick | 1.2 / 2.5 |
| C | card on a split background | 1.0 gravel card; background 4.0 grass, 10.0 gravel | 1.0 / 4.0 |

- **Disparity:** true 1/z.
- **Measured area:** only region interiors, at least 48 px from any depth edge and 24 px from the image border. Occlusion boundaries are a later experiment.

## Runs (`run_depth.py`, `dsmlp_depth.sh`)

**Grid:** 3 scenes × K ∈ {10, 20, 40} × focus ∈ {near, mid} = 18 outputs.
- **Extra outputs:**
  - per scene, K = 0, a no-blur reference through the same model;
  - per scene, K = 20 near with seed 1, a noise floor;
  - one in-process repeat for scene A.

  That makes 25 outputs, all saved as PNG.
- **Maps saved:** every disparity map, blur map, 32 × 32 token blur map and r_nom map entering the model goes to `outputs/q2_bokehdiff_depth/maps/`.
- **Convention check:** BokehDiff's own loader reads scene A (true disparity, mask on the nearest plane, K = 20, its first focal shift). Its maps must match ours: disparity and blur map within 2·10⁻³, input image within 2/255.

## Measurement (`measure.py`)

- **Blur radius r̂:** per region interior, the disk radius (0–48 px, step 0.5, parabolic refinement) that best maps the sharp input onto the output. A per-channel gain and offset are fitted by least squares.
- **Reported alongside:**
  - Gaussian σ̂;
  - r̂ measured against the K = 0 output instead of the input;
  - mean absolute error (MAE) against the K = 0 output;
  - position shift (search ±3 px, sub-pixel refinement).
- **Estimator validation, run first on the pod.** The same scenes are rendered with known per-region disk blur (r ∈ {0, 1, 2, 4, 8, 12, 16, 24, 32, 40}). Then gain/offset changes, noise (σ = 1.5/255) and 8-bit rounding are applied, plus one known (1, −2) px shift.
  - Pass: every |r̂ − r| ≤ max(1 px, 10 %) and the shift is recovered within 0.25 px.
  - Gaussian renders are reported for kernel-shape bias. In the CPU debug run, the disk fit read about 1.7σ.

## Decision rules (fixed now)

**Gate.** If any of these fails, the outcome is **"repair the test"** and nothing is concluded about BokehDiff:
- estimator validation passes;
- the convention check passes;
- the in-process repeat differs by ≤ 1/255;
- no NaN or black outputs.

**Blur amount.** These rules use the 54 region × condition cases that are not on the focal plane: 18 from A, 24 from B, 12 from C. A case "agrees" if |r̂ − c · r_nom| ≤ max(2 px, 20 % of c · r_nom).

| outcome | rule | interpretation |
|---|---|---|
| 1 | ≥ 80 % of all 54 agree with c = 1 | behaves as specified: keep BokehDiff as a baseline; no demonstrated problem here |
| 2 | not 1, but c fitted on scene A alone (least squares through 0) makes ≥ 80 % of the 36 B + C cases agree | consistently too strong or weak; one global calibration fixes it |
| 3 | neither | errors depend on depth, focus or content and one gain does not fix them: a candidate problem, to be confirmed on a few more scenes before building anything |

**Preservation**, reported separately as its own pass rate:
- **In-focus region:** r̂ ≤ r̂(K = 0, same region) + 1 px, and MAE against the K = 0 output ≤ 3/255.
- **Positions:** shift ≤ 1 px in every region.

**Always reported.** Residuals by |Δd|, front versus behind the focus, texture and K; cases at the 48 px cap; the seed-to-seed radius difference.

**Not claimed from this round:**
- that an error, if found, generalizes beyond these three planar scenes;
- anything about occlusion boundaries or real photos.

## Results

Pending.
