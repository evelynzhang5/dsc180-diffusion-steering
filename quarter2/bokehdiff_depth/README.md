# Q2 round 18: does BokehDiff blur each depth by the amount it was asked for?

**Question.** When we supply the correct depth and the requested focus, does the editor apply the right amount of blur to each region while preserving the scene?

**Status.** Run Nov 17 (pod `q2-depth`, about 4 min). The test's own checks all passed. **Outcome 3 by the pre-set rule: BokehDiff's blur amounts do not follow the requested amounts, and one global gain does not fix them.** The plan and rules below were committed (a6203b2) before the run.

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

Files:
- `results/depth.json`: every region × condition, plus gate and decision;
- `results/estimator_validation.json`;
- `results/depth_sheet.png`: rows = scenes A/B/C; columns = input, K 0, near K 10/20/40, mid K 10/20/40.

Outputs and maps are in `outputs/q2_bokehdiff_depth/`.

### Gate: passed

| check | result |
|---|---|
| estimator on known disk blur | all within tolerance, worst error 0.25 px; known (1, −2) px shift recovered |
| BokehDiff's loader vs our maps | disparity within 1.4·10⁻⁴, blur map within 2.0·10⁻⁴, input image identical |
| in-process repeat | identical (max difference 0) |
| outputs | no NaN or black images; 9.6 GB peak GPU; about 1.7 s per image |
| seed 0 vs seed 1 | radius differs by ≤ 0.42 px (mean 0.17) |

### Blur amount: outcome 3

| rule | result | needed |
|---|---|---|
| agree with r_nom (c = 1), all 54 cases | 26 / 54 = 48 % | ≥ 80 % |
| one gain fitted on scene A (c = 1.43), held-out B + C | 16 / 36 = 44 % | ≥ 80 % |

By eye the blur is clean and sorted by depth (sheet): the focused plane stays sharp, and the further a plane is from the focus, the blurrier it gets. The error is in how much blur each plane gets.

Error patterns. These are descriptive, from the same 54 cases; the frozen rule does not depend on them.

1. **Small requests are ignored.** All 8 cases with r_nom < 5 px (1.6–4.6 px) read 1.0–2.1 px, the same as the K = 0 output's own softness (0.9–2.1 px; this is the round trip through the image encoder and decoder). Some blur appears from about 6 px. One possible cause is the 16 px token grid of the attention constraint (see above). But the rest of the network also shapes the output, and the threshold is lower than 16 px, so the plateau alone does not identify the cause.
2. **Large requests overshoot, more so as they grow.** Median measured/requested radius, uncapped cases with r_nom ≥ 6 px:

   | K | in front of focus | behind focus |
   |---|---|---|
   | 10 | 0.84 (n 4) | 1.12 (n 8) |
   | 20 | 1.20 (n 5) | 1.47 (n 11) |
   | 40 | 1.13 (n 5) | 1.53 (n 8) |

   A further 5 cases at K = 40, all behind the focus, are **above range** (the estimator stops at 48 px; 48 is not an exact measurement), so the K = 40 "behind" ratio is an underestimate.
3. **Behind the focus gets more blur than in front, at the same requested radius.** Under the paper's ideal thin-lens model, equal absolute disparity differences (here equal |d_norm − d_focus| at the same K, i.e. equal r_nom) give equal nominal radii inside surfaces. Equal distances in metres would not. This asymmetry is confounded with focus position: most "behind" cases come from near-focus runs. Within mid-focus runs alone it is suggestive but mixed (e.g. scene B, K = 40: front 1.02 and 1.13, behind 1.68 and 1.46).

**Exploratory (decided after seeing the data, not a frozen test).** A monotone curve from requested to measured radius, fitted on scene A, gets 19 / 33 = 58 % of held-out uncapped cases within tolerance. Its held-out residuals differ by side: in front −36 % (median), behind −18 %. So a 1-D correction of the radius may not be enough either, but this is post hoc and small.

### Preservation

- **In-focus planes stay sharp:** the measured radius stays at the K = 0 level, within +1.1 px, in all 24 cases.
- **Their detail differs from the K = 0 output:** MAE against it is 1.9–6.9/255. Only 8 / 24 meet the ≤ 3/255 rule. Very little of this is a colour shift: after a per-channel gain and offset it is still 1.8–6.0/255. The change grows with K and the amount of blur elsewhere in the image, and it is texture-dependent: gravel and grass change more than brick.
- **Against the original input (added after review, Nov 17):** in-focus planes differ from the input by 3.0–13.4/255. The K = 0 output's own error on the same planes is 2.9–13.2/255; this is reconstruction error from the image decoder, which the paper acknowledges. The extra error with blur elsewhere is small: median +0.04 / +0.15 / +0.22 per 255 at K = 10 / 20 / 40, at most +0.78.
- **So the 2–7/255 differences from the K = 0 output are mostly a different reconstruction, not extra damage.** Blur elsewhere changes the decoder's fine-texture errors without moving them further from the input. The ≤ 3/255 rule measured against the K = 0 output therefore overstates blur-induced damage.
- **Positions:** 85 % of cases have shift ≤ 1 px. Every case above 1 px has r̂ ≥ 27 px, where the texture is mostly erased and the shift search is unreliable. In the sheet, no plane visibly moves.

### Interpretation (narrow)

- Under the conditions tested, the editor's blur is ordered correctly across depth, but its amount is not proportional to the request:
  - nothing below about 5 px;
  - growing overshoot above about 10 px;
  - possibly more behind the focus than in front.
- By the rule fixed in advance, this is a **candidate problem, to be confirmed on a few more scenes** before building any correction.
- "Requested" here means r_nom, the radius the internal attention constraint imposes. The authors may intend K as a relative aperture with a different pixel scale. However, that cannot explain the result: a free global gain fails as well, so the problem is the shape of the relationship, not its scale.

Limits:
- three synthetic planar scenes;
- disk-equivalent radius: a Gaussian-like output kernel would read differently (Gaussian σ̂ is reported in `depth.json`);
- interiors only, no occlusion boundaries;
- radius above 48 px not measurable.

### Suggested next step (not started)

A confirmation round that also separates the confound:
- a few more scenes, including one or two real all-in-focus photos with measured depth;
- the same requested radii placed both in front of and behind the focus within a single run;
- a finer K grid around the 5 px threshold.

The questions, each with its own rule fixed in advance:
1. Does the measured radius depend only on the requested radius? If so, a fixed 1-D calibration curve suffices, and there is no steering problem.
2. Does it also depend on side, focus or content? If so, that is the targeted failure for a steering correction.
