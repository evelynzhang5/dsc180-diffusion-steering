# Round 40: Bokehlicious vs calibrated BokehDiff on generated images (diagnostic pilot)

**Question.** On generated images, is the remaining limitation **choosing where and how much to blur** (focus selection, request strength) or **rendering the blur** itself (preservation, boundaries)?
- If the failure is choosing the subject or the request strength, the next step is a spatial predictor. Its first comparison is against the existing calibration.
- If blur is matched but boundaries remain wrong, the next step is a rendering intervention.

This round tests those prerequisites. It does **not** test whether integrating Bokehlicious into diffusion beats post-processing.

**Scope.** Generated-image findings apply to **RealVisXL V5.0**, which is also BokehDiff's own SDXL backbone, until another generator is tested.

## Stages

1. **Stage 1 (pod, no method runs): `gen40.py`, `realbokeh40.py`, BokehDiff's `prepare_data.py`, `regions40.py`.**
   - **Generated scenes:** 40 RealVisXL scenes, 5 categories × 8.
     - Categories: hair, fur, thin structures, highlights, existing blur.
     - The first 2 of each category are development scenes, the other 6 test.
     - All use textured backgrounds.
     - Four categories are prompted all-in-focus. "existing_blur" is prompted with shallow depth of field and is kept out of the radius-accuracy analysis.
   - **RealBokeh:** scene–aperture pairs chosen from the actual metadata.
     - Candidates must have f/2.0 and an aperture in [4, 8].
     - 25 are sampled (seed 0): 5 calibration scenes and 20 test.
     - Targets: f/2.0 and the closest available aperture to 5.6.
   - **Preparation:** BokehDiff's own pipeline, Depth Anything V2 Base disparity plus a BiRefNet salient mask.
   - **Frozen measurement regions, from the originals only:**
     - subject interior (≥ 8 px inside);
     - boundary band (±16 px);
     - up to 8 textured background tiles (64 px) with |depth − focus depth| ≥ 0.15 for at least 90% of the tile's pixels. An image with fewer than 2 tiles has unmeasurable blur.
     - Split by side: up to 8 tiles **behind** the subject (background blur, the primary measure) and up to 8 **in front** (foreground defocus, reported separately), chosen independently. *Revised after the mask review, before any method ran:* ranking all tiles together put most tiles in front of the subject in 11 images (for example hair_7's fence).
     - *Revised after stage 1, before any method ran:* the first rule tested only the tile's median, which admitted tiles up to half covered by in-focus structure (branches and ferns at the subject's depth in fur_3, fur_5, fur_7). Measurable images: 59 → 58 of 65.
   - **Contact sheets** for the mask review.
2. **Mask review (human, before any method output exists).**
   - Masks are judged on the original images, and on the real f/2.0 capture for RealBokeh.
   - Bad masks are flagged, then all masks are frozen for both methods.
   - Flagged scenes are excluded from the rendering-quality comparison, not repaired.
3. **Stage 2 (CPU): validate the boundary measures on synthetic controls.** Using round 39's textured scenes and references, deliberately inject:
   - a bright rim;
   - colour bleeding across the edge;
   - loss of thin structures.

   The measures must respond to each injection and must not penalise the reference's own correct edge transition. Until then, generated-image boundary scores stay **exploratory**.
4. **Stage 3 (pod): both methods.**
   - **Bokehlicious:** small model; f-stop grid.
   - **BokehDiff:** focus = median subject disparity; K grid.
   - **Development scenes:** map each method's setting to a target effective additional blur of 4, 8 and 12 px, then freeze the mapping.
   - **Test scenes:** apply the frozen mapping. Then score, select the blind pairs, and measure latency.

## Measures (frozen in stage 3 code before test runs)

- **Effective additional blur, not physical radius.** A generated input has unknown existing blur. A kernel fitted between input and output measures only the *additional* blur.
  - Fitted only on the preselected background tiles, using the linear-light disk estimator.
  - Fit quality (R²) is reported; poor fits are marked unmeasurable.
  - The existing-blur category is reported separately.
- **Control performance** (all test scenes, including failures): how closely each frozen mapping reaches its target. Reported as |delivered − target| and the share within tolerance.
- **Rendering quality**, only where both methods' delivered blur agrees within a tolerance fixed in advance. The number excluded is reported.
- **Preservation reference: the original input** (not a zero-request output; see round 39). Reported separately inside the subject interior:
  - colour shift (per-channel median offset);
  - detail (gain/offset-invariant residual);
  - sharpness (high-frequency energy relative to the input);
  - artefacts: noise and ringing are checked separately, so that an increase in high-frequency energy is not read as restored sharpness.
- **Focus selection, kept separate from rendering:** whether the sharpest region matches the frozen subject mask.
- **Boundaries:** halo, bleeding and thin-structure loss in the band. Exploratory until stage 2 validates them.
- **Real-photo control (RealBokeh):** error to the real capture in linear light, with each method's setting chosen on the 5 calibration scenes. Bokehlicious was trained on RealBokeh's training split, so it has a home advantage here.
- **Latency:** rendering-only, and total including depth estimation and segmentation.
- **Statistics:** aggregated by scene. The three blur targets on one scene are repeated measurements, not independent examples.
- **Blinded pairs:** 25 shuffled, unlabelled pairs at matched blur, with "tie" and "cannot judge" options. Human rating is optional. One rater makes it a pilot assessment, not a validation of the automatic metrics.

## Stage 2 result (CPU): boundary-measure validation (`validate40.py`, `results/validate40.json`)

**Design.**
- **Correct outputs:** thin-lens references for an in-focus textured foreground over a blurred textured background, at r = 7.75 and 15.5 px. Scenes: 3 cutouts plus a wheel with 2 px spokes, giving 8 cases.
- **Injected defects:** a bright rim, colour bleed, a sharp halo, and thin-structure loss.
- **Rule (frozen):**
  - the measure's own defect must move it, in every case, by at least 2× its largest |value| on the correct outputs;
  - its median shift must be at least 2× the shift caused by each other defect.

| Measure | Max \|value\| on correct outputs | Own-defect shift (min / median) | Other defects' median shift | Verdict |
|---|---|---|---|---|
| halo_sharp | 0.107 | +0.237 / +0.248 | ≤ 0.007 | **passed the specified controls** |
| thin_loss | 0.121 | +0.761 / +0.824 | ≤ 0.096 | **passed the specified controls** |
| rim (max of inner and outer side) | 8.78 | +0.00 / +3.90 | 3.6–6.1 | exploratory |
| bleed | 13.7 | +8.59 / +10.08 | up to 3.5 | exploratory |

**Reading.**
- **halo_sharp and thin_loss passed the specified synthetic controls.** They respond to their own defect, not to the others, and stay near zero on the correct thin-lens output.
- **Scope of that result.** The controls have one sharp foreground over one uniformly blurred background. Generated scenes need not have that layout, so generated-scene readings of these two measures stay conditional on the local geometry (what lies just outside the mask and at what depth).
- **rim and bleed fail because of the correct output itself.** A correctly blurred background changes the input's luminance and colour just outside the contour by up to 9–14/255, depending on the local content, so a reference-free measure cannot tell a defect from correct blur there. Both stay exploratory, and no conclusion will rest on them.
- **Post hoc candidate (not validated):** the inner-side rim alone.
  - It is ≤ 0.36 on all correct outputs and +6.3 to +7.9 under an injected rim, and does not respond to bleed or halo.
  - But it also moves with thin-structure loss, by up to +9.9 on the wheel.
  - It was chosen after seeing these results, so it would need a fresh validation set before any claim rests on it. It is reported, but no conclusion will be drawn from it.

### Addendum (after review, before stage 3): depth-gated halo_sharp

**Problem.** halo_sharp compares sharpness just outside the subject with distant background tiles. Content outside the mask that genuinely lies near the focus plane, such as a branch the subject sits on, is correctly kept sharp, and that would be scored as halo. The stricter tile rule cleans the tiles, not the edge band.

**Change (frozen before stage 3).** When a depth map is given, halo_sharp uses only edge-band pixels with |depth − focus| ≥ 0.15, the tile threshold. It reports how many band pixels remain (halo_px, halo_kept). With fewer than 500 px it is not measurable. Re-running the validation with depth supplied leaves every number in the table above unchanged.

**Control** (`validate40.py`, `branch_control` in `results/validate40.json`): the disk scene plus a sharp horizontal bar at the focus depth, outside the mask, crossing the band on both sides. The thin-lens reference keeps the bar sharp, which is correct, so the right score is ≈ 0.

| Bar texture | Bar half-width (px) | Band covered | Ungated halo_sharp | Gated halo_sharp |
|---|---|---|---|---|
| smooth (rocket) | 3 / 12 / 24 | 2% / 7% / 13% | +0.011 to +0.015 | +0.007 to +0.008 |
| high-detail (grass) | 3 / 12 / 24 | 2% / 7% / 13% | +0.023 / +0.057 / +0.098 | +0.005 to +0.008 |

Values cover both radii (7.75 and 15.5 px), which agree within 0.002.

**Reading.**
- Ungated, a detailed in-focus branch covering 13% of the band scores +0.098. That is close to the largest score of a correct output in validation (0.107) and about 40% of a real injected halo (+0.248). The reviewer's concern is real.
- Gating removes it here. But gating trusts the depth estimate: where Depth Anything places in-focus content off the focus plane, the false positive returns. For example, existing_blur_6's branch is estimated nearer than the bird.
- halo_sharp on generated scenes therefore stays conditional on the depth map, and is reported with halo_kept.

## Mask and region review (frozen in `review40.json`, before any method ran)

One reviewer checked all nine contact sheets visually. That can't certify individual hair or wire pixels. Nothing was repaired; scenes are only excluded or flagged, and all stay in the pipeline-coverage counts.

| Category | Scenes | Treatment in stage 3 |
|---|---|---|
| Mask failure | rb_53, rb_77, rb_125 (empty masks); rb_145 (one interior pixel); rb_72, rb_178, rb_186, rb_197 (mask disagrees with what is sharp in the real f/2.0 capture) | Excluded from every mask-dependent comparison. No generated scene had a clear whole-subject mask failure. |
| Mask caution | rb_62, rb_182 (mask spans different focus levels) | Kept; mask-dependent summaries reported with and without them |
| Measurement flag | existing_blur_6 (depth places the bird's branch nearer than the bird) | Excluded from blur, control and halo results; kept for preservation |
| Metric caution | highlights_3, rb_48 (transparent subjects); thin_5, thin_7 (interior is mostly thicker parts) | Kept; preservation reported with and without them |
| Background blur unmeasurable (< 2 behind tiles) | existing_blur_0, 1, 3, 7; rb_48, rb_62, rb_133; plus the empty-mask scenes | Excluded from blur and control results only |

**Region change made in response.** The reviewer flagged hair_7: its tiles measured the fence in front of the child, not the distant background. The same happened in 10 other images, so tiles are now chosen separately behind and in front of the subject. That leaves 55 of 65 images with measurable background blur, and 19 with measurable foreground defocus, which is reported separately.

**RealBokeh calibration set.** rb_125 and rb_197 were 2 of the 5 calibration scenes. That leaves 3 (rb_12, rb_120, rb_140). No test scene is moved into calibration.

## Stage 3 protocol (frozen before any method output; code: `common40`, `blur40`, `bd40`, `bkl40`, `map40`, `score40`, `lat40`, `dsmlp_stage3.sh`)

**What "blur" means.** Delivered blur is the effective **additional** disk radius, in px at the 768-px input scale. It is fitted between the input and the output in linear light, on the frozen tiles **behind** the subject, with a per-channel gain and offset (the round 18/19 estimator).
- A tile counts only if R² ≥ 0.90 and the fit is not at the grid end. An output's value is the median over counting tiles. With fewer than 2 counting tiles it is unmeasurable.
- Total blur (existing plus added) is not estimated.
- The existing_blur category is reported separately from the initially-sharp categories, and never enters the mapping.
- Tiles in front of the subject give a separate foreground-defocus value; they never enter the mapping or the blur results.

**Mapping (development scenes only).** Each method gets one global setting per target (4, 8, 12 px):
- **Scenes:** the 8 initially-sharp development scenes with measurable blur and no review exclusion. BokehDiff's setting is K, from a grid of 4–64; Bokehlicious's is the f-number, f/2–f/16.
- **Objective:** equal weight per scene. Each scene contributes one value, whatever its tile count, and the curve is the mean over scenes. The frozen setting is where this curve first crosses the target, interpolated (K linearly, f-number in log).
- **Unreachable target:** if the curve never reaches the target, the nearest grid end is used and the target is a **control failure** for that method. Its outputs are then never treated as matched unless the delivered blur itself lands within tolerance.
- **Reported:** every scene's response at every grid setting, and each scene's residual at the frozen setting.

**Test phase.** The frozen settings are applied unchanged. Scoring checks that each recorded setting equals the frozen one, and fails the run otherwise; a tampered setting was confirmed to trigger the failure.

**Control vs quality.**
- **Control:** every test scene × target, with a status: ok, miss, unmeasurable output, no tiles, or excluded by a review flag. Failures are counted, not dropped.
- **Quality** (preservation and boundary): compared only on matched cells, where both methods' delivered blur is measurable and within max(2 px, 20%) of each other. The number excluded is reported.

**Coverage.** 55 of 65 is raw tile availability, not a sample size. Counts are reported for every metric, split, method and category, after the review exclusions (`review40.json`).

**RealBokeh.**
- Settings are chosen on the 3 remaining calibration scenes, by linear-light error to the f/2.0 capture and, separately, to each scene's second aperture.
- Each scene's own calibration curve is reported, because 3 scenes can disagree.
- Test error is reported raw and after a per-channel gain fit (exposure), against the input's own error as a baseline.

**BokehDiff inputs.**
- **Depth convention (finding, frozen before any output).** BokehDiff's `prepare_data.py` stores 1 − normalised(Depth Anything output), so larger = **farther**; the contact sheets show subjects dark. But BokehDiff was trained on disparity with larger = **nearer** (`dataset.py`: disp = 1/depth), and its attention processor samples "from the point to the camera (disp=1)".
  - The blur map |d − focus| is identical either way; the occlusion ordering inside the attention is reversed.
  - Main runs use the training convention (1 − stored map). The 8 mapping scenes are also run at K = 20 with the stored, upstream convention, as a reported diagnostic only.
- **Processing size.** As upstream, the short side is ≥ 1024, then the output is resized back. The size is rounded to a multiple of 64 (upstream: 8), because the attention recovers its down-sampling factor from the depth map's size, and 4:3 images would otherwise mismatch.

**Latency.**
- Rendering-only time is the model call, synchronised.
- BokehDiff's total adds Depth Anything V2 Base plus BiRefNet, timed on the GPU with models loaded. Upstream `prepare_data.py` runs depth on the CPU, which would be slower.
- Bokehlicious needs neither, so its total equals its rendering time.

**Dry run (CPU, fake outputs).** The mapping and scoring were run on synthetic outputs with known disk blur (r = K/4 and r = 24/f).
- The mapping recovered K = 15.91 / 31.97 / 47.99 (true 16 / 32 / 48) and f/6.07 / 3.03 / 2.00 (true f/6 / 3 / 2).
- All checks passed: settings, behind-only tiles, exclusions.
- Smoke-run checks on the pod: exclusions applied, front tiles never in calibration, frozen settings reused.

**Blinded pairs.** 25 matched cells from initially-sharp test scenes, excluding mask failures. Seed 0, sides shuffled. Each pair is rated left / right / tie / cannot judge. The key is written outside the results folder.

This compares two editing pipelines on generated images. It does not test whether integrating Bokehlicious into diffusion beats post-processing.

## Stage 3 result (pod run 2026-11-24, log `q2-r40b-20261124-121028.txt`; `results/stage3/`)

All four scoring checks passed:
- test settings equal the frozen mapping;
- every blur tile is behind the subject;
- mask failures are out of the mask-based metrics;
- existing_blur_6 is out of the blur, control and halo results.

Outputs (not in git) are in `outputs/q2_compare40/s3`. The numbers below are from **24 initially-sharp generated test scenes** unless stated otherwise. Paired differences are taken per scene, as BokehDiff − Bokehlicious.

### Frozen mapping (8 development scenes, `mapping.json`)

| target | BokehDiff K | mean abs. residual | within tol. | Bokehlicious f-number | mean abs. residual | within tol. |
|---|---|---|---|---|---|---|
| 4 px | 14.9 | 3.2 px | 1/8 | f/9.5 | 2.7 px | 1/8 |
| 8 px | 21.2 | 4.8 px | 1/8 | f/5.3 | 5.0 px | 1/8 |
| 12 px | 26.9 | 5.7 px | 1/8 | f/3.1 | 6.3 px | 1/8 |

- No target was a mapping control failure: every target lies inside both mean curves.
- One global setting still fits almost no individual scene.
  - BokehDiff at K = 21 delivers 2.4 px (thin_0) to 15.4 px (highlights_0).
  - Bokehlicious at f/5.3 delivers 0.3 px (thin_1) to 15.1 px (fur_0).
  - Bokehlicious hardly blurs thin_1 at any f-number (≤ 1.1 px).
- Per-scene curves are in `mapping.json`.

### Control on test (status counts out of 24 scenes)

| target | BokehDiff ok / miss / unmeasurable | median abs. error | Bokehlicious ok / miss / unmeasurable | median abs. error |
|---|---|---|---|---|
| 4 px | 6 / 18 / 0 | 2.6 px | 8 / 16 / 0 | 3.3 px |
| 8 px | 6 / 18 / 0 | 4.6 px | 6 / 16 / 2 | 5.5 px |
| 12 px | 5 / 19 / 0 | 4.9 px | 3 / 13 / 8 | 4.9 px |

- **Delivered blur at the frozen settings (BokehDiff, min / median / max):** 0.9 / 1.9 / 9.4 px at target 4; 1.2 / 6.4 / 16.2 at target 8; 1.9 / 9.6 / 19.4 at target 12.
- **Existing_blur scenes** (separate group; 2 have no tiles and 1 is flagged): at most 1 ok per cell. No matched cells.
- **Foreground defocus** (7 scenes with front tiles): median 2.0 px for BokehDiff and 2.4 px for Bokehlicious. It is not used anywhere else.

### Focus selection

The check is whether the subject interior keeps more high-frequency detail than the far background (see Measures). This is a weak criterion: a subject that is visibly softened can still pass it.

| | passes | subject high-frequency retention (all cells, scene median) |
|---|---|---|
| BokehDiff | 64 of 72 cells | 0.63 |
| Bokehlicious | 51 of 72 cells | 0.24 (BokehDiff higher in 21 of 24 scenes) |

Bokehlicious chooses its own focus. In many outputs it softens the subject along with the background: for example hair_5 at 12 px is a matched cell, but the face is blurred. BokehDiff is given the mask, so its subject stays sharp.

Preservation (all cells; scene medians, excluding the cautions):

| | colour shift | high-frequency retention |
|---|---|---|
| BokehDiff | 0.67 | 0.62 |
| Bokehlicious | 1.67 | 0.21 |

### Rendering quality at matched blur

- **Coverage:** 21 of 72 cells are matched, from 12 scenes. Their quality is compared below; the other 51 are not.
- **Without cautions (11 scenes):**

| metric | BokehDiff | Bokehlicious | scenes where BokehDiff is higher |
|---|---|---|---|
| subject high-frequency retention | 0.66 | 0.25 | 10 of 11 |
| halo_sharp (higher = sharper band outside the subject; depth-gated) | 0.22 | 0.06 | 8 of 11 |
| thin_loss | | | 6 of 11 (no consistent difference) |

- **Post hoc, cells where both methods pass the focus check** (15 cells, 10 scenes): BokehDiff's halo is still higher in 8 of 10 scenes (median difference +0.18).
  - Caveat: Bokehlicious softens the whole subject region, so its band is soft too. The halo comparison is therefore not like for like.
  - What it does show is that BokehDiff, with the subject held sharp, leaves a sharp ring of background around the subject. The ring is visible in the hair_5 matched cell (left shoulder).
- rim, rim_inner and bleed are exploratory and their signs are mixed. They are kept in `score40.json`.

### RealBokeh (20 test scenes; linear-light MAE in units of /255; median over scenes)

| aperture | Bokehlicious raw / gain-corrected | Bokehlicious beats the input | BokehDiff raw / gain-corrected | BokehDiff beats the input | input |
|---|---|---|---|---|---|
| f/2.0 | 7.9 / 5.9 | 18 of 20 | 9.8 / 9.2 | 16 of 20 | 11.5 |
| second aperture | 3.6 / 3.4 | 19 of 20 | 7.6 / 7.2 | 7 of 20 | 6.0 |

- **Calibration on 3 development scenes:** BokehDiff K 20 (f/2.0) and K 12 (second aperture); Bokehlicious f/2.8 and f/5.6. The per-scene optima disagree (BokehDiff K 16–28 at f/2.0).
- **Training data:** Bokehlicious was trained on RealBokeh's training split, so this test is in-domain for it and not for BokehDiff.

### Latency (GTX 1080 Ti, median per image)

| | rendering | total |
|---|---|---|
| BokehDiff | 11.86 s | 12.65 s (adds depth 0.09 s and segmentation 0.70 s) |
| Bokehlicious | 0.19 s | 0.19 s |

### Diagnostic: depth convention (8 mapping scenes, K = 20)

The upstream convention gives the same or slightly less blur than the training convention:

| | min | median | max |
|---|---|---|---|
| blur, training convention | 1.8 px | 4.9 px | 14.2 px |
| blur, upstream convention | 1.8 px | 4.3 px | 12.7 px |

The mean pixel difference between the two outputs is 0.6–1.4 /255. The choice of convention does not change any conclusion here.

### Post hoc: why one global K misses (`post40_gap.py`; written after the results, not part of the frozen protocol)

- **Model:** BokehDiff's delivered blur ≈ 0.93 × K × gap. The gap is the median |normalised depth − focus| over the behind tiles, using BokehDiff's own preprocessing.
- **Fit on development scenes** (K ≥ 12, 46 points): R² = 0.81.
- **Held-out test outputs:** correlation 0.88, median absolute prediction error 2.2 px, and 34 of 72 predictions within tolerance of the delivered blur.
- **Reading:** this is what a fixed aperture does physically. A px target per scene needs a per-scene K, which is roughly target / (0.93 × gap). That rule was **not tested** here; it would need its own frozen run.

### Reading (diagnostic pilot; generated test scenes; conditional on the frozen masks, tiles and measures)

- **Choosing how much blur.**
  - Neither method delivers a specified pixel blur with one global setting: control ok in 3–8 of 24 scenes per target.
  - For BokehDiff the scene-to-scene spread is largely the depth gap. That is a request-conversion problem (setting K per scene from depth), not evidence that BokehDiff renders the wrong amount for the depth it is given.
- **Choosing where to blur.**
  - Bokehlicious frequently softens the subject: 21 of 72 cells fail the focus check, and subject detail retention is 0.24 vs 0.63.
  - With mask and depth supplied, BokehDiff mostly keeps the subject (64 of 72).
- **Rendering at matched blur (12 scenes).**
  - BokehDiff's main remaining defect is a sharp halo of background around the subject (8 of 11 scenes).
  - Bokehlicious has a lower halo, but at the cost of subject sharpness.
- **Mapping to the stage 3 decision logic:**
  - Generated-image failures are mainly in the request: how much (both methods) and where (Bokehlicious).
  - For BokehDiff, with where and how much supplied, the defect at boundaries is rendering: the halo.
  - So the next steps are (a) a per-scene, depth-based conversion from target to K, as its own frozen test, and (b) a boundary/halo rendering intervention for BokehDiff.
- **Limits.**
  - Small n: 8 calibration scenes, 12 scenes with matched cells, 3 RealBokeh calibration scenes.
  - The 21 blinded pairs (`results/stage3/pairs`; only 21 matched cells existed, so fewer than the planned 25) are not yet rated. The pair PNGs (31 MB) are kept out of git; only `ratings.csv` is committed.
  - This compares two editing pipelines and does not test integrating Bokehlicious into diffusion.
