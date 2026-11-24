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
