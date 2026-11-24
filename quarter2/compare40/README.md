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
     - up to 8 textured background tiles (64 px) with |disparity − focus| ≥ 0.15. An image with fewer than 2 tiles has unmeasurable blur.
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
