# Round 43: transparent subjects. Is the failure in BokehDiff's depth input or in the model?

Branch `q2-transparent43` (from `q2-control42`). Round 42 showed that BokehDiff's blur *amount* can be controlled from
outside the model (a depth-aware input setting). Steering inside the model is only needed for a failure that no input
can fix. Transparent subjects are the strongest candidate: in round 40, BokehDiff kept the garden seen through a glass
(highlights_3) and the shelves seen through an acrylic sign (rb_48) sharp, while a real f/2.0 lens blurs them.

**Why it happens** (`figures/why_depth.jpg`: input, BokehDiff's own depth map, mask): the depth model marks the glass and
the sign as solid near objects, so BokehDiff is *told* that everything inside them is in focus. The question is
whether BokehDiff renders them correctly once the depth input says otherwise.

## Design (frozen before any round 43 image exists)

**Scenes.**
- 16 generated scenes (`prompts43.py`): an empty glass, wine glass, vase, bottle, jar, acrylic sign, teapot, display
  box, carafe, umbrella and similar, each in front of a textured background visible through it. Same generator and
  settings as round 40 (RealVisXL V5.0, seeds 4300-4315).
- 6 real scenes from RealBokeh's test split with a real f/2.0 capture (`rb43.py`): 48 and 129 (acrylic sign), 49
  (dog through a windscreen), 51 and 131 (windows), 116 (glass grinder). Chosen by eye from all 220 test inputs as
  every scene with a clear object and scene content visible through it.
- Depth and masks from BokehDiff's own preparation (Depth Anything V2 + BiRefNet), as in round 40.

**Regions** (`regions43.py`).
- Behind tiles: as round 40, but the depth-gap cutoff is 0.08 instead of 0.15. With 0.15, rb_48 (shelves only
  0.11-0.15 behind the sign) had no tile in round 40.
- See tiles: places where the background is seen through the object. Texture alone also picks the object's own
  features (logos, water, facets), so the script numbers up to 12 candidates inside the object, and the see tiles are
  **chosen by eye on the input sheets before any output is viewed** (`review43.json`). Criterion: keep a candidate
  if, in the input, it mainly shows scene content continuing the surroundings; drop it if it mainly shows the object's
  own features (printed logos, rims, water surface, facets, reflections, labels).
- Rim band: object pixels within 6 px of its edge.

**Conditions** (`bd43.py`; round 40 driver, only the depth map and K change; focus is unchanged):

| condition | depth handed to BokehDiff | renders |
|---|---|---|
| original | as round 40 (object = near) | K for 8 and 12 px (round 42 depth rule), K = 36, K = 64 |
| see-through | inside the object, the surrounding background depth (harmonic fill) | 8, 12 px |
| see-through + outline | as see-through, but the 6 px outline keeps the object's depth | 8, 12 px |

`figures/depth_conditions_dryrun.png` shows the three maps for highlights_3 and rb_48.

**Measure** (`score43.py`).
- rho = detail lost in the see tiles / detail lost in the behind tiles: how much of the surrounding background's
  blur the through-content receives (1 = blurred like the background, about 0.1 = kept sharp).
- Detail is mid-scale (difference of Gaussians, sigma 2-5 px), chosen in the dry run, before any round 43 output.
  BokehDiff removes part of the finest detail everywhere, and dust on a real object's surface sits in the finest band.
  On round 40 images: kept-sharp subjects retain 0.84-0.96, blurred background 0.12.
- rim_keep: outline detail relative to the original condition.
- rho_real: rho on the real f/2.0 capture, i.e. what a real lens does.

**Decisions** (medians over scenes, scene bootstrap; primary 12 px, 8 px reported).
- FAILURE: original rho ≤ 0.3 at 12 px and at K = 64. BokehDiff keeps the through-content sharp, and no setting fixes it.
- If FAILURE, the cause:
  - **INPUT**: see-through + outline rho ≥ 0.7 and rim_keep ≥ 0.8. A corrected depth map fixes it without blurring the
    outline; steering is not needed.
  - **MODEL**: see-through rho < 0.5. Even told the content is far, the model keeps it sharp: a target for steering.
  - **EDGE_TRADEOFF**: see-through works but the outline version does not.
  - **MIXED**: anything else.

**Dry run** (CPU, round 40 images as stand-ins).
- highlights_3's round 40 BokehDiff output: rho 0.10, the failure.
- rb_48's real f/2.0 capture: rho_real 0.84, a real lens does blur the through-content.
- The see-through depth asks for 0.94 of the surrounding blur inside the object.

**Disclosure.** highlights_3 and rb_48 were looked at before the design. The measure and the tile-gap change were
chosen on them (dry run). Neither is among the 16 generated scenes; rb_48 is among the 6 real ones.

**Run.** One GPU job (`dsmlp43.sh`): generate, fetch, prepare, regions, render. Then, on CPU: input-sheet review
→ `review43.json` committed → `score43.py`.

## Result (`results/run/score43.json`; run 2026-11-24/25)

**Scenes scored: 10** (9 generated and rb_48). The see tiles were chosen on the input sheets before any output was
viewed (`review43.json`, committed first). 9 rendered scenes were excluded by that review (mask failures, vases full of
stems, logos, ribbed glass, masks on opaque objects); rb_49, rb_129 and rb_131 were not rendered because the mask did
not select the transparent object.

| condition | rho at 12 px [95% CI] | scenes with rho ≥ 0.7 | outline kept (rim_keep) |
|---|---|---|---|
| original | 0.04 [0.02, 0.13] | 0/10 (10/10 ≤ 0.3) | 1 (reference) |
| original, K = 64 | 0.12 [0.07, 0.22] | 0/10 (8/10 ≤ 0.3) | 1 |
| see-through | 0.99 [0.68, 1.07] | 8/10 | 0.14 [0.07, 0.23] |
| see-through + outline | 0.98 [0.70, 1.05] | 8/10 | 0.22 [0.08, 0.29] |
| real f/2.0 lens (rb_48 only) | 0.84 | | |

8 px gives the same picture. **Pre-set decision: FAILURE: EDGE_TRADEOFF** at both 12 and 8 px.

**Reading.**
- **The failure is real and no setting fixes it.** With its own depth map, BokehDiff keeps the scene seen through a
  transparent object sharp (it keeps 96% of that content's detail loss-free while the background around it is blurred),
  even at the strongest setting.
- **BokehDiff follows the depth map.** Told that the through-content is far, it blurs it like the surrounding background
  (rho 0.99). So the failure starts in the input, not in the model's ability to blur.
- **But one depth per pixel cannot describe a transparent object.** In the see-through conditions the object's own
  structure (rim, cut facets, the wine, the handle, the beer) is blurred together with what is behind it, and the glass
  turns into a ghost (`figures/r43_glass_0.jpg`, `r43_glass_10.jpg`: input, original, see-through, see-through +
  outline). Keeping the depth of a thin outline does not help much (0.22): BokehDiff works on 8x8-pixel latents, and the
  facets and liquid are inside the object, not on its outline. A real lens keeps the object's surface sharp while
  blurring the scene behind it (`r43_rb_48.jpg`, last panel).
- **Neither depth map gives the right picture.** The correct image needs two depths at the same pixel (the glass at the
  focus, the scene behind it far away), which BokehDiff's input cannot express.

**Limits.**
- 10 scenes, only 1 real scene with a reference capture. The see tiles and exclusions were chosen by eye, from inputs only.
- rim_keep partly counts background seen through the edge band, which should blur, so it understates the outline that
  is kept. The visual check above shows the object's interior structure is lost as well.
- rb_48's planned K was clipped at 64, and BokehDiff blurs its background little at any setting (round 40).

**What this means for steering.**
- This is the first failure we have found that no single input map can fix: the input format lacks the second layer.
- A cheap baseline still has to be tried before any steering: render twice (original depth, then see-through depth)
  and blend the two with a matte of the object's own structure. The hard part is that matte, which any method would
  need too.
- A steering method has a clear target only if it can produce the two-layer result from one render better than that
  blend.
