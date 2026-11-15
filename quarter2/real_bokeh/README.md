# Q2 round 15: where do single-depth and layered bokeh renderers fail on real photos?

Branch `q2-real-bokeh`. Started Nov 15, 2026. The index of all Q2 rounds is in [../README.md](../README.md).
A diagnostic that should locate an unsolved failure before any method is built.

**Why.**
- The Nov 15 review proposed that single-image bokeh editors fail because they assume one depth per pixel ("2.5D").
  Idea A was to fix this with ordered RGBA + depth layers and an inpainted hidden background.
- But MPIB (ECCV 2022) already does exactly that, and its code and checkpoints are public.
- So the open questions are: how large is the 2.5D failure on real photos, and does a layered method already fix it?
- The follow-up review split this into two subsets:
  - **A, thin occluders:** layered methods should win, except where matting breaks.
  - **B, glass and reflections:** every current framework should fail, because opaque front-to-back compositing
    cannot represent an additive ray I = I_trans(z1) + I_refl(z2).

**Data.**
- **EBB!** (Everything is Better with Bokeh; AIM 2020 challenge, ETH): about 4.8K real pairs from a Canon 7D, 50 mm,
  f/16 (input) against f/1.8 (target).
  - The pairs were aligned by SIFT + RANSAC but are not pixel-exact, so a 1 px shift floor is reported.
  - No licence is stated. Used for research evaluation only, not redistributed.
- **Not available here:**
  - The real focal stacks of Learning to Refocus (SIGGRAPH Asia 2025): browser-only, client-side-encrypted download.
  - AnyBokeh (needs about 27 GB of GPU memory and gated FLUX).
  - FlashReFocus (no code found).

**Scenes (`select`).**
- CLIP ranks the first 1500 pairs against five prompts: hair, branches, fence, foliage, glass. It takes the top 8 per
  category (no repeats), plus 20 random controls.
- `selection_check.png` (5 per category) is checked by eye after the run, and categories that CLIP got wrong are
  reported.

**Renderers (`subsetA`).** All use MPIB's own renderer, kernels and gamma 2.2, with one focus disparity and one blur K
per scene:

| renderer | what it is |
|---|---|
| `gather` | classical single-depth: each pixel takes the blur of its own depth bin, no occlusion |
| `hard` | layered scatter with one depth per pixel (one-hot layers): 2.5D with occlusion |
| `soft` | MPIB's learned soft layers, which extend behind occluders, without inpainting |
| `mpib` | full MPIB: soft layers plus a LaMa-inpainted hidden background |

- Focus and K are fitted to the real target with `hard` at half resolution (an 8 x 6 grid), then shared by every
  renderer. This favours `hard` slightly, so it is conservative against `mpib`.
- Depth: Depth Anything V2 Small, normalised per image.
- MPIB's cupy splatting and numba helpers are rewritten in PyTorch; the splatting is checked against the original
  formula to 3e-7. The only other change: its encoder is built without ImageNet weights, since the checkpoint replaces
  them.

**Measures**, against the real photo, by region from the input's disparity and the fitted focus:
- **outer band:** the defocused side within 12 px of the in-focus boundary, where foreground colour bleeds. This is
  the layer-leakage score.
- **inner band:** the in-focus side within 6 px.
- **thin near structures:** disparity top-hat, narrower than 9 px.
- **background interior:** more than 24 px from the boundary.
- MAE x 255 per region, LPIPS (AlexNet) in the boundary bands and over the whole image, and PSNR.

**Decision, thin scenes (frozen):**
- **No dominant 2.5D failure.** MPIB's outer-band error is at least 10% below the better of `gather` and `hard` in
  fewer than 60% of scenes. Boundary error on these real pairs is then not mainly a 2.5D problem; depth error and
  misalignment are the candidates.
- **Layering helps, and a residual remains.** It wins in 60% or more, and MPIB's outer-band error is still at least
  1.5x its background-interior error. There is then room beyond MPIB on thin structures, such as matting or
  frequency-bounded consistency.
- **Layering already fixes it.** It wins in 60% or more, and the residual is below 1.5. Idea A then has no gap on this
  data.

**Subset B (`subsetB`): glass.**
- EBB! keeps one fixed focus, so a reflection-to-scene focus shift cannot be measured on it.
- So subset B is synthetic, with exact truth:
  - inside a window, I = scene x 0.7 + reflection x (0.25-0.45);
  - two targets: focus on the scene (reflection blurred) and focus on the reflection (scene blurred).
- `gather` and `hard` get oracle-fitted focus and K per target; `mpib` uses `hard`'s settings.
- The additive two-depth oracle is exact by construction. It shows what the representation could reach; it is not a
  method.
- **Decision (frozen):** if the best single-image renderer stays at 70% or more of the unedited input's error on
  either target, current frameworks cannot represent the case. That would be the representational gap the review
  predicted, still to be confirmed on real focus-shift data.
- EBB!'s real glass scenes are scored in subset A at their fixed focus.

## Results (Nov 15, 2026; files in [results/](results/))

Ran on a GTX 1080 Ti, 09:08-09:26, including the 3.8 GB download. Every scene ran; every check passed.
- **Data:** EBB! has 4694 training pairs; the first 1500 were scanned.
- **Checkpoints:** mpinet 60 MB, inpnet 136 MB, aupnet 0.18 MB (a small network; the file is complete).

**Selection check (`selection_check.png`, by eye):**

| category | right | notes |
|---|---|---|
| foliage | 5/5 | |
| branches | 4/5 | |
| fence | 4/5 | a chain-link fence, railings, a gate, a swing grid |
| hair | about 3/5 | people, but no flyaway or backlit hair; one poster and a tree trunk |
| glass | about 2/5 | mostly shiny metalware and posters |

The "glass" group is therefore not a glass test.

### Subset A (real EBB! pairs): no dominant 2.5D failure; plain layered scatter is as good as MPIB

Thin-structure scenes (32), MAE x 255 against the real f/1.8 photo:

| method | outer band (leakage) | inner band | thin | background interior | boundary LPIPS |
|---|---|---|---|---|---|
| input (no edit) | 17.60 | 15.62 | 25.70 | 18.90 | 0.250 |
| gather (single depth, no occlusion) | 17.58 | 15.16 | 22.29 | 12.99 | 0.234 |
| **hard (one depth per pixel, layered with occlusion)** | **14.79** | 15.60 | 23.41 | 13.09 | **0.212** |
| soft (MPIB layers, no inpainting) | 17.92 | 15.16 | 22.48 | 13.25 | 0.252 |
| mpib (full) | 15.51 | 15.68 | 23.11 | 13.18 | 0.220 |
| 1 px shift of the target (floor) | 5.24 | 8.44 | 12.30 | 2.20 | 0.028 |

**Decision (frozen rule):**
- MPIB beat the better single-depth renderer by at least 10% at the outer band in **0 of 32** thin-structure scenes.
- **No dominant 2.5D failure on these real pairs.**

What the run shows:
- **Occlusion handling matters; MPIB's soft layers do not add to it here.**
  - The layered one-depth-per-pixel scatter (`hard`) cuts leakage against `gather` (14.8 vs 17.6; at least 10% lower
    in 16 of 32 scenes).
  - MPIB's soft layers alone are *worse* than `hard` (17.9).
  - Its inpainting brings them back to 15.5, still above `hard`.
- **The error does not come from the representation.**
  - Every renderer leaves the in-focus side as wrong as the input (about 15.6).
  - Boundary errors stay about 3x the 1 px misalignment floor.
  - The examples show why: the real targets differ from the inputs in exposure and alignment, and their blur is mild
    (35 of 60 scenes fitted the smallest K, 8 at 512 px).
- **Controls and the "glass" group show the same pattern.**

Caveats:
- **The fit favours `hard`:** focus and K were fitted with it. MPIB at its own best settings might do a little better.
- **EBB! cannot test the partial-occlusion case.** Its subjects are in focus and there are no defocused foreground
  occluders. The run therefore tests only colour bleeding into a defocused background.
- Depth is monocular (Depth Anything V2 Small).

### Subset B (synthetic glass): a representational gap, as predicted

| target | input (no edit) | gather | hard | mpib | additive oracle |
|---|---|---|---|---|---|
| focus on the scene behind the glass (blur the reflection) | 10.66 | 11.57 | 11.50 | 11.71 | 0 |
| focus on the reflection (blur the scene) | 21.02 | 11.86 | 12.53 | 12.20 | 0 |

**Decision (frozen rule):** the best renderer reaches 1.07x the input's error on the first target (worse than doing
nothing) and 0.56x on the second. With one depth per pixel inside the window, single-depth and layered renderers can
blur everything there, but cannot blur the reflection while keeping the scene behind it sharp.

**Limits:**
- The images are synthetic, and the gap follows from the additive model used to build the targets.
- The test covers only *explicit* renderers. Diffusion refocusers (Learning to Refocus, AnyBokeh, Bokeh Diffusion) do
  not composite layers and might handle reflections implicitly. That is untested: they need 20-27 GB of GPU memory.
- Real focal stacks with a reflection-to-scene focus shift are needed to show the gap matters on real photos.

## Running

`bash quarter2/real_bokeh/dsmlp_real_bokeh.sh` on a DSMLP GPU pod (setup, select, subsetA, subsetB).

## Status

- Nov 15, 2026: design frozen, code written; launched as `q2-real-bokeh`.
- Nov 15, 2026: ran; results above. Closed.
