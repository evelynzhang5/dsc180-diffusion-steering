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

## Running

`bash quarter2/real_bokeh/dsmlp_real_bokeh.sh` on a DSMLP GPU pod (setup, select, subsetA, subsetB).

## Status

- Nov 15, 2026: design frozen, code written; launched as `q2-real-bokeh`.
