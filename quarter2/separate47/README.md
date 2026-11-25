# Round 47 (design, not built): does learned layer separation fix transparent content?

Branch `q2-separate47` (from `q2-feature46`). Round 46: with the same information, mixing BokehDiff's internal features
does not beat blending its outputs (the internal features are too coarse for thin surface detail), and even the output
blend stays far from the reference (pane error 8.5/255 vs about 1.3 outside the pane). Neither render contains a clean
background or a clean surface layer. If the composite is first split into its layers, the problem becomes easy:

    separate:  I  ->  alpha_hat, F_hat (surface), B_hat (background seen through)
    render:    BokehDiff(B_hat, the background's own depth)       (same map, focus, K, seed as round 46)
    composite: out = alpha_hat * F_hat + (1 - alpha_hat) * render  (linear light)

With perfect separation this equals round 46's reference by construction (error 0). The question is how much of the
gap a practical separator closes. The whole method stays outside BokehDiff (before and after rendering).

## Question

On held-out synthetic composites, does "separate, then render, then composite" come closer to the controlled reference
than the output blend? Does it hold for surface patterns of a held-out family? Does it look right on a real glass photo?

## Design (to be frozen before training)

**Separator.**
- A small U-Net (about 8M parameters), fully convolutional.
- Input: the sRGB composite plus the pane-region mask. The region mask is privileged and labelled; in practice it would
  come from a transparent-object segmenter.
- Outputs: alpha_hat (sigmoid), F_hat and B_hat (RGB, linear light) at full resolution.
- Loss (linear light): L1 on alpha and on B inside the pane, L1 on F weighted by alpha, and a reconstruction term
  |alpha_hat F_hat + (1 - alpha_hat) B_hat - I|.
- Trained on 256-px crops, applied to full 768-px images.
- Variant "oracle alpha": the true alpha is supplied, and only F and B are predicted. This separates "find the
  surface fraction" from "split the colours".

**Training data (synthetic, disjoint from every test image).**
- Backgrounds: round 40's 40 generated scenes, round 45's 18 scenes not used in round 46, and RealBokeh test-split f/22
  sources (transparent-object scenes 48, 49, 51, 116, 129 and 131 excluded). About 270 backgrounds, with flips and crops.
- Surfaces, from a procedural generator with random parameters:
  - curved strokes (1-8 px, alpha 0.5-0.95);
  - line grids (spacing, width and angle random);
  - smooth specular streaks (alpha ≤ 0.7);
  - tints (alpha 0.15-0.5) with scratch texture;
  - rims and dust.
- Panes: rounded rectangles, ellipses and polygons. F colours random (light or tinted).
- The exact round 46 pattern instances are not used.

**Test sets.**
- **T1 (primary):** the round 46 benchmark unchanged (6 backgrounds never seen in training × 4 surface types, same
  depth maps, focus, K and seed). The round 46 renders of A, S, M2, O and C_* are reused as comparisons.
- **T2 (held-out family):** the same 6 backgrounds with two surface families absent from training: text glyphs and
  dot patterns. This tests generalisation beyond the trained families.
- **T3 (real, descriptive only):** RealBokeh rb_48, the acrylic sign with a real f/2.0 capture. The pane region is the
  round 43 mask. Reported against round 43's real-lens ratio (rho_real 0.84) and as images; no decision.

**Methods compared on T1 and T2.**
- SEP-full: alpha predicted (primary).
- SEP-oracle-alpha.
- O (output blend, the practical baseline from round 46).
- A, S and M2 for context.
- The reference (error 0 by construction).

**Measures** (as round 46):
- Linear-light mean absolute error against the reference (×255): whole pane, surface-dominant / transmission-dominant /
  overlap regions, and outside the pane.
- High-pass correlation.
- Separation quality itself: errors of alpha_hat, F_hat (where alpha > 0.2) and B_hat inside the pane.
- Background-level units with a bootstrap over the 6 backgrounds; results per surface type.

**Controls.**
- Rendering B_hat := the true B must reproduce the reference exactly (pipeline check).
- The separator applied to a pane-free image (alpha = 0 everywhere) must return B_hat ≈ I (it must not invent a surface).

**Decision (to be frozen).**
- SUCCESS if SEP-full lowers the T1 whole-pane error versus O by ≥ 25% (relative), with the CI of the paired difference
  excluding 0, and no available region worse than O by more than 5%.
- PARTIAL if only SEP-oracle-alpha meets this: splitting the colours works, but estimating alpha is the bottleneck.
- Otherwise: no advantage is established for the tested separator; output blending remains the practical baseline.
- T2 and T3 are reported, not decided.

**Cost.**
- Data generation (CPU) and about 1 h of training on a GPU.
- BokehDiff renders: 24 (T1) + 12 (T2) composites × 2 variants + 1 real = about 75.
- One pod run, smoke test first.

**Caveats, stated in advance.**
- Separation from one image is ambiguous: a sharp detail can be on the glass or behind it.
- Synthetic training may not transfer to real glass. T3 is a single real example.
- The pane-region mask is privileged.
