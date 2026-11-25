# Round 47: does learned layer separation fix transparent content?

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

## Build (2026-11-25; frozen before training)

Files:
- `synth47.py`: procedural surfaces. Training families: strokes, grid, specular, tint, rim, dust. Held out: text, dots.
- `train47.py`: U-Net, 4.9M parameters, 8,000 steps, batch 16, 256-px crops.
- `build47.py`: T2 composites.
- `infer47.py`: runs the separators.
- `bd47.py`: BokehDiff renders with fp32 PISA.
- `score47.py`: scoring.
- `dsmlp47.sh`: one pod run.

Changes from the draft:
- **Training backgrounds:** 82 from rounds 40 and 45 plus about 214 RealBokeh f/22 sources (transparent-object scenes
  and round 46's test backgrounds excluded).
- **Held-out families densified:** text and dots were made denser before freezing (neither family is used in training),
  so that T2 has an overlap region. T2 has no surface-dominant region (≥ 0.8).
- **T3 input:** rendered with round 43's see-through map, focus and K for rb_48. Measured with round 43's see-through
  ratio on its frozen tiles, next to the real f/2.0 capture and round 43's three conditions.

**CPU dry run with perfect-separation stand-ins** (separator outputs set to the true layers; renders = round 46's
reference):
- SEP error 0.00-0.01/255 on T1 and T2.
- The round 46 numbers for A, S, M2 and O are reproduced exactly.
- The decision logic runs end to end.

## Result (`results/run/score47.json`; run 2026-11-25 on an A30)

**Controls: passed.**
- Pipeline check: rendering the true background reproduces round 46's reference exactly (max diff 0) for all 6
  backgrounds.
- No-glass control: the separator finds almost no glass where there is none (mean alpha_hat in the pane 0.001-0.023,
  limit 0.05). It shifts the background by 2.0-6.4/255.
- All 1,860 PISA calls computed in fp32.

**Training (8,000 steps each, 296 backgrounds), final training losses:**

| model | alpha | B | F | reconstruction |
|---|---|---|---|---|
| full | 0.017 | 0.014 | 0.080 | 0.017 |
| oracle-alpha | (given) | 0.009 | 0.041 | 0.015 |

**T1, round 46 benchmark** (error against the reference, ×255, linear light, background-level means):

| method | pane | surface (α ≥ 0.8) | transmission (α ≤ 0.2) | overlap | outside | high-pass corr. |
|---|---|---|---|---|---|---|
| O, output blend (round 46) | 8.48 | 26.1 | 7.08 | 17.1 | 1.35 | 0.82 |
| A, original map | 8.57 | 23.4 | 8.08 | 12.0 | 2.43 | 0.77 |
| **SEP_full (separate → render → composite)** | **5.46** | 24.9 | **3.95** | **9.8** | 2.09 | **0.92** |
| SEP_oracle (alpha given) | 2.86 | 10.9 | 2.29 | 3.5 | 1.67 | 0.99 |

- SEP_full − O on the pane: −3.01 [−4.10, −1.88], better on 6/6 backgrounds.
- Relative changes: pane −36%, surface −5%, transmission −44%, overlap −43%.
- SEP_oracle − O: −5.62 [−6.83, −4.29], 6/6 backgrounds (−66%).
- Separation errors: alpha 0.053 (full); F 14.0/255 (full) vs 7.1/255 (oracle); B 12.6/255 (full) vs 4.8/255 (oracle).

Pane error by surface type:

| surface type | O | SEP_full | SEP_oracle |
|---|---|---|---|
| etched | 8.28 | 5.28 | 2.56 |
| grid | 16.08 | 6.71 | 3.28 |
| specular | 2.65 | 3.25 | 1.72 |
| tinted | 6.89 | 6.61 | 3.87 |

**Pre-set decision: SUCCESS.** SEP_full lowers pane error by 36% (≥ 25%), the CI excludes 0, and no region is worse.

**T2, held-out families** (text, dots; reported, not decided):

| method | pane error | vs O (paired, 95% CI) | backgrounds better |
|---|---|---|---|
| O | 3.24 | | |
| SEP_full | 5.08 | +1.84 [+1.47, +2.19] (worse) | 0/6 |
| SEP_oracle | 2.13 | −1.11 [−1.87, −0.35] | 5/6 |

- The full separator's surface-colour error jumps to 36/255 on these unseen patterns. It does not recognise text and
  dots as glass.
- Given alpha, the colour split still generalises.

**T3, real acrylic sign rb_48** (round 43's see-through ratio; reported, not decided):

| condition | ratio |
|---|---|
| real f/2.0 lens | 0.84 |
| round 43, original | 0.13 |
| round 43, see-through | 0.46 |
| round 43, see-through + rim | 0.45 |
| SEP_full | 0.45 |

- The separator assigns almost no glass to the real sign: mean alpha_hat 0.028 inside the mask, while the printed logo
  is effectively opaque surface. So the pipeline behaves like the see-through map.
- It also darkens the counter below the sign (`figures/round47_real_rb48.png`).

**Reading.**
- **On the trained kind of glass, separating the layers before rendering works.**
  - Pane error falls by about a third versus the output blend, and the ghosting and smearing of rounds 43 and 46
    disappear (`figures/round47_examples.png`).
  - With the true surface fraction the error falls by two thirds. The remaining gap is mostly in estimating alpha.
- **It does not yet generalise.**
  - Unseen surface patterns (T2): estimating alpha fails, and the full pipeline is worse than the output blend.
  - The one real photograph (T3): the separator finds almost no glass.
  - The colour split itself generalises when alpha is known (T2 oracle).
- **This is the first approach that clearly improves transparent content, and it lies entirely outside BokehDiff**
  (learned decomposition before rendering, compositing after). Round 46's internal feature blend lost on the same
  benchmark with the same information.

**Limits.**
- Synthetic panes over 6 backgrounds; the pane region is privileged.
- The reference is BokehDiff's own background render.
- One real example.
- The training families resemble T1's families (different instances, same kinds).

**Next (proposed, not run).**
- Train alpha estimation on a much wider set of surface patterns: text, logos, dots and real transparent-object mattes
  (e.g. Trans10K masks), or combine it with a transparent-object matting model.
- Test on more real glass photographs.
- The pane region would come from a transparent-object segmenter.
