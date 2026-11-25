# Round 46: can internal feature mixing beat output blending on transparent content? (controlled feasibility test)

Branch `q2-feature46` (from `q2-confirm45`). Round 43 found that no tested single depth map both blurs the content seen
through a transparent object and keeps the object's own structure. Review #8 narrowed the next question:

> Given the same composite image, surface fraction and depth information, can internal feature mixing preserve surface
> structure and blur transmitted background better than output blending?

This is a feasibility pilot. An advantage would need confirmation on fresh backgrounds and real photos.

## Design (frozen before any round 46 render; `compose46.py`, `bd46.py`, `score46.py`, `dsmlp46.sh`)

**Composites** (linear light): I = α·F + (1 − α)·B inside a rounded pane; outside the pane α = 0.
- **B, the independent units:** 6 independently generated, all-in-focus backgrounds (round 45's fresh scenes
  f_fur_0, f_hair_3, f_highlights_0, f_highlights_2, f_thin_1, f_hair_5).
- **F and α, 4 surface types** varied independently of the background:

| type | α | F |
|---|---|---|
| etched | 0.85 on curved strokes, 0.06 elsewhere | frost |
| grid | 0.70 on 2 px lines, 0.06 elsewhere | grey |
| specular | up to 0.60 on soft diagonal streaks | white |
| tinted | 0.35 ± 0.10 (scratch texture) with a 0.90 rim | blue tint |

- **Regions:**
  - surface-dominant: α ≥ 0.8;
  - transmission-dominant: α ≤ 0.2;
  - **overlap: 0.2 < α < 0.8**, where both layers must be handled at the same pixels.
- A region with fewer than 2,000 pixels is marked unavailable. Grid and specular have no surface-dominant region, and
  tinted has no transmission-dominant one.

**Identical conditioning.**
- Depth maps are built directly in BokehDiff's processing convention: no subject mask, no re-normalisation, no
  preprocessing choices.
- For each background: d_B is its own map rescaled to [0.05, 0.70]; pane depth = focus = 0.95; one K (round 45's
  monotone rule for 12 px).
- Every render of a background (reference and all methods) uses the same map normalisation, focus, K, resolution and seed.

**Reference: a controlled compositing reference using BokehDiff-rendered backgrounds, not physical ground truth.**
R = α·F + (1 − α)·BokehDiff(B alone).

**Methods** (each sees the same composite, α and depths; all fp32 PISA with the in-call assertion):

| method | what it is |
|---|---|
| A | original map: the pane is solid and in focus |
| S | see-through map: d_B everywhere ("blur everything", negative control) |
| M2 | surface-aware single map: α ≥ 0.5 pixels at the focus depth |
| O | pixel output blend α·A + (1 − α)·S (CPU) |
| **C_pisa (primary)** | feature blend α·h_A + (1 − α)·h_S of the output of BokehDiff's **only** depth-conditioned block (`down_blocks.1.attentions.0`, the PISA transformer), tile by tile, then the full continuation re-run |
| C_latent, C_dec (exploratory) | the latent entering the VAE decoder, and the decoder's mid-block output, blended |

**Why C_pisa blends the complete differing state.**
- The pipeline installs PISA in exactly one module (`inference_hf.fn_recursive_attn_processor`).
- Depth enters only through it (`cross_attention_kwargs['disp_coc']`).
- Everything computed before this block, including every skip connection it or later layers read from earlier layers,
  is depth-free and identical in A and S, and nothing after it receives depth.
- Blending its output therefore blends the whole state in which the two runs differ.
- The endpoint checks verify this through the complete continuation.

**α delivery.** α reaches the tiled hook as a third `disp_coc` channel. PISA reads only channels 0 and 1, and the
zero-intervention check verifies that the extra channel changes nothing.

**These are two-run methods.** C_* need both depth-conditioned runs plus a continuation run.

**Controls** (must pass before quality is interpreted):
- **zero intervention:** 2-channel vs 3-channel disp_coc;
- **endpoints:** α := 1 reproduces A and α := 0 reproduces S, for every C_*, through the full continuation (≤ 1/255);
- **flat colour:** on a flat grey background with the tinted surface, the *additional* brightness change of each C_*
  relative to the α-weighted endpoint renders, mean over the pane, |Δ| ≤ 1/255.

**Measures.**
- Linear-light mean absolute error against R (×255) for the whole pane, each region, and outside the pane (sanity check).
- High-pass correlation with R over the pane.
- Background-level values average the 4 surface types. Paired differences are bootstrapped over the 6 backgrounds and
  also shown per surface type.

**Decision.**
- **ADVANTAGE** if C_pisa lowers whole-pane error versus O by ≥ 10% (relative), with the CI of the difference
  excluding 0, and no available region worse than O by more than 5%.
- Otherwise: **"No advantage is established for the tested feature blends; output blending remains the practical
  baseline."**
- O's absolute error against R is reported either way: beating a poor blend slightly may still be unusable, and
  failing to beat it does not mean transparency is solved.
- Two-source attention is **not** launched automatically afterwards.

**Dry run** (CPU; stand-in images).
- With the unblurred background as a stand-in "reference", the composite reproduces R to 0.23/255, which validates the
  compositing and reference code.
- The scorer, bootstrap and decision run end to end.

Renders: 6 references + 24 × 6 methods + checks (6 endpoint renders + 1 zero-intervention render on the first composite,
plus the flat composite) ≈ 165.

## Result (`results/run/score46.json`; run 2026-11-25 on an A30; 6 backgrounds × 4 surface types)

**Controls: all passed.**
- Zero intervention and both endpoints (α := 1 → A, α := 0 → S) reproduce exactly (max diff 0) for every blend
  location, on the first composite and on the flat composite.
- Flat colour, additional brightness change relative to the α-weighted endpoint renders (×255): C_pisa −0.71 and
  C_dec −0.85 pass; **C_latent −2.08 fails** (> 1/255), so C_latent's quality numbers are not interpretable.
- All 3,042 PISA calls computed in float32.

Mean absolute error against the reference (×255, linear light), background-level means:

| method | pane | surface (α ≥ 0.8) | transmission (α ≤ 0.2) | overlap | outside | high-pass corr. |
|---|---|---|---|---|---|---|
| A, original map | 8.57 | **23.4** | 8.08 | **12.0** | 2.43 | 0.77 |
| S, see-through (negative control) | 12.41 | 77.2 | 7.49 | 34.0 | 1.35 | 0.38 |
| M2, surface-aware single map | 9.62 | 57.2 | **5.92** | 23.1 | 2.31 | 0.60 |
| **O, output blend** | **8.48** | 26.1 | 7.08 | 17.1 | 1.35 | **0.82** |
| **C_pisa, feature blend (primary)** | 11.33 | 56.7 | 7.00 | 31.7 | 1.31 | 0.46 |
| C_latent (exploratory; flat control failed) | 11.09 | 55.6 | 6.99 | 31.5 | 1.33 | 0.53 |
| C_dec (exploratory) | 10.99 | 54.4 | 6.95 | 31.3 | 1.33 | 0.58 |

Paired against O (whole pane, bootstrap over 6 backgrounds):

| method | difference [95% CI] | backgrounds better |
|---|---|---|
| C_pisa | +2.85 [+2.11, +3.65] | 0/6 |
| C_dec | +2.51 | 0/6 |
| M2 | +1.14 | 1/6 |
| A | +0.10 | 3/6 |
| S | +3.93 | 0/6 |

Whole-pane error by surface type:

| surface type | O | C_pisa | C_dec | M2 | A |
|---|---|---|---|---|---|
| etched | 8.28 | 10.79 | 10.87 | **7.10** | 8.73 |
| grid | 16.08 | 23.27 | 22.78 | 17.76 | **9.75** |
| specular | **2.65** | 2.73 | 2.73 | 3.52 | 7.36 |
| tinted | **6.89** | 8.51 | 7.58 | 10.08 | 8.43 |

**Pre-set decision: No advantage is established for the tested feature blends; output blending remains the practical
baseline.** C_pisa is 34% worse than O on the pane, and more than twice as wrong on surface-dominant pixels.

**Reading.**
- **Why the internal blends lose:** they mix at the model's feature resolution. The PISA block works at 1/16 of the image
  resolution (one cell ≈ 16 px at the processing scale), the latent at 1/8. α has to be averaged over a cell, so thin,
  sharp surface structure (2-4 px strokes and grid lines) cannot be mixed separately from the background around it. The
  surface error roughly doubles, and transmission is unchanged.
- **Where α is smooth, blending location does not matter.** On specular streaks every blend gives the same error (2.7).
- **No method is close to the reference.** The output blend's pane error (8.5) is about 6× the error outside the pane
  (1.35, BokehDiff's own background reproduction). Its surface-dominant error is 26/255 and its overlap error 17/255.
  The input image already mixes surface and background, so neither render contains the sharp surface alone or the
  background alone.
- **No method wins everywhere.** Keeping the pane solid (A) is best on surface and overlap pixels. The surface-aware map
  (M2) is best on transmission. The output blend is the best compromise and has the best high-pass agreement (0.82).

**Limits.**
- 6 backgrounds; synthetic panes and α known exactly (privileged).
- The reference uses BokehDiff's own background render, not physical ground truth.
- One K per background (12 px target, approximate).
- C_latent failed the flat-colour control.

**What this means.**
- With the same information, mixing BokehDiff's internal features does not beat blending its outputs. The tested
  internal location's resolution is coarser than the surface detail that has to be kept.
- An internal method would need finer-resolution access, or a way to separate surface from background before blending,
  not a different blending point.
- Two-source attention acts at the same 1/16 resolution, so this result gives no reason to expect it to do better.
  It is not launched.
