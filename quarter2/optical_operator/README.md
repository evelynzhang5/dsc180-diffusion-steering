# Q2 round 14: first tests of two post-steering hypotheses

Branch `q2-optical-operator`. Started Nov 15, 2026. The index of all Q2 rounds is in [../README.md](../README.md).

**Why.** Round 13 stopped the activation-steering approach. Editing internal activations redraws the scene, and
post-hoc blur beat every steering variant. Two candidate directions from the Nov 15 review are tested here. Both are
candidates, not novelty claims.

**Prior work that bounds them** (from the review, not yet checked paper by paper):
- **BokehDiff:** one-step diffusion with physics-informed attention; reports VAE loss of fine structures.
- **FlashReFocus:** diffusion restoration once, then a lightweight renderer and a decoder refiner.
- **Dr.Bokeh:** a differentiable layered renderer with occlusion.
- **Bokeh Diffusion:** reports hallucinated foregrounds when there is no clear subject, and residual blur in unusual
  scenes.

So "generate once, render afterwards", "restore lost detail", kernel prediction and synthetic counterfactual pairs
are each already established.

## Hypothesis 1: predict an optical transformation of the original pixels (`optical.py operator`)

Output pixel = sum over q of W(p, q) x source pixel q:
- W is a 17 x 17 softmax per pixel, predicted by a small U-Net from the image, the blur-radius map and the protected
  mask;
- identity on the protected region;
- a convex combination, so it can spread light but cannot invent colours or move structures far.

**The decisive first test.**
- **Scenes:** synthetic layered scenes with known geometry, built from real photos (round 4's set, split 80/20 by
  photo).
  - Foreground: an in-focus blob plus 1 px hair-like strands, with known alpha.
  - Background: a plane whose disk-blur radius varies 0.6-1.0 x the level.
- **Target:** rendered independently from the full background layer, which no method sees.
- **Methods**, each identity on the protected mask (alpha > 0.5), all given the same composite, radius map and mask:

  | method | what it is |
  |---|---|
  | `gather` | classical depth-dependent disk blur |
  | `normconv` | classical blur that excludes protected pixels: the strong classical baseline |
  | `image` | the same U-Net predicting output pixels: 472k parameters |
  | `operator` | the same U-Net predicting the operator: 482k parameters |

- **Training:** identical scenes and schedule for both learned models (4000 steps x 16 patches of 128 x 128, L1
  outside the mask).
- **Evaluation:** 128 held-out scenes per blur level (2, 4, 8 px). Error by region of the true geometry: protected,
  thin strands, boundary band, background. Plus a blur gap (matched-blur check) and inference time at 512 x 512.

**Decision (frozen):**
- **Continue** only if all three hold, at 2 or more of the 3 levels:
  1. the operator's boundary error is <= 0.9 x the best of `gather`, `normconv` and `image`;
  2. its thin-strand error is no more than 5% above the best other method's;
  3. its background error is no more than 10% above the best other method's.
- Otherwise, stop as framed.

**Not in this test:**
- **Diffusion features.** The question here is the output formulation alone; frozen SD features would be the next
  ablation if it passes.
- **Real scenes and occluded defocused foregrounds.**

As the review notes, an operator cannot recover sharp detail that is truly missing.

## Hypothesis 2: optical control independent of photographic stereotypes (`optical.py stereotype`)

**An existence check, not the method.** Does a blur request on SD 1.5 change the *kind* of scene?
- **Scenes** (4 seeds each):
  - a wide landscape: no foreground object should appear;
  - a small, far-away bird: it should stay small;
  - a deer with a sharp-focus request: it should stay possible;
  - an empty street: no person should appear.
- **Conditions:** none; the blur prompt as a third branch (s 2); the paper's RFM direction (global, s 4).
- **Measures:**
  - background blur reached;
  - CLIPSeg area of the subject (or of a person in the empty street);
  - CLIP preference for "a close-up object in the foreground";
  - DINOv2 to the unsteered image.

The method itself needs an adapter trained on counterfactual camera-setting groups, and its critical baseline is
ordinary balanced training data (the MNIST result: balancing alone can fix known confounding). Neither can run until
there is a dataset. This check only asks whether the failure appears in our setup at all.

## Results (Nov 15, 2026; files in [results/](results/))

Ran on a GTX 1080 Ti, 08:06-08:34, 28 min. Every check passed.

### Hypothesis 1: stop as framed

Mean absolute error x 255 against the independent target, on 128 held-out scenes per level:

| level | method | thin strands | boundary band | background | all |
|---|---|---|---|---|---|
| 2 px | gather | 8.76 | 4.66 | 0.00 | 0.75 |
| 2 px | normconv | 11.88 | 3.39 | 0.00 | 0.64 |
| 2 px | image | **2.89** | **1.25** | 0.37 | 0.46 |
| 2 px | operator | 3.01 | 1.62 | 0.07 | **0.32** |
| 4 px | image | **3.65** | **1.47** | 0.43 | 0.63 |
| 4 px | operator | 4.31 | 1.89 | 0.07 | **0.52** |
| 8 px | normconv | 20.10 | 3.08 | 0.00 | 1.31 |
| 8 px | image | **4.76** | **1.78** | 0.67 | **0.98** |
| 8 px | operator | 6.87 | 2.76 | 0.10 | 1.05 |

**Decision:**
- The operator beat the best other method's boundary error by 10% or more at **0 of 3** levels.
- Thin strands were within 5% of the best at 1 of 3; background within 10% at 0 of 3.
- **Do not continue as framed.**

**What the comparison shows.**
- **Both learned models clearly beat the classical renderers at the boundary and on thin strands.** The image model is
  about 3x better at the boundary than `normconv` and 3-6x better on thin strands. In the examples, `gather` puts a
  halo around the subject and `normconv` erases the thin strands.
- **Learning helps, but the operator form does not.** At the boundary and on thin strands, the image model beats the
  operator at every level, and the gap widens with blur (8 px: boundary 1.78 vs 2.76).
- **The operator's one advantage is the background.** It barely alters what should just be blurred (0.07-0.10 vs
  0.37-0.67 for the image model), which gives it the best overall error at 2 and 4 px.

**Caveats.**
- The target uses the same disk blur as the classical renderers. Their background-interior error is 0 by
  construction; the test is independent only through the hidden background layer.
- The scenes are synthetic: a flat in-focus foreground, no occluded defocused foreground, 128 px patches.
- One training run each, 4000 steps; differences of a few hundredths of a level are within noise.
- The 17 x 17 window fits a radius of at most 8 px, so at level 8 the operator has no margin.
- Inference at 512 x 512: operator 29 ms vs image model 9.5 ms.

### Hypothesis 2: no stereotype failure appeared in our setup

| scene | what should hold | prompt | RFM steering |
|---|---|---|---|
| small bird | stays small (CLIPSeg area x 1000, unsteered 17.3) | 17.5 (background blurred, -1.07) | 17.2 (-0.99) |
| empty street | no person appears (area) | 0.0 | 0.0 |
| landscape | no foreground object (CLIP foreground preference, unsteered -4.00) | -4.09; background not blurred (+0.18) | -3.11; background barely blurred (-0.17), scene hazier |
| deer, sharp forest requested | deer kept (area, unsteered 124) | 126 | 132 (+7%) |

- Within 4 scenes x 4 seeds, a blur request on SD 1.5 did not:
  - enlarge the small bird;
  - add a person to the empty street;
  - put an object in front of the landscape.
- The failure modes reported for Bokeh Diffusion therefore do not show up here. Steering still changes the image
  globally (DINOv2 0.52-0.81), as in rounds 9-13. On the landscape, the blur request mostly failed to blur: the
  conditions moved background sharpness by only -0.17 to +0.18, and the steered landscape came out hazier.
- **This does not rule the hypothesis in or out for stronger models.** The documented failures are in Bokeh Diffusion
  (FLUX), which we cannot run here. Testing them needs that model, or a model trained with explicit camera
  conditioning.
- Round 3's portrait-bias finding (people enlarged under a biased direction) came from a biased reference set, not
  from a blur request on neutral data.

## Running

`bash quarter2/optical_operator/dsmlp_optical.sh` on a DSMLP GPU pod (both stages, one after the other).

## Status

- Nov 15, 2026: design frozen, code written; launched as `q2-optical-operator`.
- Nov 15, 2026: ran; results above. Closed.
