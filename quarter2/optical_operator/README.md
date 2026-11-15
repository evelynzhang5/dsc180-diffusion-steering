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

## Running

`bash quarter2/optical_operator/dsmlp_optical.sh` on a DSMLP GPU pod (both stages, one after the other).

## Status

- Nov 15, 2026: design frozen, code written; launched as `q2-optical-operator`.
