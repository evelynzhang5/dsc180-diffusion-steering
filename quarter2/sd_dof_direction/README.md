# Q2 round 9: at matched background blur, does another direction blur the foreground less? (SD 1.5, depth of field)

Branch `q2-sd-dof-direction`. Design written Nov 13, 2026, while round 8 (`q2-direction-choice`) runs; no code and no
images yet. The index of all Q2 rounds is in [../README.md](../README.md).

**Why.** Round 4 (`q2-sd-dof`) reproduced the paper's depth-of-field experiment (Appendix G.2, Table 24) and found its
failure: background sharpness fell 35.6% (paper: 27.6%), but foreground sharpness fell about as much. The paired change
in log FG/BG ratio was +0.02 (95% CI -0.22 to +0.24), and steered images often had changed objects and layout. The
direction makes the whole image softer instead of blurring the background. Round 4's mean-difference direction
(cosine 0.73 with RFM) behaved the same way, and only the paper's direction and one alternative were compared.

**Question.** With strength tuned for each direction to the same background-sharpness change, does any direction
from the same reference data raise the FG/BG sharpness ratio more than the paper's, with no more scene change? This
is the Stable Diffusion counterpart of round 8's "choose the direction before learning combinations": if no single
candidate is more selective, there is no case for learning a combination of them.

## Design (frozen on Nov 13, 2026 with `design.sha256`, before any image of this round was generated)

**Fixed, as in round 4** (`sd_dof.py`):
- SD 1.5, 50 DDIM steps, text CFG 7.5. A third branch `eps_s` (the conditional prediction with the edit
  `h <- h + 0.7 |h| d / |d|` at `unet.down_blocks[2].resnets[-1]`) enters as
  `eps = eps_u + 7.5 (eps_c - eps_u) + s (eps_s - eps_c)` inside sigma in [0.041, 13.12].
- The hook normalises `d`, so only a direction's orientation matters, and the amplification `s` is the strength knob
  tuned for each direction.
- Reference data: round 4's activations (`outputs/q2_sd_dof/acts.pt` on DSMLP: 1,200 svnfs/depth-of-field photos
  at sigma 0.344, empty prompt).

**Candidates:**

| name | direction |
|---|---|
| `rfm_eig1` | the paper's: top RFM eigenvector (round 4 used `get_direction(topk=1)`; baseline) |
| `rfm_eig2`, `rfm_eig3` | the 2nd and 3rd RFM eigenvectors, each signed so shallow-focus photos project higher |
| `meandiff` | shallow minus deep mean of the raw activations (round 4's) |
| `meandiff_std` | the same divided by each feature's standard deviation (the RFM fits on standardized features) |

Round 4 saved only the top direction and the eigenvalues, so `build` refits the RFM on `acts.pt` with round 4's
settings (Laplace kernel, bandwidth 5000, reg 1e-3, 5 iterations, top_k 5).
- **Refit check:** if the refit's top eigenvector has cosine below 0.99 with round 4's direction, stop and report.
- **Duplicates:** a candidate whose cosine with an earlier one is above 0.95 is dropped as a duplicate, as `rfm_eig1`
  was in round 8.
- **What `build` reports:** held-out probe AUCs, cosines and the eigenvalue spectrum (lambda1 / lambda2). The AUCs
  are not used to choose.

**Controls** (reported, not eligible as challengers):

| control | what it shows |
|---|---|
| `none` | unsteered, the pair for every paired change |
| `random` | a Gaussian random direction through the same hook: how much of any effect is generic perturbation |
| `prompt` | round 4's `eps_s` from caption + ", shallow depth of field, blurred bokeh background", no activation edit |
| `oracle` | the `none` image with Gaussian blur applied outside its depth foreground (feathered), radius picked on dev to match the target: the selectivity an ideal intervention would reach. Post-hoc, no generation |

**Captions.** COCO-Karpathy validation, excluding the 33 captions round 4 used. A keyword rule fixed in code before
sampling splits them into two groups:
- P: the subject is a person or an animal.
- O: neither: objects, indoor scenes, landscapes.

| split | captions | seeds | images per condition |
|---|---|---|---|
| dev | 16 from P | 1, 2 | 32 |
| test | 32 new from P, 32 from O | 42, 123, 256, 789 | 256 |

Dev uses only P, so test group O also measures transfer to unseen scene types. New P captions separate that from
ordinary caption-to-caption noise.

**Measurements**, all paired against `none` on the same caption and seed:

| role | measure |
|---|---|
| target | change in log background sharpness (Laplacian variance) |
| primary | change in log FG/BG ratio; positive = selective |
| leakage | change in log foreground sharpness |
| scene change | DINOv2 CLS cosine with the unsteered image; mean absolute pixel change in foreground and background |
| framing | fitted scale from round 4's `zoom_control.py` alignment |
| text fit | CLIP caption score |

- **Masks:** foreground and background come from the **unsteered** image's depth mask (Depth Anything V2 Small,
  nearest 40%). A steered image's own mask moves if the edit changes the layout. The paper's per-image mask is
  reported as a secondary column, for comparison with round 4 and the paper.
- **Intervals:** bootstrap over captions, then over seeds within each caption.

**Stages**

| stage | what | images |
|---|---|---|
| `build` | refit, candidates, refit check, AUCs, cosines, spectrum | none |
| `tune` | each candidate and `random` x s {1, 2, 4, 8, 16}; `prompt` x s {0.5, 1, 2, 4}; `none` | about 1,000 (dev) |
| `select` | one s per candidate, one challenger; frozen with a sha256 | |
| `confirm` | `none`, `rfm_eig1`, challenger, `random`, `prompt` on test; `oracle` computed afterwards | 1,280 (test) |

At the 14-21 s per image measured in round 4's smoke test (1080 Ti), `tune` is about 5 GPU hours and `confirm` about 7. `tune`
runs as one pod and `confirm` as another, each starting with `--smoke`.

**Selection rule (dev).**
1. **Target:** the paper direction's dev change in log background sharpness at s = 4, the paper's setting.
2. **One s per candidate:** the grid value whose dev background change is closest to the target.
3. **Reach:** a candidate that cannot get within 25% of the target is reported as unable to reach it and is not
   eligible.
4. **Challenger:** the eligible non-baseline candidate with the highest dev change in log FG/BG ratio (ties: higher
   DINOv2 similarity).

The full dev table (every candidate at every s, i.e. blur-vs-leakage curves) is reported, not only the picks.

**Decision (confirm, test captions and seeds).** The challenger beats the paper's direction only if all of these hold:
- its change in log FG/BG ratio is at least **0.15** higher, and the paired 95% CI of the difference is above 0;
- the difference has the same sign in group P and in group O;
- its background change is at least 75% of the paper direction's;
- it is not worse on scene change or text fit. The lower ends of the paired 95% CIs must be above -0.02 (DINOv2
  cosine) and -0.5 (CLIP caption score, round 4's units).

| result | interpretation | next |
|---|---|---|
| all conditions hold | direction choice controls selectivity | learn a small combination of the candidates on dev, freeze it, test it on new captions (Direction A proper) |
| otherwise, `oracle` far above every candidate | no global direction from this data is selective | round 10: spatially restricted steering (edit applied only where the depth mask is background), compared with a depth-based blur renderer and `oracle` (Direction B) |
| otherwise, `random` about as "selective" as the paper's | the measured selectivity is generic perturbation | revisit the metric before any method |

**Secondary, after the decision:** text CFG {5, 10} for the paper direction and the challenger (32 test captions x 2
seeds). Round 7 showed that strengths interact (NA x RFM), so a challenger's advantage should not depend on CFG 7.5.

**Caveats fixed in advance:**
- Laplacian variance falls when texture is removed as well as when it is blurred, so lower sharpness alone does not
  show bokeh. `oracle` and the scene-change measures are there to separate the two.
- One model, one reference dataset (Unsplash photos; round 4 found the composition confounds in it), one layer.
- Captions are COCO only.
- The CLIP caption score is a weak measure of scene preservation; DINOv2 and pixel change carry that role.

**Implementation choices the draft left open** (fixed with the design):
- Caption groups: the keyword lists `PEOPLE` and `ANIMALS` in `sd_dof_direction.py` (whole words, after removing
  "teddy bear" and "hot dog"); captions are each validation image's first caption, in dataset order from image 33.
- `rfm_eig1` and `meandiff` are round 4's saved vectors exactly (so the baseline is round 4's direction); the refit
  supplies `rfm_eig2` and `rfm_eig3`; `meandiff_std` divides round 4's `meandiff` by the per-feature standard deviation
  over the 1,200 photos. `random`: standard Gaussian, torch seed 0.
- Duplicates: signed cosine above 0.95 with an earlier kept candidate, in the order of the candidate table.
- `oracle`: Gaussian blur of each colour channel (sigma from {0.5, 1, 1.5, 2, 3, 4, 6, 8} px, picked on dev),
  composited through the unsteered depth foreground mask feathered with a Gaussian of sigma 8 px.
- `random` and `prompt` get their test strength by the same closest-background rule as the candidates.
- DINOv2: `facebook/dinov2-base`, CLS token, cosine. Images are saved as PNG (round 4 used JPEG).
- Framing: log of the scale from round 4's `zoom_control.align`.
- Smoke runs (`--smoke`) skip the refit check and use 2 dev captions, 1 seed and s 4 only.

## Status

- Nov 13, 2026: design drafted.
- Nov 13, 2026: code written (`sd_dof_direction.py`, `dsmlp_sd_dof_direction.sh`; round 4's `sd_dof.py`,
  `zoom_control.py` and `requirements.txt` copied unchanged from `q2-sd-dof` into `quarter2/sd_dof/`); design frozen;
  first pod: smoke test, then build, tune and select.

- Nov 13, 2026: build, tune and select done (dev, 32 images per condition); test pod launched (`confirm`).
- Nov 14, 2026: confirm done (test, 256 images per condition, 5.5 h on one GPU). Decision: no challenger; `oracle` far
  above the paper direction, so the next step is round 10 (spatially restricted steering). Secondary (CFG 5, 10) not
  run: it was planned for the paper direction against a challenger, and there is none.

## Dev results (16 person/animal captions x seeds 1, 2)

**Build.** The refit reproduces round 4's direction exactly (|cos| = 1.0000). The RFM spectrum is one direction:
lambda1 / lambda2 = 18,730. `meandiff_std` was dropped as a duplicate of `meandiff` (cosine 0.968). Kept: `rfm_eig1`
(baseline), `rfm_eig2`, `rfm_eig3`, `meandiff`. Full tables: `results/build.md`, `results/tune.md`.

**The paper direction does not blur the background under the unsteered mask.** At s = 4, paired with the unsteered
image, inside its depth mask:

| measure | change (95% CI) |
|---|---|
| log background sharpness | -0.010 (-0.243, +0.231) |
| log foreground sharpness | -0.170 (-0.344, -0.016) |
| log FG/BG ratio | -0.160 (-0.485, +0.114) |
| log FG/BG ratio, each image's own mask (the paper's measure) | +0.373 |
| DINOv2 cosine with the unsteered image | 0.666 |

With the paper's per-image mask the ratio rises, as in the paper; with the mask fixed to the unsteered layout there is
no background blur. A reading to test on the test set: the rise comes from the steered image's changed layout moving
its own mask, not from blur.

**Selection.** The target background change is -0.010, so "within 25% of the target" means within 0.0025: no other
candidate or control can meet it, and there is no challenger. The design did not anticipate a target near zero; the
rule was applied as frozen, not changed. The challenger branch of the decision fails. The other candidates: `rfm_eig2`
and `rfm_eig3` *sharpen* the background (up to +0.96 and +0.19) and soften the foreground; `meandiff` softens the
foreground (to -0.97 at s 16) with no background blur. None is selective in the intended sense. The oracle's smallest
radius (0.5 px) gives -0.70, far beyond the target, so the oracle comparison on test is not at matched blur.


## Test results (32 person/animal + 32 object/scene captions x seeds 42, 123, 256, 789)

Full table: `results/confirm.md`; examples (columns: none, paper, random, prompt, oracle): `results/examples.png`.
Paired with the unsteered image, inside its depth mask (log changes, 95% CI over captions then seeds):

| condition | background | foreground | FG/BG ratio | ratio, own mask | DINOv2 cosine |
|---|---|---|---|---|---|
| paper (`rfm_eig1`, s 4) | -0.279 (-0.398, -0.152) | -0.386 (-0.486, -0.282) | -0.106 (-0.223, +0.011) | +0.146 (-0.020, +0.326) | 0.675 |
| random (s 1) | +0.019 | -0.003 | -0.022 (-0.052, +0.004) | -0.009 | 0.936 |
| prompt (s 2) | -0.058 | -0.046 | +0.012 (-0.068, +0.091) | +0.100 (+0.008, +0.195) | 0.829 |
| oracle (0.5 px) | -0.705 (-0.742, -0.664) | -0.051 | +0.654 (+0.614, +0.690) | n/a | 0.999 |

Paired differences in ratio change: oracle - paper +0.760 (+0.648, +0.868); random - paper +0.084 (-0.034, +0.192).

**Decision.** The challenger branch fails (no challenger, see Selection). `oracle` is far above the paper direction
(+0.76, CI well above 0), so by the frozen table the outcome is "no global direction from this data is selective"
and the next step is round 10: spatially restricted steering, compared with a depth-based blur and `oracle`.
The `random` row does not cleanly apply: `random` was matched to the dev target (about 0) and barely changes the
image on test (DINOv2 0.94), so it is not a matched-blur comparison.

**The paper direction blurs the background on test, but blurs the foreground more.** The dev reading ("no background
blur") did not hold: on test, background sharpness falls (-0.28) but foreground falls further (-0.39), so with the
unsteered mask the FG/BG ratio goes slightly *down* (-0.11, CI just includes 0), in both groups. Pixel change is
about equal inside and outside the mask (0.127 / 0.119): the edit is global.

**The paper's measure gives the paper's number; the fixed mask does not.** With each image's own mask the ratio rises
by +0.146, close to the paper's 2.25 -> 2.58 (log 0.137). With the mask fixed to the unsteered layout it is -0.106.
The two differ only in which mask is used, and the steered images change layout and content (DINOv2 0.675; in the
examples a mural's subject changes). So the paper's selectivity number depends on the mask moving with the image,
and is not evidence of background-only blur. The prompt control ("shallow depth of field") shows the same gap
(+0.012 fixed vs +0.100 own mask).

**Caveats.** One model, layer and reference dataset; COCO captions only; Laplacian variance falls when texture is
removed as well as when it is blurred. The oracle is not at matched blur (its smallest radius already exceeds the
paper direction's background change on dev and on test), so "+0.76" overstates the gap at equal blur; the sign and
size of the paper direction's own fixed-mask ratio (about 0 or below) are the more direct result.
