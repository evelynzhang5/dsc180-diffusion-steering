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

## Design (draft; frozen with a sha256 before any image of this round is generated)

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

## Status

- Nov 13, 2026: design drafted. Next: code (`sd_dof_direction.py`, reusing `sd_dof.py`'s sampler and meters,
  `dsmlp_sd_dof_direction.sh`), then freeze the design and run `build`.
