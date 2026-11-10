# Q2 round 4: SD 1.5 depth-of-field steering and reference-set bias, closed

Branch `q2-sd-dof`. Nov 9-10, 2026. Reproduce the NA-RFM paper's documented SD 1.5 depth-of-field setup, check
whether its sharpness reduction is more than reframing, then test whether the reference set's composition changes
framing at matched sharpness reduction. **Closed on Nov 10, 2026 without further blur diagnostics:** the steering
reduces sharpness across the whole image rather than selectively blurring the background, and the dataset cannot
supply the reference groups the bias test needs. The index of all Q2 rounds is in [../README.md](../README.md).

**Results in git.** The scripts write to `outputs/` (not in git). Copies of the small result files (reports, figures,
logs, audit labels; files over 1 MB left out, large PNGs converted to JPEG) are in `results/`, under the same folder
names: `outputs/q2_sd_dof/` -> `results/q2_sd_dof/` (with `zoom/`), `outputs/q2_ref_bias/` -> `results/q2_ref_bias/`.
Generated images and the 787 MB activation file `acts.pt` are on DSMLP only, under
`/dsmlp/workspaces-fs05/DSC180A_FA26_001/home/yuz271/dsc180-diffusion-steering/outputs/`. `sd_dof.py` needs the
paper's code from `python cifar10_replication/replicate.py setup`.

| File | What it does | Outputs |
|---|---|---|
| `sd_dof.py` | Reproduces the NA-RFM paper's SD 1.5 depth-of-field setup (Appendix G.2, Table 24), then compares RFM, mean-difference and prompt steering at matched background-sharpness change | `outputs/q2_sd_dof/` |
| `zoom_control.py` | Zoom-only control on the saved images: is the sharpness reduction more than reframing? | `outputs/q2_sd_dof/zoom/` |
| `ref_bias.py` | Reference-set bias pilot: portrait composition of the reference set vs. framing and content at matched sharpness reduction | `outputs/q2_ref_bias/` |
| `dsmlp_*.sh` | Run the scripts above on a DSMLP GPU pod (smoke test first) | pod logs next to the outputs |
| `requirements.txt` | Python packages on top of the DSMLP scipy-ml image | |

## Results so far

**Reproduction** (25 COCO captions x 4 seeds). The paper's RFM setting lowers background sharpness (mean Laplacian
variance) by 35.6% (paper: 27.6%), but foreground sharpness falls by as much: the FG/BG ratio goes 2.57 -> 2.67
(paper: 2.25 -> 2.59), and the paired change in the log ratio is +0.02 (95% interval -0.22 to +0.24). The
mean-difference direction (cosine 0.73 with RFM) does the same at s = 2. Steered images often show changed objects
and layout, not only blur.

**Zoom-only control** (`zoom_control.py`). Each steered image is aligned to its unsteered pair (scale and shift);
the unsteered image warped by that transform is the zoom-only control. 78 of 100 RFM pairs and 87 of 100
mean-difference pairs align (low-passed NCC >= 0.7, overlap >= 50%). The fitted zoom-only controls explain little of
the observed sharpness reduction: in the same regions, the control lowers log sharpness by about 0.02, while the
steered images lower it by 0.25 (background) and 0.32 (foreground) for RFM, and 0.25 and 0.52 for mean difference.
The reduction is not selective to the background. This does not separate softening from changed content: pairs that
align in coarse layout still differ in objects and details (see `results/q2_sd_dof/zoom/pairs.jpg`), so we report the measured quantity as
sharpness reduction and use "image softness" only as the working name of the target.

**Resampling artifact (fixed).** The first version warped every unsteered image with bicubic interpolation, even
when the fitted scale was within one step of the alignment grid (e.g. 1.004). Resampling at such a scale alone lowers
log Laplacian variance by about 0.19, which made the zoom-only control look like it explained most of the reduction
(the first run said "RFM: explained by zoom"). Scales within 1.5% of 1 are now treated as no zoom and applied as
whole-pixel shifts (no resampling); pairs at a fitted scale of exactly 1 lost 0.00 under the old code too, which is
how the artifact was found.

**Reference-set bias on this dataset: stopped before generation** (`ref_bias.py`, `results/q2_ref_bias/audit.md`).
Portrait composition is too rare in svnfs/depth-of-field to build balanced, biased and opposed reference sets (a
blind audit found 4 real portraits among 12 CLIP "portraits"). Close-up composition was the alternative: CLIP's
confident labels put 84% of shallow-focus photos and 11% of deep-focus photos in close-up, but an AI audit of the
rare groups confirmed 32 deep-focus close-ups (of 52), just enough for 40 photos per class at an 80% share, but only
15 shallow-focus wide scenes (of 54; 32 needed), nearly all landscapes and about as sharp as deep-focus photos. No images were generated for this test.

The AI audit's human spot-check (20 listed photos in `results/q2_ref_bias/audit.md`) was not done before closing.
