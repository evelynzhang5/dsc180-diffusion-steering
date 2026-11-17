# Round 17: can we run BokehDiff on our GPU? (one-image smoke test)

**Status:** closed (Nov 17). **BokehDiff runs on the GTX 1080 Ti** in fp16: 9.6 GB peak, 1.7 s per image, repeat runs identical, no NaN or black outputs, blur increases with K. The baseline is usable for a small study on this GPU.

## Why this round

The project question (Nov 16) is: *when can lightweight activation steering reproduce a specified depth-of-field change while preserving the scene, and what extra control is needed when it cannot?* The first step needs one working depth-of-field editor, in which to look for a specific, repeatable failure. The hypothesis is that overall blur is right but its distribution across known depths is wrong.

[BokehDiff](https://github.com/FreeButUselessSoul/bokehdiff) fits best:
- it edits a real image;
- it conditions on a depth map with a thin-lens-style circle of confusion, K/10 × |disparity − focus disparity|, where the focus disparity is the median disparity inside the salient-object mask;
- it has a depth-dependent attention module with occlusion handling.

It is built on SDXL (RealVisXL V5.0) with a LoRA and a fine-tuned VAE, and does a single denoising step at timestep 499. Its README says about 10 GB of GPU memory at 512 × 512, with fp16 available. Our only GPU is a GTX 1080 Ti (11 GB). Pascal supports fp16 storage, but its fp16 arithmetic is slow. In fp32 the SDXL UNet alone is about 10 GB, so fp16 is required.

Passing would establish "we can run this baseline for a small study". It would not establish a scientific failure or a contribution.

## Plan (fixed before running)

Script: [dsmlp_bokehdiff.sh](dsmlp_bokehdiff.sh), on one DSMLP GPU pod.

- **Environment.**
  - Separate venv that reuses the image's PyTorch: diffusers 0.32.1 (the version BokehDiff pins), transformers 4.47.1, peft 0.14.0, accelerate 1.2.1, cupy-cuda12x and others.
  - Upstream code pinned to commit `38ff937` and not modified.
  - No xformers: the inference script already sets PyTorch's own attention.
- **Input.** The repository's own test photo `EBB_4690.jpg` (flower, layered foliage), centre-cropped and resized to 512 × 512.
  - Depth (Depth-Anything-V2 Base) and mask (BiRefNet) are prepared first in a separate process, so those models do not share GPU memory with inference.
- **Inference.**
  - `inference_hf.py --mixed_precision fp16 --seed 0`, each call renders three focal planes (foreground, middle, background).
  - Runs: K = 20, the same again (repeatability), then K = 10 and K = 40.
  - The depth-dependent attention and occlusion handling are kept as released. Removing them to save memory would defeat the later physical-consistency test.
- **Recorded per run:** exit status, wall time, peak GPU memory (whole card, polled every 0.5 s), peak resident system memory, and the output images.
- **Checks:** outputs finite and not black (fp16 VAE overflow is a known SDXL failure); repeat-run max difference; background sharpness (Laplacian variance) against the input.

**Budget.** At most about 60-90 minutes of active compatibility debugging, not counting downloads (about 15 GB, mostly RealVisXL) or queue time. Allowed:
- fp16;
- moving idle parts (text encoders) to the CPU;
- running the VAE in fp32 if fp16 gives NaNs or black images;
- small version pins.

If it needs substantial rewriting, stop and use larger hardware.

**Outcome.**

| result | meaning / next step |
|---|---|
| runs, outputs sane, blur increases with K | the baseline is usable for a small study on this GPU; next, design the depth-distribution test |
| runs only with changes that remove its mechanism, or does not fit | not usable here; needs a 24 GB+ GPU |
| runs but produces NaNs or black images that the allowed fixes do not solve | numerical issue on Pascal fp16; same conclusion |

## Hardware note

For the team project, ask course staff about larger-GPU access:
- 24 GB is a sensible target for BokehDiff;
- AnyBokeh's README reports about 27 GB for its full inference pipeline;
- FLUX-based editors (Bokeh Diffusion) need more, so ask whether 48 GB is available if those become necessary.

## Results (Nov 17, 01:02-01:18)

Files: [results/smoke_sheet.png](results/smoke_sheet.png) and [results/smoke.json](results/smoke.json). The full logs and images are in `outputs/q2_bokehdiff_smoke/`.

| step | exit | wall time | peak GPU memory (card) | peak system memory |
|---|---|---|---|---|
| depth and mask preparation | 0 | 10 s | 4.6 GB | 1.7 GB |
| K = 20 | 0 | 35 s (model load included; 1.69 s per image) | 9.5 GB | 16.6 GB |
| K = 20 repeat | 0 | 22 s (1.72 s per image) | 9.5 GB | 16.4 GB |
| K = 10 | 0 | 23 s (1.75 s per image) | 9.6 GB | 16.3 GB |
| K = 40 | 0 | 21 s (1.71 s per image) | 9.6 GB | 16.2 GB |

**Checks.**
- **Repeat run:** identical to the first (max difference 0 on all three focal planes).
- **Outputs:** all finite, none black.
- **Blur grows with K.** Whole-image Laplacian variance (input 113.7) by focal-plane shift:

  | K | shift −0.48 (focus behind the flower) | shift 0 (focus on the flower) | shift +0.12 |
  |---|---|---|---|
  | 10 | 54.7 | 63.5 | 63.9 |
  | 20 | 27.7 | 54.9 | 43.1 |
  | 40 | 12.6 | 39.7 | 25.2 |

**By eye** (sheet rows: input, depth and mask; then K = 10, 20, 40; columns: the three focal planes):
- With focus on the flower, the flower stays sharp and the background blurs more as K grows.
- With focus behind the flower, the flower blurs and the background stays sharp.

Whether this blur follows the depth map quantitatively is the next study's question. It is not judged here.

**Compatibility debugging used about 10 minutes of the 60-90-minute budget.** Three problems, all in packaging; BokehDiff's own code was not modified:
1. **numpy.** pip installed numpy 2, and the image's TensorFlow (which transformers imports) is built for numpy 1. Fixed by pinning `numpy<2` and setting `USE_TF=0`.
2. **PyTorch.** pip replaced the image's PyTorch 2.2.1 + CUDA 12.1 with 2.15 + CUDA 13.2, which ran on the CPU (0 MB of GPU memory used), most likely because that build no longer supports Pascal. Fixed by rebuilding the venv with the image's PyTorch pinned, plus a check that stops the run if CUDA is not available.
3. **fp16 input.** Upstream's fp16 path feeds an fp32 image into the fp16 VAE encoder ("Input type (float) and bias type (c10::Half)"). [run_fp16.py](run_fp16.py) casts that input to fp16 and runs the call under autocast. No component or mechanism was changed.

**Limits of this result.**
- One image, at 512 × 512.
- The repository's own test photo, which is already mildly defocused (an EBB! sample).
- Depth and focus come from Depth-Anything-V2 and BiRefNet, not ground truth.
- It establishes "we can run this baseline", not a failure or a contribution.
- Peak GPU memory is 9.6 of 11 GB. Larger images would need tiling (the pipeline has a `TILE_SIZE` option) or a bigger card.

**Next (not run).** The depth-distribution test from the project question: on scenes with known depth (for example rendered or layered synthetic scenes), check whether BokehDiff's blur at each depth follows its own requested circle of confusion, K/10 × |disparity − focus disparity|.
