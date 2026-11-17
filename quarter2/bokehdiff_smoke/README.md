# Round 17: can we run BokehDiff on our GPU? (one-image smoke test)

**Status:** planned (Nov 16); runs after round 16's capacity test.

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
