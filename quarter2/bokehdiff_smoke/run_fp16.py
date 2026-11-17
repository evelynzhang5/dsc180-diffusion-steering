"""Run BokehDiff's inference_hf.py unchanged, except that the image entering the VAE encoder is cast to the
pipeline's dtype and the call runs under CUDA autocast (fp16). Upstream's fp16 path passes an fp32 image to the
fp16 VAE ("Input type (float) and bias type (c10::Half)"). No model component or mechanism is changed.
Usage (from the BokehDiff repo directory): python run_fp16.py <inference_hf.py arguments>"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402

_orig = I.log_validation


def log_validation(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
                   aif_image=None, disp_coc=None, timeit=False):
    with torch.autocast("cuda", dtype=torch.float16):
        return _orig(pipeline, seed, train_T, accelerator, validation_prompt=validation_prompt,
                     aif_image=aif_image.to(pipeline.vae.dtype), disp_coc=disp_coc, timeit=timeit)


I.log_validation = log_validation
if __name__ == "__main__":
    I.main()
