"""Round 45: generate the 24 fresh scenes. Identical settings to round 40 gen40.py
(RealVisXL V5.0, 1024x1024, 30 steps DPM++ 2M Karras, CFG 5, fixed seeds; saved at 768x768 Lanczos).
usage: python gen45.py <data root>     -> <root>/input/<id>.png, <root>/gen45.json
"""
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts45 as P  # noqa: E402

ROOT = sys.argv[1]
SMOKE = os.environ.get("SMOKE", "0") == "1"
os.makedirs(f"{ROOT}/input", exist_ok=True)
pipe = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V5.0", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.to("cuda")
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()
rows = []
for s in (P.scenes()[6:9] if SMOKE else P.scenes()):
    t = time.time()
    g = torch.Generator("cuda").manual_seed(s["seed"])
    im = pipe(prompt=s["prompt"], negative_prompt=s["negative"], num_inference_steps=30, guidance_scale=5.0,
              height=1024, width=1024, generator=g).images[0]
    im.resize((768, 768), resample=3).save(f"{ROOT}/input/{s['id']}.png")
    rows.append(dict(s, seconds=time.time() - t))
    print(s["id"], f"{time.time() - t:.0f}s", flush=True)
    del im
    torch.cuda.empty_cache()
json.dump(dict(model="SG161222/RealVisXL_V5.0", steps=30, cfg=5.0, size=768, rows=rows),
          open(f"{ROOT}/gen45.json", "w"), indent=1)
