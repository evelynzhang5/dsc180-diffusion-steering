"""Round 18 inference driver. Run from the BokehDiff repo directory (pinned commit), inside the round 17 venv:
    python run_depth.py <inference_hf.py arguments, with --K 20 and --test_data_dir pointing at scene A only>

BokehDiff's main() builds the pipeline, loads the released LoRA and VAE, installs the PISA attention processor and
reads scene A through its own loader. Its first call to log_validation is intercepted:
  1. convention check: the disparity / blur maps upstream built from scene A (mask = nearest plane, first focal
     shift var = -0.6 * min(defocus), K = 20) are compared with the maps this script builds itself;
  2. the test grid runs with this script's maps (3 scenes x K {0, 10, 20, 40} x focus {near, mid}, plus seed 1
     and one in-process repeat), every map entering the model saved;
  3. the process exits (upstream's own loop never writes outputs).
Same fp16 handling as round 17 (input cast to the VAE dtype, CUDA autocast). The global RNG is reseeded before
every call: PISA applies dropout (p=0.1, train=True) at inference and samples occlusion rays with torch.rand.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure  # noqa: E402
import scenes  # noqa: E402

OUT = os.environ["R18_OUT"]
UPSTREAM_K = 20.0
KS = [10, 20, 40]
FOCI = ["near", "mid"]
_orig = I.log_validation


def _call(pipeline, seed, train_T, accelerator, prompt, img, disp_coc):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.autocast("cuda", dtype=torch.float16):
        return _orig(pipeline, seed, train_T, accelerator, validation_prompt=prompt,
                     aif_image=img.to(pipeline.vae.dtype), disp_coc=disp_coc, timeit=True)


def _maps(s, K, focus):
    dn = measure.dnorm(s["disparity"]).astype(np.float32)
    if focus is None:
        return dn, np.zeros_like(dn), None
    df = float(dn[s["depth"] == s["focus"][focus]][0])
    return dn, np.abs(dn - df) * K / 10.0, df


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    # 1. convention check against upstream's loader (scene A, mask on the nearest plane)
    up = disp_coc.float().cpu().numpy()[0]
    sA = scenes.make("A")
    dn, _, df = _maps(sA, 1.0, "near")
    var1 = -0.6 * float((dn - df).min())
    coc_expected = np.abs(dn - df + var1) * UPSTREAM_K / 10.0
    img_up = aif_image.float().cpu().numpy()[0].transpose(1, 2, 0)
    img_ours = sA["image"].astype(np.float32) / 255.0 * 2 - 1
    conv = dict(disp_max_abs=float(np.abs(up[0] - dn).max()), coc_max_abs=float(np.abs(up[1] - coc_expected).max()),
                coc_range=[float(up[1].min()), float(up[1].max())], focus_disp_ours=df, var1=var1,
                image_max_abs=float(np.abs(img_up - img_ours).max()), shape=list(up.shape))
    conv["passed"] = bool(conv["disp_max_abs"] <= 2e-3 and conv["coc_max_abs"] <= 2e-3 * max(1.0, up[1].max())
                          and conv["image_max_abs"] <= 2.0 / 255 * 2)
    np.savez_compressed(f"{OUT}/maps/upstream_sceneA.npz", disp=up[0].astype(np.float16), coc=up[1].astype(np.float16))
    json.dump(conv, open(f"{OUT}/convention.json", "w"), indent=1)
    print("convention check", conv, flush=True)

    # 2. grid
    conds, times = [], []
    for n in scenes.LAYOUTS:
        s = scenes.make(n)
        img = torch.from_numpy(s["image"].astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        jobs = [(0, None, 0, False)] + [(k, f, 0, False) for f in FOCI for k in KS] + [(20, "near", 1, False)]
        if n == "A":
            jobs.append((20, "near", 0, True))
        for K, focus, sd, rep in jobs:
            dn, coc, df = _maps(s, K, focus)
            dc = torch.from_numpy(np.stack([dn, coc])[None]).to(dev, dt)
            tok = F.interpolate(dc.float(), size=(32, 32), mode="bilinear", align_corners=False)[0, 1].cpu().numpy()
            image, dur = _call(pipeline, sd, train_T, accelerator, validation_prompt, img, dc)
            tag = f"scene{n}_K{K}_{focus or 'none'}_s{sd}" + ("_repeat" if rep else "")
            image.save(f"{OUT}/{tag}.png")
            np.savez_compressed(f"{OUT}/maps/{tag}.npz", disp=dn.astype(np.float16), coc=coc.astype(np.float16),
                                coc_tokens32=tok.astype(np.float32), r_nom_px=(coc * measure.NOM * 10).astype(np.float16))
            a = np.asarray(image).astype(np.float32)
            conds.append(dict(scene=n, K=K, focus=focus, seed=sd, repeat=rep, file=f"{tag}.png", d_focus=df,
                              seconds=dur, finite=bool(np.isfinite(a).all()), black=bool(a.max() < 5)))
            times.append(dur)
            print(tag, f"{dur:.2f}s", flush=True)
    a = np.asarray(Image.open(f"{OUT}/sceneA_K20_near_s0.png")).astype(np.float32)
    b = np.asarray(Image.open(f"{OUT}/sceneA_K20_near_s0_repeat.png")).astype(np.float32)
    meta = dict(conds=conds, repeat_max_abs_diff=float(np.abs(a - b).max()),
                peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 1e9, mean_seconds=float(np.mean(times)),
                prompt=validation_prompt, train_T=train_T, dtype=str(dt))
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    print("repeat max |diff|", meta["repeat_max_abs_diff"], "peak alloc GB", meta["peak_mem_alloc_gb"], flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
