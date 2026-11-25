"""Round 45 (GPU): fresh-scene confirmation of the monotone controller, all renders with fp32 PISA attention.

Same driver as rounds 40/42/44 (processing size, preprocessing, training depth convention, prompt, seed 0, fp16
pipeline) with round 44's fp32 PISA wrapper. The wrapper ASSERTS that PISA's attention result is computed in float32
(the dtype of the attention output before it is cast back) and logs input and compute dtypes for every call.
Arms (K from plan45.json, frozen before rendering): fixed, linear, monotone; 3 targets each; one render per cell.
usage (BokehDiff repo as cwd): python bd45.py --test_data_dir <one input png> --output_dir <unused> ...
env: R45_DATA, R45_OUT, SMOKE
"""
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402
import PISA_attn_processor as P  # noqa: E402

DATA, OUT = os.environ["R45_DATA"], os.environ["R45_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
_orig_val = I.log_validation
_orig_attn = P.customized_scaled_dot_product_attention
CALLS = Counter()


def _f(t):
    return t.float() if torch.is_tensor(t) else t


def attn_fp32(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):                     # = run22.attn_fp32 / bd44 fp32 mode
        out = _orig_attn(_f(query), _f(key), _f(value), _f(weight_matrix), _f(disp_coc), attn_mask=attn_mask,
                         dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard, train=train,
                         occ_map=_f(occ_map))
    assert out.dtype == torch.float32, out.dtype
    CALLS[(str(query.dtype), str(out.dtype))] += 1
    return out.to(value.dtype)


P.customized_scaled_dot_product_attention = attn_fp32


def proc_size(w, h):                                                # = bd40.proc_size
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def prepared_disp(i, W2, H2):                                       # = bd40.prepared_disp
    d = np.load(f"{DATA}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = cv2.dilate(cv2.erode(d, np.ones((3, 3))), np.ones((3, 3)))
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.open(f"{DATA}/mask/{i}.png").resize((d.shape[1], d.shape[0]), resample=Image.LANCZOS)
                 .convert("L")) / 255.0
    f = float(np.median(d[m > 0.5]))
    d, f = 1 - d, 1 - f
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR))
    return d.astype(np.float32), f


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    plan = json.load(open(f"{DATA}/plan45.json"))["plan"]
    ids, targets = sorted(plan), ["4", "8", "12"]
    if SMOKE:
        ids, targets = ids[:1], ["4"]
    rows = []
    for i in ids:
        x = Image.open(f"{DATA}/input/{i}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        d, f = prepared_disp(i, W2, H2)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        for t in targets:
            for arm in ["fixed", "linear", "monotone"]:
                K = plan[i]["targets"][t][arm]
                dc = torch.from_numpy(np.stack([d, np.abs(d - f) * K / 10])[None]).to(dev, dt)
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.float16):
                    image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                           aif_image=x_in, disp_coc=dc, timeit=True)
                torch.cuda.synchronize()
                tag = f"{arm}/{i}_t{t}"
                os.makedirs(f"{OUT}/{arm}", exist_ok=True)
                image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
                rows.append(dict(id=i, target=float(t), arm=arm, K=float(K), file=f"{tag}.png", render_s=float(dur),
                                 total_s=float(time.time() - t0)))
                print(tag, f"K {K:.2f}", f"{dur:.2f}s", flush=True)
        json.dump(dict(smoke=SMOKE, rows=rows, calls={"|".join(k): v for k, v in CALLS.items()}),
                  open(f"{OUT}/bd45.json", "w"), indent=1)
    json.dump(dict(smoke=SMOKE, rows=rows, calls={"|".join(k): v for k, v in CALLS.items()}),
              open(f"{OUT}/bd45.json", "w"), indent=1)
    print("bd45 done", len(rows), "images; PISA calls (input dtype, compute dtype):", dict(CALLS), flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
