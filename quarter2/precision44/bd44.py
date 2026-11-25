"""Round 44 (GPU): does fp16 PISA attention explain BokehDiff's weak-blur misses? A precision DIAGNOSTIC only.

Same driver as round 40 bd40.py / round 42 bd42.py (processing size, preprocessing, training depth convention, prompt,
seed 0, fp16 pipeline); the ONLY difference between the two modes is PISA's attention function:
  fp16  as shipped (a pass-through wrapper that only logs the dtypes it receives);
  fp32  round 20-22's wrapper (q2-confirm-fp32 run22.py): query/key/value/weights/coc/occlusion cast to float32 with
        autocast disabled inside PISA's scaled-dot-product attention, result cast back.
Every call's input dtype is logged per mode, so the precision actually used is recorded, not assumed.
Parts (identical inputs and settings in both modes):
  A  dev curves   round 40's 8 development (mapping) scenes x K in K_DEV, both modes. The fp16 renders at round 40's
                  grid K must reproduce round 40's bd_dev outputs (determinism check, max |diff| <= 2/255).
  B  test diag    round 42's 24 test scenes x 3 targets at round 42's depth-rule K (plan42.json), fp32 only; the fp16
                  counterparts are round 42's depth-arm outputs (first 3 scenes re-rendered in fp16 as a check).
                  These scenes were already used: diagnostic only, no controller is fitted or chosen on them.
No controller is fitted here. Recalibration on fp32 development curves and the fresh-scene test are a later round.
usage (BokehDiff repo as cwd): python bd44.py --test_data_dir <one input png> --output_dir <unused> ...
env: R40_DATA, R40_S3, R42_RUN, R44_OUT, SMOKE
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compare40"))
import common40 as C  # noqa: E402

DATA, S3, R42, OUT = os.environ["R40_DATA"], os.environ["R40_S3"], os.environ["R42_RUN"], os.environ["R44_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
PLAN = json.load(open(os.path.join(HERE, "..", "control42", "results", "plan42.json")))
MAP = json.load(open(os.path.join(HERE, "..", "compare40", "results", "stage3", "mapping.json")))["bokehdiff"]
K_DEV = [4, 6, 8, 10, 12, 14, 16, 20, 28, 36, 48, 64]
_orig_val = I.log_validation
_orig_attn = P.customized_scaled_dot_product_attention
MODE = {"m": "fp16"}
CALLS = Counter()


def _f(t):
    return t.float() if torch.is_tensor(t) else t


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    CALLS[(MODE["m"], str(query.dtype), str(disp_coc.dtype) if torch.is_tensor(disp_coc) else "none")] += 1
    if MODE["m"] == "fp16":
        return _orig_attn(query, key, value, weight_matrix, disp_coc, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, hard=hard, train=train, occ_map=occ_map)
    with torch.autocast("cuda", enabled=False):                     # = run22.attn_fp32
        return _orig_attn(_f(query), _f(key), _f(value), _f(weight_matrix), _f(disp_coc), attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard, train=train,
                          occ_map=_f(occ_map)).to(value.dtype)


P.customized_scaled_dot_product_attention = attn


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
    rows = []

    def render(i, K, mode, tag):
        MODE["m"] = mode
        x = Image.open(f"{DATA}/input/{i}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        d, f = prepared_disp(i, W2, H2)
        dc = torch.from_numpy(np.stack([d, np.abs(d - f) * K / 10])[None]).to(dev, dt)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                   aif_image=x_in, disp_coc=dc, timeit=True)
        torch.cuda.synchronize()
        os.makedirs(os.path.dirname(f"{OUT}/{tag}"), exist_ok=True)
        image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
        print(tag, f"K {K:.2f}", f"{dur:.2f}s", flush=True)
        return f"{tag}.png", float(time.time() - t0)

    dev_ids = sorted(MAP["responses"])
    test = PLAN["test_scenes"]
    k_dev, targets = K_DEV, [f"{t:g}" for t in C.TARGETS]
    if SMOKE:
        dev_ids, k_dev, test, targets = dev_ids[:1], [4, 8], test[:1], ["4"]
    for i in dev_ids:                                                 # part A
        for K in k_dev:
            for mode in ["fp16", "fp32"]:
                f, s = render(i, K, mode, f"dev/{mode}/{i}_K{K:g}")
                row = dict(part="dev", id=i, K=float(K), mode=mode, file=f, seconds=s)
                ref = f"{S3}/bd_dev/{i}_K{K:g}.png"
                if mode == "fp16" and os.path.exists(ref):
                    a, b = C.img(f"{OUT}/{f}"), C.img(ref)
                    row["max_abs_diff_vs_r40"] = float(np.abs(a - b).max())
                rows.append(row)
        json.dump(dict(smoke=SMOKE, rows=rows, calls={"|".join(k): v for k, v in CALLS.items()}),
                  open(f"{OUT}/bd44.json", "w"), indent=1)
    for n, i in enumerate(test):                                      # part B
        for t in targets:
            K = PLAN["plan"][i]["targets"][t]["depth_K"]
            modes = ["fp32", "fp16"] if n < 3 else ["fp32"]
            for mode in modes:
                f, s = render(i, K, mode, f"test/{mode}/{i}_t{t}")
                row = dict(part="test", id=i, target=float(t), K=float(K), mode=mode, file=f, seconds=s)
                if mode == "fp16":
                    a, b = C.img(f"{OUT}/{f}"), C.img(f"{R42}/depth/{i}_t{t}.png")
                    row["max_abs_diff_vs_r42"] = float(np.abs(a - b).max())
                rows.append(row)
        json.dump(dict(smoke=SMOKE, rows=rows, calls={"|".join(k): v for k, v in CALLS.items()}),
                  open(f"{OUT}/bd44.json", "w"), indent=1)
    json.dump(dict(smoke=SMOKE, k_dev=k_dev, rows=rows, calls={"|".join(k): v for k, v in CALLS.items()}),
              open(f"{OUT}/bd44.json", "w"), indent=1)
    print("bd44 done", len(rows), "images; PISA calls", dict(CALLS), flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
