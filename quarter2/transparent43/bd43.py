"""Round 43 (GPU): is BokehDiff's transparent-subject failure in the depth INPUT or in the MODEL?

Same driver as round 40 bd40.py / round 41 bd41.py (processing size, preprocessing, training depth convention, prompt,
seed 0, fp16, focus = median prepared disparity inside the subject); only the depth map handed to BokehDiff and K change.
Conditions (processing scale):
  original       the prepared map, exactly as round 40.
  seethrough     inside the subject (mask > 0.5, dilated 3 px) the depth is replaced by the surrounding background,
                 filled in from outside the subject (harmonic fill at 1/4 scale: start from the nearest outside value,
                 then 400 Jacobi iterations of the Laplace equation inside the hole, upsampled): the map now says
                 "what you see through the object is as far as what is around it". The focus value is unchanged.
  seethrough_rim as seethrough, but subject pixels within RIM_P px of the subject edge keep the original depth, so the
                 object's own outline is still requested sharp. The best a one-depth-per-pixel map can say.
K per scene from regions43.json: round 42's depth rule for 8 and 12 px ("8", "12"); the original condition also at
K = 36 and 64 (does ANY setting blur the through-content when the depth is unchanged?).
The edited depth maps are saved (depth/<id>_<cond>.npy, processing scale) so the scorer can compute the requested
blur over the see tiles chosen at review.
usage (BokehDiff repo as cwd): python bd43.py --test_data_dir <one input png> --output_dir <unused> ...
env: R43_DATA, R43_OUT, SMOKE
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402

DATA, OUT = os.environ["R43_DATA"], os.environ["R43_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
RIM_P, DIL = 8, 3                 # rim = 6 px at the 768 input scale = 8 px at the 1024 processing scale
_orig_val = I.log_validation


def proc_size(w, h):                                      # = bd40.proc_size
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def prepared(i, W2, H2):                                  # = bd41.prepared (bd40.prepared_disp + mask)
    d = np.load(f"{DATA}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = cv2.dilate(cv2.erode(d, np.ones((3, 3))), np.ones((3, 3)))
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.open(f"{DATA}/mask/{i}.png").resize((d.shape[1], d.shape[0]), resample=Image.LANCZOS)
                 .convert("L")) / 255.0
    f = float(np.median(d[m > 0.5]))
    d, f = 1 - d, 1 - f
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR)).astype(np.float32)
    mp = np.array(Image.open(f"{DATA}/mask/{i}.png").convert("L").resize((W2, H2), resample=Image.LANCZOS)) / 255.0
    return d, f, mp > 0.5


def seethrough(d, m):
    hole = cv2.dilate(m.astype(np.uint8), np.ones((2 * DIL + 1, 2 * DIL + 1), np.uint8)) > 0
    H2, W2 = d.shape
    ds = cv2.resize(d, (W2 // 4, H2 // 4), interpolation=cv2.INTER_AREA).astype(np.float64)
    hs = cv2.resize(hole.astype(np.uint8), (W2 // 4, H2 // 4), interpolation=cv2.INTER_NEAREST) > 0
    hs = cv2.dilate(hs.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    _, lab = cv2.distanceTransformWithLabels(hs.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~hs)
    lut = np.zeros(lab.max() + 1, np.int64)
    lut[lab[ys, xs]] = np.arange(len(ys))
    fill = ds.copy()
    fill[hs] = ds[ys[lut[lab[hs]]], xs[lut[lab[hs]]]]             # nearest outside value
    for _ in range(400):                                          # Laplace (harmonic) relaxation inside the hole
        p = np.pad(fill, 1, mode="edge")
        fill[hs] = ((p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4)[hs]
    fill = cv2.resize(fill.astype(np.float32), (W2, H2), interpolation=cv2.INTER_LINEAR)
    out = d.copy()
    out[hole] = fill[hole]
    return out


def condition(i, cond, W2, H2):
    d, f, m = prepared(i, W2, H2)
    if cond == "original":
        return d, f
    st = seethrough(d, m)
    if cond == "seethrough":
        return st, f
    din = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    rim = m & (din <= RIM_P)
    st[rim] = d[rim]
    return st, f


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    regions = json.load(open(f"{DATA}/regions43.json"))
    ids = [i for i in sorted(regions) if regions[i]["eligible"]]
    plan = {"original": ["8", "12", "K36", "K64"], "seethrough": ["8", "12"], "seethrough_rim": ["8", "12"]}
    if SMOKE:
        ids, plan = ids[:2], {c: ["12"] for c in plan}
    os.makedirs(f"{OUT}/img", exist_ok=True)
    os.makedirs(f"{OUT}/depth", exist_ok=True)
    rows = []
    for i in ids:
        rg = regions[i]
        x = Image.open(f"{DATA}/input/{i}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        for cond, ks in plan.items():
            d, f = condition(i, cond, W2, H2)
            np.save(f"{OUT}/depth/{i}_{cond}.npy", d.astype(np.float16))
            for kk in ks:
                K = rg["K"][kk]
                coc = np.abs(d - f) * K / 10
                dc = torch.from_numpy(np.stack([d, coc])[None]).to(dev, dt)
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                with torch.autocast("cuda", dtype=torch.float16):
                    image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                           aif_image=x_in, disp_coc=dc, timeit=True)
                tag = f"img/{i}_{cond}_{kk}"
                image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
                rows.append(dict(id=i, cond=cond, k_tag=kk, K=float(K), file=f"{tag}.png", focus=f, seconds=float(dur),
                                 depth=f"depth/{i}_{cond}.npy"))
                print(tag, f"K {K:.2f}", f"{dur:.2f}s", flush=True)
        json.dump(dict(smoke=SMOKE, rim_p=RIM_P, rows=rows), open(f"{OUT}/bd43.json", "w"), indent=1)
    json.dump(dict(smoke=SMOKE, rim_p=RIM_P, rows=rows), open(f"{OUT}/bd43.json", "w"), indent=1)
    print("bd43 done", len(rows), "images", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
