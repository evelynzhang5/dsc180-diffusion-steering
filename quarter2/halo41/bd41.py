"""Round 41 step 1 (GPU): where does BokehDiff's sharp halo come from: the depth map it is given, or the model?

Same driver as round 40 (bd40.py: processing size, preprocessing, training depth convention, prompt, seed, fp16);
only the depth/disparity map handed to BokehDiff changes. Three conditions per scene x K:
  A  original   : the prepared map, exactly as round 40.
  B  two_layer  : subject (mask > 0.5) = focus; everything outside = ONE value, the median prepared disparity over the
                  scene's frozen behind tiles. The requested blur is then identical everywhere outside the subject,
                  so any extra sharpness next to the subject cannot come from the depth input.
  C  hard_edge  : subject = focus; outside pixels within R = 24 px (processing scale) of the mask take the value of
                  the nearest outside pixel at >= R px; everything else as A. Removes only the depth smoothing across
                  the edge, keeps genuine background depth variation (reported, not used for the decision).
Scenes: the 8 round 40 mapping scenes (development only; no test scene is rendered).
K: the three round 40 frozen settings (mapping.json, targets 4 / 8 / 12 px).
usage (BokehDiff repo as cwd): python bd41.py --test_data_dir <one input png> --output_dir <unused> ...
env: R40_DATA, R40_MAP (round 40 mapping.json), R41_OUT, SMOKE
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compare40"))
import common40 as C  # noqa: E402

DATA, OUT = os.environ["R40_DATA"], os.environ["R41_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
R_EDGE = 24
_orig_val = I.log_validation


def proc_size(w, h):                                      # = bd40.proc_size
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def prepared(i, W2, H2):
    """bd40.prepared_disp (training convention) plus the mask at processing size."""
    d = np.load(f"{DATA}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = cv2.dilate(cv2.erode(d, np.ones((3, 3))), np.ones((3, 3)))
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.open(f"{DATA}/mask/{i}.png").resize((d.shape[1], d.shape[0]), resample=Image.LANCZOS)
                 .convert("L")) / 255.0
    assert m.max() > 0, i                                  # mapping scenes have non-empty masks
    f = float(np.median(d[m > 0.5]))
    d, f = 1 - d, 1 - f
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR)).astype(np.float32)
    mp = np.array(Image.open(f"{DATA}/mask/{i}.png").convert("L").resize((W2, H2), resample=Image.LANCZOS)) / 255.0
    return d, f, mp > 0.5


def condition(i, cond, W2, H2, regions):
    d, f, m = prepared(i, W2, H2)
    if cond == "original":
        return d, f
    if cond == "two_layer":
        W, H = regions[i]["size"][1], regions[i]["size"][0]
        sx, sy = W2 / W, H2 / H
        vals = np.concatenate([d[int(t["y"] * sy):int((t["y"] + 64) * sy), int(t["x"] * sx):int((t["x"] + 64) * sx)]
                               .ravel() for t in regions[i]["tiles"]])
        out = np.full_like(d, float(np.median(vals)))
        out[m] = f
        return out, f
    if cond == "hard_edge":
        dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)       # distance outside to the mask
        src = (~m) & (dist >= R_EDGE)
        _, lab = cv2.distanceTransformWithLabels((~src).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(src)
        lut = np.zeros(lab.max() + 1, np.int64)
        lut[lab[ys, xs]] = np.arange(len(ys))
        k = lut[lab]
        out = d.copy()
        near = (~m) & ~src
        out[near] = d[ys[k[near]], xs[k[near]]]
        out[m] = f
        return out, f
    raise ValueError(cond)


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    scenes, regions, review = C.load(DATA)
    mp = json.load(open(os.environ["R40_MAP"]))["bokehdiff"]["gen"]
    Ks = {t: mp[t]["setting"] for t in ["4", "8", "12"]}
    ids = C.mapping_scenes(scenes, regions, review)
    conds = ["original", "two_layer", "hard_edge"]
    if SMOKE:
        ids, Ks = ids[:1], {"8": Ks["8"]}
    os.makedirs(f"{OUT}/img", exist_ok=True)
    os.makedirs(f"{OUT}/depth", exist_ok=True)
    rows = []
    for i in ids:
        x = Image.open(f"{DATA}/input/{i}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        for cond in conds:
            d, f = condition(i, cond, W2, H2, regions)
            np.save(f"{OUT}/depth/{i}_{cond}.npy", d.astype(np.float16))
            for t, K in Ks.items():
                coc = np.abs(d - f) * K / 10
                dc = torch.from_numpy(np.stack([d, coc])[None]).to(dev, dt)
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                with torch.autocast("cuda", dtype=torch.float16):
                    image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                           aif_image=x_in, disp_coc=dc, timeit=True)
                tag = f"img/{i}_{cond}_t{t}"
                image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
                rows.append(dict(id=i, cond=cond, target=float(t), K=float(K), file=f"{tag}.png", focus=f,
                                 seconds=float(dur)))
                print(tag, f"K {K:.2f}", f"{dur:.2f}s", flush=True)
    json.dump(dict(smoke=SMOKE, r_edge=R_EDGE, rows=rows), open(f"{OUT}/bd41.json", "w"), indent=1)
    print("bd41 done", len(rows), "images", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
