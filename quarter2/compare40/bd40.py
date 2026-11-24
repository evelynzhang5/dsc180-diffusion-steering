"""Round 40 stage 3 (GPU): BokehDiff on the round 40 scenes, driven exactly as its own inference does, except where noted.

Per image (as upstream inference_hf.py + dataset.py TestDataset):
  - image resized (Lanczos) so the short side is >= 1024. DEVIATION: the size is rounded to a multiple of 64 (upstream:
    8), because the attention processor recovers its down-sampling factor from the disparity map's size, which is
    exact only for multiples of 64 (4:3 RealBokeh images would otherwise mismatch); output resized back (Lanczos).
  - disparity: BokehDiff's prepared map (prepare_data.py), normalised and morphologically clipped as dataset.py does.
  - focus: median disparity inside the BiRefNet mask (> 0.5); an empty mask falls back to the image centre box.
  - blur map: |disparity - focus| * K / 10 (upstream amplify, upsample 1); prompt "an excellent photo with a large
    aperture", train_T 499, seed 0, fp16.
DEPTH CONVENTION (frozen before any output): prepare_data.py stores 1 - normalised(Depth Anything output), i.e. larger
= FARTHER, but BokehDiff was trained on disparity with larger = NEARER (dataset.py: disp = 1/depth; the attention
processor samples "from the point to the camera (disp=1)"). The blur map |d - f| is identical either way; the
occlusion ordering is not. Main runs use the training convention (1 - stored map). Diagnostic: the mapping scenes are
also run at K = 20 with the stored (upstream) convention, reported, not used for any decision.
Phases (PHASE env): dev = K grid on all development scenes (+ diagnostic); test = frozen settings from mapping.json.
usage (BokehDiff repo as cwd): python bd40.py --test_data_dir <one input png> --output_dir <unused> ...
env: R40_DATA (stage 1 data root), R40_OUT (stage 3 output root), PHASE, SMOKE
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common40 as C  # noqa: E402

DATA, OUT = os.environ["R40_DATA"], os.environ["R40_OUT"]
PHASE = os.environ["PHASE"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
_orig_val = I.log_validation


def proc_size(w, h):
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def prepared_disp(i, W2, H2, train_convention=True):
    d = np.load(f"{DATA}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = cv2.dilate(cv2.erode(d, np.ones((3, 3))), np.ones((3, 3)))
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.open(f"{DATA}/mask/{i}.png").resize((d.shape[1], d.shape[0]), resample=Image.LANCZOS)
                 .convert("L")) / 255.0
    if m.max() == 0:
        h, w = m.shape
        m[int(h * .4):int(h * .6), int(w * .4):int(w * .6)] = 1
    f = float(np.median(d[m > 0.5]))
    if train_convention:
        d, f = 1 - d, 1 - f
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR))
    return d.astype(np.float32), f


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    scenes, regions, review = C.load(DATA)
    rows = []

    def run(i, K, tag, train_convention=True):
        x = Image.open(f"{DATA}/input/{i}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        d, f = prepared_disp(i, W2, H2, train_convention)
        coc = np.abs(d - f) * K / 10
        dc = torch.from_numpy(np.stack([d, coc])[None]).to(dev, dt)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                   aif_image=x_in, disp_coc=dc, timeit=True)
        torch.cuda.synchronize()
        image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
        rows.append(dict(id=i, K=float(K), file=f"{tag}.png", seconds=float(dur), proc_size=[W2, H2], focus=f,
                         train_convention=train_convention))
        print(tag, f"{dur:.2f}s", flush=True)

    if PHASE == "dev":
        os.makedirs(f"{OUT}/bd_dev", exist_ok=True)
        dev_ids = [i for i, s in sorted(scenes.items()) if s["split"] == "dev"]
        grid = C.K_SMOKE if SMOKE else C.K_GRID
        if SMOKE:
            dev_ids = C.mapping_scenes(scenes, regions, review)[:2] + C.rb_dev_scenes(review)[:1]
        for i in dev_ids:
            for K in grid:
                run(i, K, f"bd_dev/{i}_K{K:g}")
        for i in C.mapping_scenes(scenes, regions, review)[:2 if SMOKE else None]:
            run(i, 20, f"bd_dev/{i}_K20_upstreamconv", train_convention=False)
    else:
        os.makedirs(f"{OUT}/bd_test", exist_ok=True)
        mp = json.load(open(f"{OUT}/mapping.json"))["bokehdiff"]
        test = [i for i, s in sorted(scenes.items()) if s["split"] == "test"]
        if SMOKE:
            test = [i for i in test if scenes[i]["kind"] == "gen"][:2] + [i for i in test if scenes[i]["kind"] == "rb"][:1]
        for i in test:
            if scenes[i]["kind"] == "gen":
                for t in C.TARGETS:
                    run(i, mp["gen"][f"{t:g}"]["setting"], f"bd_test/{i}_t{t:g}")
            else:
                for a in ["2.0", "second"]:
                    run(i, mp["rb"][a]["setting"], f"bd_test/{i}_{a}")
    json.dump(dict(phase=PHASE, smoke=SMOKE, rows=rows), open(f"{OUT}/bd_{PHASE}.json", "w"), indent=1)
    print("bokehdiff", PHASE, "done", len(rows), "images; median render", np.median([r["seconds"] for r in rows]),
          "s", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
