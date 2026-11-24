"""Round 40 stage 3 (GPU): Bokehlicious (Seizinger et al., ICCV 2025; small checkpoint) on the round 40 scenes.

Input: the RGB image and an f-number only (no depth, no mask: it chooses its own focus). Images at native size
(768x768 generated, 1024x768 RealBokeh; the loader's max_dim 2000 does not resize them). Rendering-only latency is
the model forward pass (synchronised); Bokehlicious needs no depth or segmentation, so its total equals it.
Phases (PHASE env): dev = f-number grid on all development scenes; test = frozen settings from mapping.json
(interpolated f-numbers are valid: the aperture embedding is continuous).
usage (Bokehlicious repo as cwd): python bkl40.py     env: R40_DATA, R40_OUT, PHASE, SMOKE
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from dataset.util import load_image  # noqa: E402
from method.config import bokehlicious_size_builder  # noqa: E402
from method.model import Bokehlicious  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common40 as C  # noqa: E402

DATA, OUT = os.environ["R40_DATA"], os.environ["R40_OUT"]
PHASE = os.environ["PHASE"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
model = Bokehlicious(**bokehlicious_size_builder("small"))
model.load_state_dict(torch.load("./checkpoints/small.pt", map_location="cpu"))
model.to("cuda").eval()
scenes, regions, review = C.load(DATA)
rows = []


def run(i, av, tag):
    path = f"{DATA}/input/{i}.png"
    inp = load_image(path, target_av=float(av), max_dim=2000, device="cuda")
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad():
        out = model(**inp)
    torch.cuda.synchronize()
    dur = time.time() - t
    y = np.round(out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    x = Image.open(path)
    assert y.shape[:2] == (x.height, x.width), (y.shape, x.size)
    Image.fromarray(y).save(f"{OUT}/{tag}.png")
    rows.append(dict(id=i, av=float(av), file=f"{tag}.png", seconds=float(dur)))
    print(tag, f"{dur:.3f}s", flush=True)


if PHASE == "dev":
    os.makedirs(f"{OUT}/bkl_dev", exist_ok=True)
    dev_ids = [i for i, s in sorted(scenes.items()) if s["split"] == "dev"]
    grid = C.AV_SMOKE if SMOKE else C.AV_GRID
    if SMOKE:
        dev_ids = C.mapping_scenes(scenes, regions, review)[:2] + C.rb_dev_scenes(review)[:1]
    run(dev_ids[0], 2.0, "bkl_dev/warmup")
    rows.clear()
    for i in dev_ids:
        for av in grid:
            run(i, av, f"bkl_dev/{i}_f{av:g}")
else:
    os.makedirs(f"{OUT}/bkl_test", exist_ok=True)
    mp = json.load(open(f"{OUT}/mapping.json"))["bokehlicious"]
    test = [i for i, s in sorted(scenes.items()) if s["split"] == "test"]
    if SMOKE:
        test = [i for i in test if scenes[i]["kind"] == "gen"][:2] + [i for i in test if scenes[i]["kind"] == "rb"][:1]
    run(test[0], 2.0, "bkl_test/warmup")
    rows.clear()
    for i in test:
        if scenes[i]["kind"] == "gen":
            for t in C.TARGETS:
                run(i, mp["gen"][f"{t:g}"]["setting"], f"bkl_test/{i}_t{t:g}")
        else:
            for a in ["2.0", "second"]:
                run(i, mp["rb"][a]["setting"], f"bkl_test/{i}_{a}")
json.dump(dict(phase=PHASE, smoke=SMOKE, rows=rows), open(f"{OUT}/bkl_{PHASE}.json", "w"), indent=1)
print("bokehlicious", PHASE, "done", len(rows), "images; median render", np.median([r["seconds"] for r in rows]), "s",
      flush=True)
