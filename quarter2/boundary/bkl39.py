"""Round 39 step 1, second baseline (GPU): Bokehlicious (Seizinger et al., ICCV 2025; small checkpoint) on setting A.

Bokehlicious takes an RGB image and an f-stop; it has no depth or focus input and keeps the focus it infers from the
image. Only setting A (foreground sharp, background blurred) has a well-posed counterpart. The same all-in-focus
inputs, references (run39's ref/*_A_L*.npy) and measurement (measure39) as BokehDiff. Its adjustment knob is the
f-stop: AV in {2.0, 2.8, 4.0, 5.6, 8.0, 11, 16}; per scene and level the best valid f-stop (lowest E), as for
BokehDiff's request search. Latency per image is recorded.
usage (repo root of Bokehlicious as cwd): python bkl39.py <R39_OUT>
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure39 as Q  # noqa: E402
import scenes39 as S39  # noqa: E402

OUT = sys.argv[1]
SMOKE = os.environ.get("SMOKE", "0") == "1"
AV = [2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0]
os.makedirs(f"{OUT}/bkl", exist_ok=True)
model = Bokehlicious(**bokehlicious_size_builder("small"))
model.load_state_dict(torch.load("./checkpoints/small.pt", map_location="cpu"))
model.to("cuda").eval()
rows, lat = [], []
plan = S39.plan()[:1] if SMOKE else S39.plan()
levels = ["L1"] if SMOKE else list(S39.LEVELS)
for sc in plan:
    b = S39.build(sc)
    b["bands"] = S39.band_masks(b["dist"])
    path = f"{OUT}/input/{sc['name']}.png"
    assert np.array_equal(np.asarray(Image.open(path)), b["img"]), "input mismatch"
    refs = {lv: np.load(f"{OUT}/ref/{sc['name']}_A_{lv}.npy").astype(np.float64) for lv in levels}
    for av in AV:
        inp = load_image(path, target_av=av, max_dim=2000, device="cuda")
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            out = model(**inp)
        torch.cuda.synchronize()
        lat.append(time.time() - t)
        y = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0)
        y = np.round(y).astype(np.uint8)
        assert y.shape == b["img"].shape, y.shape
        Image.fromarray(y).save(f"{OUT}/bkl/{sc['name']}_f{av}.png")
        for lv in levels:
            m = Q.measure(y, refs[lv], b, "A", dict(bg=S39.LEVELS[lv]), None, b["img"])
            rows.append(dict(scene=sc["name"], shape=sc["shape"], tex=sc["tex"], bg=sc["bg_kind"], setting="A",
                             level=lv, av=av, **m))
    best = {lv: min([r for r in rows if r["scene"] == sc["name"] and r["level"] == lv],
                    key=lambda r: (not r["valid"], r["E"])) for lv in levels}
    print(sc["name"], " ".join(f"{lv}: f{r['av']} E {r['E']:.2f} valid {r['valid']}" for lv, r in best.items()), flush=True)
json.dump(dict(rows=rows, latency_s=dict(median=float(np.median(lat[1:] or lat)), n=len(lat)), av=AV, smoke=SMOKE),
          open(f"{OUT}/bkl.json", "w"), indent=1, default=float)
print("bokehlicious done", len(rows), "rows; median latency", np.median(lat[1:] or lat), flush=True)
