"""Round 43 stage 1: real transparent scenes with real wide-aperture captures (RealBokeh_3MP test split).

Selection (frozen, by eye, before any method ran on them): all 220 test-split f/22 inputs were viewed as contact
sheets on 2026-11-24 and every scene whose subject is a clear object with scene content visible through it was kept:
  48, 129  clear acrylic sign on a bar, shelves behind (strongest; 48 was also in round 40's RealBokeh sample)
  49       dog seen through a car windscreen        51   window with a vase on the sill
  116      glass pepper grinder on a table           131  window with a town view behind
Each scene keeps its f/22 source (method input) and the real f/2.0 capture (reference). Same crop/resize as round 40.
usage: python rb43.py <data root>   -> <root>/input/rb_<id>.png, <root>/targets/rb_<id>_f2.0.png, rb43.json
"""
import json
import os
import sys

from huggingface_hub import snapshot_download
from PIL import Image

ROOT = sys.argv[1]
SMOKE = os.environ.get("SMOKE", "0") == "1"
IDS = [48, 129, 49, 51, 116, 131]
if SMOKE:
    IDS = IDS[:1]
REPO = "timseizinger/RealBokeh_3MP"
for d in ("input", "targets"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
cache = snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"test/metadata/{i}.json" for i in IDS])
meta = {i: json.load(open(f"{cache}/test/metadata/{i}.json")) for i in IDS}
files = {i: dict(zip(m["target_avs"], m["target_images"]))[2.0] for i, m in meta.items()}
cache = snapshot_download(REPO, repo_type="dataset",
                          allow_patterns=[f"test/{m['source_image']}" for m in meta.values()]
                          + [f"test/{f}" for f in files.values()])


def prep(path):                                   # = realbokeh40.prep
    im = Image.open(path).convert("RGB")
    w, h = im.size
    land = w >= h
    tw, th = (4, 3) if land else (3, 4)
    if w * th > h * tw:
        nw = h * tw // th
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = w * th // tw
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize((1024, 768) if land else (768, 1024), resample=Image.LANCZOS)


rows = []
for i, m in meta.items():
    prep(f"{cache}/test/{m['source_image']}").save(f"{ROOT}/input/rb_{i}.png")
    prep(f"{cache}/test/{files[i]}").save(f"{ROOT}/targets/rb_{i}_f2.0.png")
    rows.append(dict(id=f"rb_{i}", source=m["source_image"], target=files[i],
                     focus_plane_distance=m["focus_plane_distance"], focal_length=m["focal_length"]))
    print("realbokeh", i, flush=True)
json.dump(dict(rows=rows), open(f"{ROOT}/rb43.json", "w"), indent=1)
