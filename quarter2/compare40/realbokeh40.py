"""Round 40 stage 1: RealBokeh real-photo control (Seizinger et al. 2025; test split, 3 MP).

Each test scene has an f/22 source capture and real captures at f/2.0 plus a few randomly chosen apertures. The
scene-aperture pairs are selected from the actual metadata (frozen rule):
  candidates  scenes whose targets include f/2.0 AND at least one aperture in [4.0, 8.0]
  sample      25 candidates, numpy RandomState(0) without replacement; the first 5 are calibration (dev) scenes,
              the other 20 test scenes
  targets     f/2.0 and the available aperture in [4.0, 8.0] closest to 5.6
Images are centre-cropped to 4:3 (orientation kept) and resized (Lanczos) to 1024x768 / 768x1024.
usage: python realbokeh40.py <data root>   -> <root>/input/rb_<id>.png, <root>/targets/rb_<id>_f<av>.png, realbokeh40.json
"""
import glob
import json
import os
import sys

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

ROOT = sys.argv[1]
SMOKE = os.environ.get("SMOKE", "0") == "1"
REPO = "timseizinger/RealBokeh_3MP"
for d in ("input", "targets"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
cache = snapshot_download(REPO, repo_type="dataset", allow_patterns=["test/metadata/*"])
meta = [json.load(open(f)) for f in sorted(glob.glob(f"{cache}/test/metadata/*.json"))]
cand = [m for m in meta if 2.0 in m["target_avs"] and any(4.0 <= a <= 8.0 for a in m["target_avs"])]
cand.sort(key=lambda m: m["id"])
pick = [cand[i] for i in np.random.RandomState(0).choice(len(cand), 25, replace=False)]
sel = []
for j, m in enumerate(pick):
    mid = min([a for a in m["target_avs"] if 4.0 <= a <= 8.0], key=lambda a: abs(a - 5.6))
    sel.append(dict(id=m["id"], split="dev" if j < 5 else "test", targets=[2.0, mid],
                    files={str(a): f for a, f in zip(m["target_avs"], m["target_images"]) if a in (2.0, mid)},
                    source=m["source_image"], focal_length=m["focal_length"], focus_plane_distance=m["focus_plane_distance"]))
if SMOKE:
    sel = sel[:2]
pats = [f"test/{s['source']}" for s in sel] + [f"test/{f}" for s in sel for f in s["files"].values()]
cache = snapshot_download(REPO, repo_type="dataset", allow_patterns=pats)


def prep(path):
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


for s in sel:
    prep(f"{cache}/test/{s['source']}").save(f"{ROOT}/input/rb_{s['id']}.png")
    for a, f in s["files"].items():
        prep(f"{cache}/test/{f}").save(f"{ROOT}/targets/rb_{s['id']}_f{float(a)}.png")
    print("realbokeh", s["id"], s["split"], s["targets"], flush=True)
json.dump(dict(n_scenes=len(meta), n_candidates=len(cand), selected=sel), open(f"{ROOT}/realbokeh40.json", "w"), indent=1)
print(f"RealBokeh: {len(meta)} test scenes, {len(cand)} candidates, {len(sel)} selected", flush=True)
