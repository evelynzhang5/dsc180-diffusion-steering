"""Round 40 stage 3: shared definitions (scene lists, splits, frozen review, image I/O, frozen constants).

Everything a stage 3 script needs to agree on lives here, so the drivers, the mapping and the scoring cannot drift.
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [4.0, 8.0, 12.0]                       # effective ADDITIONAL blur radius (px, at the 768-px input scale)
TOL = lambda r: max(2.0, 0.20 * r)               # frozen tolerance: control success and blur matching
R2_MIN = 0.90                                    # tile fit quality below this -> that tile is not used
MIN_TILES = 2                                    # fewer good tiles -> blur unmeasurable for that output
K_GRID = [4, 8, 12, 16, 20, 28, 36, 48, 64]      # BokehDiff aperture parameter (upstream default 20)
AV_GRID = [2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0]  # Bokehlicious f-number (model range 2-20)
K_SMOKE, AV_SMOKE = [8, 20, 48], [2.0, 5.6, 16.0]


def load(root):
    gen = json.load(open(f"{root}/gen40.json"))["rows"]
    rb = json.load(open(f"{root}/realbokeh40.json"))["selected"]
    regions = json.load(open(f"{root}/regions40.json"))
    review = json.load(open(f"{HERE}/review40.json"))
    scenes = {}
    for s in gen:
        scenes[s["id"]] = dict(id=s["id"], kind="gen", category=s["category"], split=s["split"])
    for s in rb:
        i = f"rb_{s['id']}"
        scenes[i] = dict(id=i, kind="rb", category="realbokeh", split=s["split"],
                         apertures=[float(a) for a in s["targets"]])
    return scenes, regions, review


def mapping_scenes(scenes, regions, review):
    """Development scenes used to fit the generated-image blur mapping: initially-sharp categories only, blur
    measurable (>= 2 behind tiles), not a mask failure, not measurement-flagged. Behind tiles only."""
    bad = set(review["mask_failure"]) | set(review["measurement_flag"])
    return [i for i, s in sorted(scenes.items()) if s["kind"] == "gen" and s["split"] == "dev"
            and s["category"] != "existing_blur" and regions[i]["blur_measurable"] and i not in bad]


def rb_dev_scenes(review):
    return list(review["realbokeh_dev_after_review"])


def img(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64)


def to_lin(x):
    x = np.clip(x / 255.0, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def subject(root, i, shape):
    H, W = shape
    m = np.asarray(Image.open(f"{root}/mask/{i}.png").convert("L").resize((W, H))).astype(np.float64) / 255
    return m >= 0.5


def depth(root, i, shape):
    """BokehDiff's prepared map, as stored (larger = farther; see README stage 3, depth convention)."""
    H, W = shape
    d = np.load(f"{root}/depth/{i}_pred.npy").astype(np.float32)
    if d.shape != (H, W):
        d = np.asarray(Image.fromarray(d).resize((W, H), resample=Image.BILINEAR))
    return d.astype(np.float64)
