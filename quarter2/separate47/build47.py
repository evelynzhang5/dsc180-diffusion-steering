"""Round 47 (CPU): build the T2 held-out-family composites (text, dots) on round 46's 6 backgrounds, with round 46's
pane geometry, maps, focus and K (read from round 46's compose46.json). FROZEN before training.
usage: python build47.py <round 46 data> <round 47 T2 data>
"""
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth47 as S  # noqa: E402

SRC, DST = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{SRC}/compose46.json"))
for d in ["input", "alpha", "surf", "maps"]:
    os.makedirs(f"{DST}/{d}", exist_ok=True)
pane = np.load(f"{SRC}/maps/pane.npy")
shutil.copy(f"{SRC}/maps/pane.npy", f"{DST}/maps/pane.npy")
rng = np.random.default_rng(47)
surf = {k: S.surface_family(rng, pane, k) for k in S.HELDOUT_FAMILIES}
out = dict(meta, composites={}, types=S.HELDOUT_FAMILIES)
for bg in meta["bgs"]:
    shutil.copy(f"{SRC}/maps/{bg}_dB.npy", f"{DST}/maps/{bg}_dB.npy")
    shutil.copy(f"{SRC}/input/{bg}__bgonly.png", f"{DST}/input/{bg}__bgonly.png")
    B = S.to_lin(np.asarray(Image.open(f"{SRC}/input/{bg}__bgonly.png").convert("RGB")) / 255.0)
    for k, (a, Fs) in surf.items():
        I = a[..., None] * Fs + (1 - a[..., None]) * B
        c = f"{bg}__{k}"
        Image.fromarray((S.to_srgb(I) * 255).round().astype(np.uint8)).save(f"{DST}/input/{c}.png")
        np.save(f"{DST}/alpha/{c}.npy", a.astype(np.float32))
        np.save(f"{DST}/surf/{c}.npy", Fs.astype(np.float16))
        inside = pane > 0.5
        out["composites"][c] = dict(bg=bg, type=k, control=False,
                                    regions_px=dict(surface=int((inside & (a >= 0.8)).sum()),
                                                    transmission=int((inside & (a <= 0.2)).sum()),
                                                    overlap=int((inside & (a > 0.2) & (a < 0.8)).sum())))
        print(c, out["composites"][c]["regions_px"], flush=True)
json.dump(out, open(f"{DST}/compose47.json", "w"), indent=1)
