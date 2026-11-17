"""Round 18 test scenes: three all-in-focus 512x512 layered scenes with known depth.

Each scene is a set of fronto-parallel textured planes (piecewise-constant depth). Textures are sharp grayscale
photos from scikit-image (brick, grass, gravel), tinted, used at native resolution (no upsampling, so the input
has no blur of its own). Each texture appears at two or more depths, so texture cannot explain a blur difference.

make(name) -> dict(image uint8 HxWx3, depth z, disparity 1/z, labels int, regions list, focus dict).
"""
import numpy as np
from skimage import data

S = 512
TINT = dict(grass=(0.78, 1.0, 0.62), gravel=(1.0, 0.92, 0.80), brick=(1.0, 0.72, 0.62))
TEX = dict(grass=data.grass, gravel=data.gravel, brick=data.brick)
MARGIN_EDGE, MARGIN_BORDER = 48, 24   # interiors stay this far from depth edges / the image border


def _tex(name, y0, y1, x0, x1):
    rot = name.endswith("_r")                         # "_r": same texture rotated 90 degrees (different patch content)
    name = name[:-2] if rot else name
    g = TEX[name]().astype(np.float32) / 255.0
    g = np.rot90(g) if rot else g
    g = 0.15 + 0.8 * (g - g.min()) / (g.max() - g.min())       # keep away from clipping
    return g[y0:y1, x0:x1, None] * np.asarray(TINT[name], np.float32)


# (label, texture, z, y0, y1, x0, x1); later entries are drawn on top
LAYOUTS = {
    "A": [  # four quadrants at four depths; grass at z=1 and 8, gravel at 1.6 and 3
        (0, "grass", 1.0, 0, 256, 0, 256), (1, "gravel", 1.6, 0, 256, 256, 512),
        (2, "gravel", 3.0, 256, 512, 0, 256), (3, "grass", 8.0, 256, 512, 256, 512)],
    "B": [  # 2x3 grid, three depths; each texture at two depths, each depth with two textures
        (0, "brick", 1.2, 0, 256, 0, 171), (1, "gravel", 2.5, 0, 256, 171, 342), (2, "grass", 6.0, 0, 256, 342, 512),
        (3, "grass", 2.5, 256, 512, 0, 171), (4, "brick", 6.0, 256, 512, 171, 342), (5, "gravel", 1.2, 256, 512, 342, 512)],
    "C": [  # central card in front of a two-part background; gravel at z=1 and 10
        (0, "grass", 4.0, 0, 512, 0, 256), (1, "gravel", 10.0, 0, 512, 256, 512), (2, "gravel_r", 1.0, 156, 356, 156, 356)],
}
# focus: 'near' = nearest plane; 'mid' = a plane with planes both in front and behind (or the middle depth)
FOCUS = {"A": dict(near=1.0, mid=3.0), "B": dict(near=1.2, mid=2.5), "C": dict(near=1.0, mid=4.0)}


def interior(labels, lab):
    from scipy.ndimage import distance_transform_edt
    m = labels == lab
    pad = np.pad(m, 1, mode="edge")                  # image border is not a depth edge
    d = distance_transform_edt(pad)[1:-1, 1:-1]
    yy, xx = np.mgrid[:S, :S]
    border = np.minimum(np.minimum(yy, S - 1 - yy), np.minimum(xx, S - 1 - xx))
    return m & (d > MARGIN_EDGE) & (border >= MARGIN_BORDER)


def make(name):
    img = np.zeros((S, S, 3), np.float32)
    z = np.zeros((S, S), np.float32)
    lab = np.full((S, S), -1, np.int32)
    regions = []
    for (l, t, zz, y0, y1, x0, x1) in LAYOUTS[name]:
        img[y0:y1, x0:x1] = _tex(t, y0, y1, x0, x1)
        z[y0:y1, x0:x1] = zz
        lab[y0:y1, x0:x1] = l
        regions.append(dict(label=l, texture=t, z=zz))
    assert (lab >= 0).all()
    disp = 1.0 / z
    for r in regions:
        r["interior_px"] = int(interior(lab, r["label"]).sum())
        assert r["interior_px"] >= 64 * 64, (name, r)
    return dict(image=np.clip(np.round(img * 255), 0, 255).astype(np.uint8), depth=z, disparity=disp,
                labels=lab, regions=regions, focus=FOCUS[name])


if __name__ == "__main__":
    import json
    import sys
    from PIL import Image
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for n in LAYOUTS:
        s = make(n)
        Image.fromarray(s["image"]).save(f"{out}/scene{n}.png")
        print(n, json.dumps(s["regions"]), s["focus"])
