"""Round 39 step 1 scenes: one textured foreground cutout over a uniform or gradient background (fixed before any
generation).

Foreground: antialiased shape (4x4 supersampled coverage alpha) filled with a test-source texture (reflect-padded to
512 px), at disparity 1. Background: uniform colour or a planar sRGB gradient, at disparity 0 (the map spans [0, 1]).
  shapes     disk (r 120) | ellipse (140 x 100, rotated 30 deg) | rounded square (half-size 110, corner radius 30)
  textures   gravel, astronaut, chelsea, rocket          (round 24's test sources; not used for calibration)
  centres    cycled through (256, 256), (240, 272), (272, 240)
  background each (shape, texture) pair gets one uniform and one gradient background (cycled colours/directions)
  -> 3 x 4 x 2 = 24 scenes.
The all-in-focus input is composited in LINEAR light: lin(I) = alpha lin(F) + (1 - alpha) lin(B).
Settings (fg at 1, bg at 0; requests from the linear calibration curve):
  A  d_f = 1: foreground sharp, background blurred to radius r_b
  B  d_f = 0: background sharp, foreground blurred to r_f
  C  d_f = 0.5: both blurred (fg in front), each to r/2
  levels: r = 15.5 px (L1) and 11.625 px (L2); in C each layer gets r/2 (7.75 / 5.81 px).
Edge geometry: signed distance to the alpha = 0.5 contour (+ inside the foreground, - outside).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import data

S = 512
SS = 4
TEXTURES = ["gravel", "astronaut", "chelsea", "rocket"]
SHAPES = ["disk", "ellipse", "rsquare"]
CENTRES = [(256, 256), (240, 272), (272, 240)]
UNIFORM = [(128, 128, 128), (60, 90, 140), (200, 170, 120)]
GRADIENT = [((40, 40, 40), (220, 220, 220), 0.0), ((30, 60, 120), (230, 200, 150), 90.0),
            ((180, 60, 50), (60, 160, 200), 45.0)]
SETTINGS = {"A": 1.0, "B": 0.0, "C": 0.5}
LEVELS = {"L1": 15.5, "L2": 11.625}
BANDS = [(-1e9, -32), (-32, -16), (-16, -8), (-8, 0), (0, 8), (8, 16), (16, 32), (32, 1e9)]
BNAME = ["out>32", "out16-32", "out8-16", "out0-8", "in0-8", "in8-16", "in16-32", "in>32"]
EDGE_BANDS = ["out8-16", "out0-8", "in0-8", "in8-16"]
BORDER = 24                                                 # background interior excludes the image border


def to_lin(s):
    s = np.asarray(s, np.float64) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def to_srgb(l):
    l = np.clip(l, 0, None)
    return np.clip(np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055), 0, 1) * 255.0


def texture(name):
    a = getattr(data, name)()
    a = a[..., :3] if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
    H, W = a.shape[:2]
    a = np.pad(a, ((0, max(0, S - H)), (0, max(0, S - W)), (0, 0)), mode="reflect")
    return a[:S, :S].astype(np.float64)


def inside_fn(shape, cy, cx):
    if shape == "disk":
        return lambda y, x: (y - cy) ** 2 + (x - cx) ** 2 <= 120 ** 2
    if shape == "ellipse":
        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        return lambda y, x: (((x - cx) * c + (y - cy) * s) / 140) ** 2 + ((-(x - cx) * s + (y - cy) * c) / 100) ** 2 <= 1
    h, rc = 110, 30
    def f(y, x):
        qy, qx = np.maximum(np.abs(y - cy) - (h - rc), 0), np.maximum(np.abs(x - cx) - (h - rc), 0)
        return qy ** 2 + qx ** 2 <= rc ** 2
    return f


def alpha_of(shape, cy, cx):
    yy, xx = np.mgrid[:S, :S].astype(np.float64)
    f = inside_fn(shape, cy, cx)
    sub = (np.arange(SS) + 0.5) / SS
    return sum(f(yy + oy, xx + ox) for oy in sub for ox in sub) / SS ** 2


def background(kind, i):
    if kind == "uniform":
        return np.broadcast_to(np.array(UNIFORM[i], np.float64), (S, S, 3)).copy()
    c0, c1, ang = GRADIENT[i]
    yy, xx = np.mgrid[:S, :S].astype(np.float64)
    t = (np.cos(np.radians(ang)) * (xx - S / 2) + np.sin(np.radians(ang)) * (yy - S / 2))
    t = (t - t.min()) / (t.max() - t.min())
    return np.array(c0, np.float64) * (1 - t[..., None]) + np.array(c1, np.float64) * t[..., None]


def plan():
    out, n = [], 0
    for si, shape in enumerate(SHAPES):
        for ti, tex in enumerate(TEXTURES):
            for bk in ("uniform", "gradient"):
                cy, cx = CENTRES[n % 3]
                out.append(dict(name=f"s{n:02d}_{shape}_{tex}_{bk}", shape=shape, tex=tex, bg_kind=bk,
                                bg_i=(si + ti) % 3, centre=(cy, cx)))
                n += 1
    return out


def build(sc):
    """adds alpha, F (fg texture, sRGB), B (true background, sRGB), img (linear-light all-in-focus input, uint8),
    dist (signed distance), fgmask (alpha >= 0.5), disp."""
    a = alpha_of(sc["shape"], *sc["centre"])
    F, B = texture(sc["tex"]), background(sc["bg_kind"], sc["bg_i"])
    img = to_srgb(a[..., None] * to_lin(F) + (1 - a[..., None]) * to_lin(B))
    inside = a >= 0.5
    dist = np.where(inside, distance_transform_edt(inside), -distance_transform_edt(~inside))
    return dict(sc, alpha=a, F=F, B=B, img=np.round(img).astype(np.uint8), dist=dist, fgmask=inside,
                disp=inside.astype(np.float32))


def band_masks(dist):
    yy, xx = np.mgrid[:S, :S]
    border = (yy >= BORDER) & (yy < S - BORDER) & (xx >= BORDER) & (xx < S - BORDER)
    return {n: (dist > lo) & (dist <= hi) & border for n, (lo, hi) in zip(BNAME, BANDS)}


def reconstruct_bg(img, alpha, kind):
    """available-information background: estimated from VISIBLE background pixels only (alpha == 0).
    uniform -> per-channel mean; gradient -> per-channel least-squares plane in sRGB."""
    vis = alpha <= 0
    y = img.astype(np.float64)
    if kind == "uniform":
        return np.broadcast_to(y[vis].mean(0), (S, S, 3)).copy()
    yy, xx = np.mgrid[:S, :S].astype(np.float64)
    A = np.stack([np.ones(vis.sum()), yy[vis], xx[vis]], 1)
    Af = np.stack([np.ones(S * S), yy.ravel(), xx.ravel()], 1)
    return np.stack([(Af @ np.linalg.lstsq(A, y[vis][:, c], rcond=None)[0]).reshape(S, S) for c in range(3)], -1)


if __name__ == "__main__":
    for sc in plan():
        b = build(sc)
        rb = reconstruct_bg(b["img"], b["alpha"], sc["bg_kind"])
        bm = band_masks(b["dist"])
        print(sc["name"], "fg frac %.2f" % b["fgmask"].mean(), "bg recon max|err| %.2f" % np.abs(rb - b["B"]).max(),
              {n: int(m.sum()) for n, m in bm.items() if n in ("out>32", "in>32")})
