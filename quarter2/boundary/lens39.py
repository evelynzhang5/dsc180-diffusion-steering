"""Round 39 reference renderers for two-layer boundary scenes (CPU, linear light).

Geometry. Fronto-parallel layers, each a colour image F_l and opacity alpha_l given in projected image coordinates, at
disparity d_l (larger = nearer). Thin lens focused at disparity d_f. A ray for image point x through aperture point a
(unit disk) meets layer l at image coordinate x + a * c_l with signed blur radius c_l = R * (d_l - d_f) px
(R = NOM * K px per unit disparity, BokehDiff's convention for the nominal radius). For planar layers this is exact
thin-lens geometry (magnification absorbed into the projected coordinates).

raytrace   aperture-sampled thin lens: per aperture sample, layers composited front to back (nearest first, opacity
           as transmission), averaged over samples. Occlusion-correct for any focus setting. The reference.
composite  analytic premultiplied layering, I = K_f * (alpha F) + (1 - K_f * alpha) * (K_b * B), with K_l the aperture
           kernel of layer l. Exact for one layer in focus; an approximation when both layers are blurred.
naive      blur colour and opacity separately and composite with the blurred opacity (non-premultiplied), a common
           mistake; for comparison only.
All blending is in linear light; inputs/outputs are sRGB uint8-range floats.
"""
import numpy as np

NOM, K = 31 * 16 / 51.2 / 10, 40.0
R = NOM * K


def to_lin(s):
    s = np.asarray(s, np.float64) / 255.0
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def to_srgb(l):
    l = np.clip(l, 0, None)
    return np.clip(np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055), 0, 1) * 255.0


def aperture(n_side=40):
    """stratified samples in the unit disk (concentric mapping of an n_side x n_side grid)."""
    u = (np.arange(n_side) + 0.5) / n_side * 2 - 1
    a, b = np.meshgrid(u, u)
    a, b = a.ravel(), b.ravel()
    r = np.where(np.abs(a) > np.abs(b), a, b)
    th = np.where(np.abs(a) > np.abs(b), np.pi / 4 * b / np.where(a == 0, 1, a),
                  np.pi / 2 - np.pi / 4 * a / np.where(b == 0, 1, b))
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)            # (n, 2) as (dx, dy)


def _shift(P, pad, dx, dy, H, W):
    """bilinear sample P (padded by pad) at (y + dy, x + dx) for the H x W grid."""
    x0, y0 = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = dx - x0, dy - y0
    ys, xs = pad + y0, pad + x0
    s = lambda oy, ox: P[ys + oy: ys + oy + H, xs + ox: xs + ox + W]
    return (1 - fy) * ((1 - fx) * s(0, 0) + fx * s(0, 1)) + fy * ((1 - fx) * s(1, 0) + fx * s(1, 1))


def raytrace(layers, d_f, n_side=40):
    """layers: list of dict(F sRGB HxWx3, alpha HxW, d). Returns sRGB image."""
    H, W = layers[0]["alpha"].shape
    L = sorted(layers, key=lambda l: -l["d"])                       # nearest first
    pads, Ps = [], []
    for l in L:
        c = R * (l["d"] - d_f)
        pad = int(np.ceil(abs(c))) + 2
        P = np.concatenate([to_lin(l["F"]) * l["alpha"][..., None], l["alpha"][..., None]], -1)   # premultiplied
        Ps.append((np.pad(P, ((pad, pad), (pad, pad), (0, 0)), mode="edge"), pad, c))
    acc = np.zeros((H, W, 3))
    S = aperture(n_side)
    for ax, ay in S:
        T = np.ones((H, W, 1))
        out = np.zeros((H, W, 3))
        for P, pad, c in Ps:
            v = _shift(P, pad, ax * c, ay * c, H, W)
            out += T * v[..., :3]
            T = T * (1 - v[..., 3:])
        acc += out
    return to_srgb(acc / len(S))


def kernel_blur(X, c, n_side=40):
    """the same sampled aperture kernel applied to one (linear, premultiplied) array: K_c * X."""
    H, W = X.shape[:2]
    pad = int(np.ceil(abs(c))) + 2
    P = np.pad(X, ((pad, pad), (pad, pad)) + ((0, 0),) * (X.ndim - 2), mode="edge")
    S = aperture(n_side)
    return sum(_shift(P, pad, ax * c, ay * c, H, W) for ax, ay in S) / len(S)


def composite(fg, bg, d_f, n_side=40):
    cf, cb = R * (fg["d"] - d_f), R * (bg["d"] - d_f)
    aF = kernel_blur(to_lin(fg["F"]) * fg["alpha"][..., None], cf, n_side)
    a = kernel_blur(fg["alpha"][..., None], cf, n_side)
    B = kernel_blur(to_lin(bg["F"]), cb, n_side)
    return to_srgb(aF + (1 - a) * B)


def naive(fg, bg, d_f, n_side=40):
    cf, cb = R * (fg["d"] - d_f), R * (bg["d"] - d_f)
    F = kernel_blur(to_lin(fg["F"]), cf, n_side)
    a = kernel_blur(fg["alpha"][..., None], cf, n_side)
    B = kernel_blur(to_lin(bg["F"]), cb, n_side)
    return to_srgb(a * F + (1 - a) * B)
