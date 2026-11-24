"""Round 40 measures for an output y of an all-in-focus input x with an in-focus subject (mask). No reference image is
needed, so they apply to generated images. Every boundary measure compares the edge with the image's own interior, so
a correct, uniformly blurred background scores ~0. Exploratory until validated on synthetic controls (validate40.py).

Regions: din / dout = distance inside / outside the subject contour (px); bg tiles from regions40.

  halo_sharp   residual background sharpness next to the subject: HF retention (output / input high-frequency
               energy, Gaussian sigma 1 px) in the outer band 2-12 px, minus the median HF retention over the
               background tiles. > 0 means the background next to the subject was left sharper than further away.
               Depth-gated (added before stage 3, after review): when a depth map is given, only band pixels with
               |depth - focus| >= 0.15 count, so content outside the mask that lies near the focus plane (a branch
               the subject sits on) and is correctly kept sharp is not scored as halo. halo_px / halo_kept report
               how many band pixels remained; with < 500 px the score is NaN (not measurable).
  rim          max |signed luminance change (y - x)| in the 0-3 px band on either side of the contour, minus that
               side's typical change (inner: subject interior >= 8 px; outer: background tiles). Bright or dark
               outline.
  bleed        subject colour spilling outward: in the outer 0-8 px band, (y - x) projected on the unit vector from
               the local background colour (input, outer 8-24 px) to the subject edge colour (input, inner 0-8 px),
               minus the same projection's median over the background tiles (/255).
  thin_loss    1 - (HF retention in the inner 0-6 px band) / (HF retention in the subject interior >= 8 px).
               > 0 means edge detail (hair, spokes) is lost more than interior detail.
Preservation (subject interior >= 8 px; reference = the original input):
  colour_shift mean |per-channel median (y - x)|
  detail       gain/offset-invariant residual: per-channel least squares y ~ a x + b, mean |residual|
  hf_ratio     HF energy y / x (> 1 can be noise/ringing, not restored sharpness)
  noise        HF energy of (y - x) after removing the gain/offset fit, relative to the input HF energy; with
               hf_ratio this separates detail loss (hf_ratio < 1) from added noise or ringing.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

LUM = np.array([0.2126, 0.7152, 0.0722])


def hf(z):
    z = z.astype(np.float64)
    return z - np.stack([gaussian_filter(z[..., c], 1.0) for c in range(z.shape[2])], -1)


def geometry(subj):
    return distance_transform_edt(subj), distance_transform_edt(~subj)


def tile_mask(shape, tiles, size=64):
    m = np.zeros(shape, bool)
    for t in tiles:
        m[t["y"]:t["y"] + size, t["x"]:t["x"] + size] = True
    return m


def _hfr(hy, hx, m):
    return float((hy[m] ** 2).mean() / max((hx[m] ** 2).mean(), 1e-9))


def boundary(x, y, subj, tiles, depth=None, focus=None, gap=0.15, min_px=500):
    x, y = x.astype(np.float64), y.astype(np.float64)
    din, dout = geometry(subj)
    hx, hy = hf(x), hf(y)
    T = tile_mask(subj.shape, tiles)
    tile_r = [_hfr(hy, hx, tile_mask(subj.shape, [t])) for t in tiles]
    far = float(np.median(tile_r)) if tile_r else float("nan")
    band = (dout >= 2) & (dout <= 12)
    outer = band & (np.abs(depth - focus) >= gap) if depth is not None else band
    halo_px = int(outer.sum())
    halo = _hfr(hy, hx, outer) - far if halo_px >= min_px else float("nan")
    dl = (y - x) @ LUM
    interior = din >= 8
    ri = float(dl[(din > 0) & (din <= 3)].mean() - np.median(dl[interior]))
    ro = float(dl[(dout > 0) & (dout <= 3)].mean() - (np.median(dl[T]) if T.any() else 0.0))
    rim = ri if abs(ri) >= abs(ro) else ro
    c_s = x[(din > 0) & (din <= 8)].mean(0)
    c_b = x[(dout > 8) & (dout <= 24)].mean(0)
    v = c_s - c_b
    v = v / max(np.linalg.norm(v), 1e-6)
    proj = (y - x) @ v
    bleed = float(proj[(dout > 0) & (dout <= 8)].mean() - (np.median(proj[T]) if T.any() else 0.0))
    edge_in = (din > 0) & (din <= 6)
    thin = 1 - _hfr(hy, hx, edge_in) / max(_hfr(hy, hx, interior), 1e-9)
    return dict(halo_sharp=float(halo), rim=rim, rim_inner=ri, rim_outer=ro, bleed=bleed, thin_loss=float(thin),
                far_hf_retention=far, halo_px=halo_px, halo_kept=halo_px / max(int(band.sum()), 1))


def preservation(x, y, subj):
    x, y = x.astype(np.float64), y.astype(np.float64)
    din, _ = geometry(subj)
    m = din >= 8
    d = y[m] - x[m]
    shift = float(np.abs(np.median(d, 0)).mean())
    res = np.zeros_like(y)
    det = []
    for c in range(3):
        A = np.stack([x[..., c][m], np.ones(m.sum())], 1)
        g, o = np.linalg.lstsq(A, y[..., c][m], rcond=None)[0]
        r = y[..., c] - (g * x[..., c] + o)
        res[..., c] = r
        det.append(np.abs(r[m]).mean())
    hx, hy, hr = hf(x), hf(y), hf(res)
    return dict(colour_shift=shift, detail=float(np.mean(det)), hf_ratio=_hfr(hy, hx, m),
                noise=float((hr[m] ** 2).mean() / max((hx[m] ** 2).mean(), 1e-9)))
