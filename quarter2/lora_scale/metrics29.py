"""Round 29 shared definitions: scene, cases, target render and error measures (used by run29.py and analyze29.py)."""
import numpy as np

import design as D
import measure18 as m18
import patches as PT

CASES = [(1, "gravel_0"), (2, "astronaut_0"), (3, "chelsea_0"), (5, "coins_0")]   # (cell, patch) in group 0, position 0
TARGETS = [8, 11.625]
LOCS = ["pre_pisa", "post_pisa", "mid", "up_late", "latent", "dec_early"]
WIDTHS = [16, 32]                       # Gaussian sigma in final-image pixels
ALPHAS = [-0.10, -0.03, 0.03, 0.10]
FOCUS_ADD_MAX = 0.5                     # /255, added focused-region MAE vs the source


def scene():
    lib = PT.library()
    grp = [p for p in lib if p["split"] == "test"][0::2]
    img, cid, where = PT.layout(grp, 0)
    assert all(where[k] == p for k, p in CASES), where
    return img, cid, where


def target(img, t):
    return np.clip(m18.blur(img.astype(np.float32), m18.disk(t)), 0, 255)


def aff_mae(y, tgt, m):
    """primary error: mean |y - (a_c * target + b_c)| in mask m, per-channel gain/offset fitted (blur shape, not colour)"""
    a, b = y[m].astype(np.float64), tgt[m].astype(np.float64)
    res = []
    for c in range(3):
        A = np.stack([b[:, c], np.ones(len(b))], 1)
        coef = np.linalg.lstsq(A, a[:, c], rcond=None)[0]
        res.append(np.abs(a[:, c] - A @ coef).mean())
    return float(np.mean(res))


def raw_mae(y, ref, m):
    return float(np.abs(y[m].astype(np.float64) - ref[m].astype(np.float64)).mean())


def focus_mask(cid):
    return D.interior(cid, 4)
