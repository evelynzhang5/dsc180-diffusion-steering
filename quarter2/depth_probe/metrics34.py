"""Round 34 revised error (fixes round 29's gain/offset score, which rewards contrast loss).

off_mae   primary: mean |y - (t + b_c)| in the mask, b_c = per-channel median offset. Global brightness/colour
          offsets are ignored; contrast loss, detail removal, shifts and kernel changes are all penalised.
shape_mae reported: gain/offset fit, residual expressed in target units (|y - b - a t| / a); contrast-invariant
          (neither rewards nor penalises contrast scaling).
contrast  reported: per-channel std ratio output / target in the mask (detail/contrast preservation).
"""
import numpy as np

from metrics29 import aff_mae, raw_mae, scene, target, focus_mask, CASES, TARGETS, FOCUS_ADD_MAX  # noqa: F401


def off_mae(y, tgt, m):
    a, b = y[m].astype(np.float64), tgt[m].astype(np.float64)
    d = a - b
    return float(np.abs(d - np.median(d, 0)).mean())


def shape_mae(y, tgt, m):
    a, b = y[m].astype(np.float64), tgt[m].astype(np.float64)
    res = []
    for c in range(3):
        A = np.stack([b[:, c], np.ones(len(b))], 1)
        g, o = np.linalg.lstsq(A, a[:, c], rcond=None)[0]
        res.append(np.abs(a[:, c] - g * b[:, c] - o).mean() / max(abs(g), 1e-3))
    return float(np.mean(res))


def contrast(y, tgt, m):
    return float(np.mean(y[m].astype(np.float64).std(0) / (tgt[m].astype(np.float64).std(0) + 1e-6)))
