"""Round 39 step 1 measurement (frozen before generation).

Given a BokehDiff output y, the reference ref (aperture-sampled thin lens, linear light) and the scene:
  offsets     per side, the per-channel median of (y - ref) in that side's interior (fg: > 32 px inside the edge;
              bg: > 32 px outside, excluding a 24 px image border). Global colour/brightness shifts are not boundary
              errors, so each side's offset is removed from its own bands.
  band error  mean |y - ref - offset_side| (0-255, averaged over channels) in each signed-distance band.
  excess E    max over the edge bands (out8-16, out0-8, in0-8, in8-16) of (band error - that side's interior error).
  ratio       the band error at the argmax divided by that side's interior error.
  profile     mean signed luminance residual (y - ref - offset, Rec.709 weights) in 1 px distance bins, -32..+32.
Validity of a request (interior blur must stay right):
  blurred textured fg (B, C)  linear-estimator radius r_hat of the fg interior within max(2, 20 %) of its target
  sharp fg (A)                fg-interior offset-only MAE to the input <= 3.0 (round 25's focus limit)
  sharp bg (B)                bg-interior offset-only MAE to the input <= 3.0
  blurred bg (A, C)           bg-interior error to ref <= 3.0 (a uniform/gradient bg carries no radius information)
"""
import numpy as np

import measure18 as m18
import measure19 as M
import scenes39 as S39

LUM = np.array([0.2126, 0.7152, 0.0722])
FOCUS_MAX = 3.0
E_MIN, RATIO_MIN, FRAC_MIN = 2.0, 1.5, 18 / 24


def off_mae(a, b, m):
    d = a[m].astype(np.float64) - b[m].astype(np.float64)
    return float(np.abs(d - np.median(d, 0)).mean())


def lin_stack_fg(b):
    """linear-light disk stack of the input, sampled at the fg interior: (G, n, 3) for fit_fast."""
    m = b["bands"]["in>32"]
    lin = S39.to_lin(b["img"])
    return np.stack([S39.to_srgb(m18.blur(lin, m18.disk(r)))[m] for r in m18.R_GRID])


def measure(y, ref, b, setting, targets, Bfg):
    y, ref = y.astype(np.float64), ref.astype(np.float64)
    bm = b["bands"]
    res = y - ref
    off = {"in": np.median(res[bm["in>32"]], 0), "out": np.median(res[bm["out>32"]], 0)}
    band = {}
    for n, m in bm.items():
        side = "in" if n.startswith("in") else "out"
        band[n] = float(np.abs(res[m] - off[side]).mean())
    interior = {"in": band["in>32"], "out": band["out>32"]}
    ex = {n: band[n] - interior["in" if n.startswith("in") else "out"] for n in S39.EDGE_BANDS}
    nE = max(ex, key=ex.get)
    sideE = "in" if nE.startswith("in") else "out"
    lumres = ((res - np.where(b["fgmask"][..., None], off["in"], off["out"])) * LUM).sum(-1)
    d = np.round(b["dist"]).astype(int)
    border = bm["out>32"] | bm["out16-32"] | bm["out8-16"] | bm["out0-8"] | bm["in0-8"] | bm["in8-16"] | bm["in16-32"] | bm["in>32"]
    prof = {int(k): float(lumres[(d == k) & border].mean()) for k in range(-32, 33) if ((d == k) & border).any()}
    out = dict(band=band, interior=interior, E=float(ex[nE]), E_band=nE,
               ratio=float(band[nE] / max(interior[sideE], 1e-6)), excess=ex, profile=prof)
    # validity
    img = b["img"].astype(np.float64)
    v = {}
    if setting in ("B", "C"):
        r_hat = float(M.fit_fast(y[bm["in>32"]], Bfg)["v"])
        out["r_hat_fg"] = r_hat
        v["fg_radius"] = abs(r_hat - targets["fg"]) <= m18.TOL(targets["fg"])
    else:
        out["fg_focus"] = off_mae(y, img, bm["in>32"])
        v["fg_focus"] = out["fg_focus"] <= FOCUS_MAX
    if setting == "B":
        out["bg_focus"] = off_mae(y, img, bm["out>32"])
        v["bg_focus"] = out["bg_focus"] <= FOCUS_MAX
    else:
        v["bg_interior"] = interior["out"] <= FOCUS_MAX
    out["valid_parts"], out["valid"] = v, bool(all(v.values()))
    return out


def qualifies(m):
    return m["E"] >= E_MIN and m["ratio"] >= RATIO_MIN
