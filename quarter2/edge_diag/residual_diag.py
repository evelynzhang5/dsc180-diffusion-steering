"""Round 38 D2 (CPU): what is the calibrated request's remaining error made of? (complementary diagnostics, not an
additive decomposition)

Data: round 36's unedited calibrated-request outputs (a = 0) for the 12 test scenes x 6 free cells = 72 cells
(also the 72 development cells, reported separately). Per cell interior (round 36's mask):
  radius     r* requested; r_fit = round 36's affine stack fit; r_best = the disk radius (0.5 px grid) minimising the
             revised error (offset-only MAE) directly
  error      E(r*) = revised error vs the requested-radius disk (round 36's score); E(r_best); share of E(r*) removed by
             moving to r_best (how much of the error is radius, within or beyond tolerance)
  contrast   std ratio output / disk at r* and at r_best
  gain       per-channel least-squares gain of output on disk(r_best) (contrast-like scaling of the blurred signal)
  shape      metrics34.shape_mae at r_best (gain/offset-invariant residual, in target units)
  colour     per-channel median offset (removed by the score) and its spread across channels (colour cast)
  pattern    residual e = y - disk(r_best) - offset:
               low/high split: share of e's energy below a Gaussian low-pass (sigma 4 px)
               detail retained: regression slope of e on (source - disk(r_best)) (> 0: output keeps source detail that
               the disk removes, i.e. texture under-smoothed; < 0: over-smoothed / contrast lost)
usage: python residual_diag.py <R36_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import metrics34 as Q  # noqa: E402
import patches as PT  # noqa: E402
import scenes36 as SC  # noqa: E402

R36, RES = sys.argv[1], sys.argv[2]
os.makedirs(RES, exist_ok=True)
meta = json.load(open(f"{R36}/conds.json"))
rows = {r["scene"]: r for r in meta["rows"] if r["kind"] == "grid" and r["a"] == 0.0}
out = []
for split in ("test", "dev"):
    for sc in SC.plan(split):
        y = np.asarray(Image.open(f"{R36}/img/{sc['name']}_a0.png")).astype(np.float32)
        src = sc["img"].astype(np.float32)
        grid, st = m18.stack(src, "disk")
        for k in PT.FREE:
            m = D.interior(sc["cid"], k)
            rs = SC.rstar(sc["gaps"][k])
            ym = y[m].astype(np.float64)
            errs = np.array([Q.off_mae(y, b, m) for b in st])
            i = int(np.argmin(errs))
            rb = float(grid[i])
            tb = st[i]
            ts = Q.target(sc["img"], rs)
            e_star, e_best = Q.off_mae(y, ts, m), float(errs[i])
            tbm = tb[m].astype(np.float64)
            d = ym - tbm
            off = np.median(d, 0)
            gain = [float(np.linalg.lstsq(np.stack([tbm[:, c], np.ones(len(tbm))], 1), ym[:, c], rcond=None)[0][0]) for c in range(3)]
            e = np.zeros_like(y)
            e[m] = (d - off).astype(np.float32)
            el = np.stack([gaussian_filter(e[..., c] * m, 4) / np.maximum(gaussian_filter(m.astype(np.float32), 4), 1e-6)
                           for c in range(3)], -1)
            low = float((el[m] ** 2).sum() / max((e[m] ** 2).sum(), 1e-9))
            det = (src[m] - tbm).astype(np.float64).ravel()
            ev = e[m].astype(np.float64).ravel()
            slope = float((det - det.mean()) @ (ev - ev.mean()) / max(((det - det.mean()) ** 2).sum(), 1e-9))
            out.append(dict(split=split, scene=sc["name"], cell=k, src=sc["where"][k].rsplit("_", 1)[0], r_star=rs,
                            r_fit=rows[sc["name"]]["cells"][str(k)]["r_hat"], r_best=rb,
                            within_tol=abs(rows[sc["name"]]["cells"][str(k)]["r_hat"] - rs) <= m18.TOL(rs),
                            E_star=e_star, E_best=e_best, radius_share=(e_star - e_best) / max(e_star, 1e-9),
                            contrast_star=Q.contrast(y, ts, m), contrast_best=Q.contrast(y, tb, m), gain=gain,
                            shape_best=Q.shape_mae(y, tb, m), offset=[float(v) for v in off],
                            colour_cast=float(np.ptp(off)), low_share=low, detail_slope=slope))
        print(sc["name"], "done", flush=True)

json.dump(out, open(f"{RES}/residual_cells.json", "w"), indent=1)


def summ(L):
    g = lambda k: np.array([r[k] for r in L], float)
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]
    return dict(n=len(L), within_tol=int(sum(r["within_tol"] for r in L)),
                r_best_minus_r_star=q(g("r_best") - g("r_star")), E_star=q(g("E_star")), E_best=q(g("E_best")),
                radius_share=q(g("radius_share")), contrast_star=q(g("contrast_star")), contrast_best=q(g("contrast_best")),
                gain_mean=q(np.array([np.mean(r["gain"]) for r in L])), shape_best=q(g("shape_best")),
                offset_mean=q(np.array([np.mean(r["offset"]) for r in L])), colour_cast=q(g("colour_cast")),
                low_share=q(g("low_share")), detail_slope=q(g("detail_slope")),
                corr_E_best_detail_slope=float(np.corrcoef(g("E_best"), np.abs(g("detail_slope")))[0, 1]),
                corr_E_best_low_share=float(np.corrcoef(g("E_best"), g("low_share"))[0, 1]))


S = {}
for split in ("test", "dev"):
    L = [r for r in out if r["split"] == split]
    S[split] = dict(all=summ(L), within_tol=summ([r for r in L if r["within_tol"]]),
                    by_source={s: dict(E_star=round(float(np.median([r["E_star"] for r in L if r["src"] == s])), 2),
                                       E_best=round(float(np.median([r["E_best"] for r in L if r["src"] == s])), 2),
                                       detail_slope=round(float(np.median([r["detail_slope"] for r in L if r["src"] == s])), 3),
                                       contrast_best=round(float(np.median([r["contrast_best"] for r in L if r["src"] == s])), 3),
                                       low_share=round(float(np.median([r["low_share"] for r in L if r["src"] == s])), 3))
                               for s in sorted({r["src"] for r in L})})
json.dump(S, open(f"{RES}/residual_summary.json", "w"), indent=1)
for split in S:
    print("==", split)
    for k, v in S[split]["within_tol"].items():
        print("  ", k, v)
    for s, v in S[split]["by_source"].items():
        print("   ", s, v)
