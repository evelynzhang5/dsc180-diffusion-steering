"""Round 40 stage 3: effective ADDITIONAL blur on preselected tiles (linear light).

A generated input has unknown existing blur, so this measures only the blur ADDED by a method: per tile, the disk
radius r (anti-aliased disk, 0-24 px step 0.5, parabolic refinement) such that the INPUT blurred by the disk best
matches the OUTPUT, allowing a per-channel gain and offset (round 18/19 estimator, closed form), in linear light.
The input is blurred with its real surroundings (tile plus a 26 px margin, reflect padding only at the image border).
Fit quality R^2 = 1 - residual / output variance. A tile counts if R^2 >= R2_MIN and the fit is not at the grid end;
an output's delivered blur is the median over counting tiles, unmeasurable (NaN) with fewer than MIN_TILES.
Total blur (existing + added) is NOT estimated; the existing_blur category is reported separately.
"""
import numpy as np

import common40 as C
import measure18 as m18

GRID = np.arange(0, 24.01, 0.5)
MARGIN, T = 26, 64


class TileStack:
    """Blurred-input stacks for one input image's tiles (computed once, reused for every output)."""

    def __init__(self, x, tiles):
        self.tiles = tiles
        xl = C.to_lin(x)
        H, W = x.shape[:2]
        pad = np.pad(xl, ((MARGIN, MARGIN), (MARGIN, MARGIN), (0, 0)), mode="reflect")
        self.B = []
        for t in tiles:
            y0, x0 = t["y"], t["x"]
            crop = pad[y0:y0 + T + 2 * MARGIN, x0:x0 + T + 2 * MARGIN]
            st = []
            for r in GRID:
                b = m18.blur(crop, m18.disk(r)) if r > 0 else crop
                st.append(b[MARGIN:MARGIN + T, MARGIN:MARGIN + T].reshape(-1, 3))
            self.B.append(np.stack(st))                         # (G, n, 3)

    def measure(self, y):
        yl = C.to_lin(y)
        per = []
        for t, B in zip(self.tiles, self.B):
            ym = yl[t["y"]:t["y"] + T, t["x"]:t["x"] + T].reshape(-1, 3)
            yc = ym - ym.mean(0)
            bc = B - B.mean(1, keepdims=True)
            cov = (bc * yc[None]).mean(1)
            vb = (bc ** 2).mean(1)
            vy = (yc ** 2).mean(0)
            e = (vy[None] - np.where(vb > 1e-12, cov ** 2 / np.maximum(vb, 1e-12), 0)).mean(1)
            i = int(np.argmin(e))
            r = m18._parabola(GRID, e, i)
            r2 = float(1 - e[i] / max(vy.mean(), 1e-12))
            per.append(dict(y=t["y"], x=t["x"], r=float(r), r2=r2, at_cap=bool(i == len(GRID) - 1)))
        good = [p["r"] for p in per if p["r2"] >= C.R2_MIN and not p["at_cap"]]
        r = float(np.median(good)) if len(good) >= C.MIN_TILES else float("nan")
        return dict(r=r, n_good=len(good), n_tiles=len(per), tiles=per)
