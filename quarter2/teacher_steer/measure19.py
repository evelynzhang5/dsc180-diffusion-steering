"""Round 19 measurement and calibration.

Radius estimator = round 18's (disk radius 0..48 px step 0.5, per-channel gain + offset, parabolic refinement), computed
in closed form (least squares with intercept: residual = var(y) - cov(y,b)^2 / var(b)), which is algebraically the
same fit. A fit at the grid end is "above range" (true radius >= ~48 px; 48 is not a measurement).

Calibrations, fitted on the development scenes only, then frozen:
  C0  one global gain (round 18's baseline)
  C1  monotone curve: measured radius as a function of requested radius (isotonic regression, pooled)
  C2  monotone curves per side of focus (front / behind)
A target radius t is reached by requesting g^-1(t): the smallest request whose curve value reaches t (linear
interpolation); the corrected request replaces that cell's blur map value and the editor is rerun.
"""
import numpy as np

import measure18 as m18

R_GRID = m18.R_GRID
TOL = m18.TOL


def stack_at(st, mask):
    """blurred-input values at mask pixels for every grid radius: (G, n, 3)"""
    return np.stack([b[mask] for b in st[1]])


def fit_fast(y_m, B):
    """y_m (n,3) output at mask; B (G,n,3). Returns dict(v, mse, at_cap, i)."""
    yc = y_m - y_m.mean(0)
    bc = B - B.mean(1, keepdims=True)
    cov = (bc * yc[None]).mean(1)
    vb = (bc ** 2).mean(1)
    vy = (yc ** 2).mean(0)
    e = (vy[None] - np.where(vb > 1e-9, cov ** 2 / np.maximum(vb, 1e-9), 0)).mean(1)
    i = int(np.argmin(e))
    return dict(v=m18._parabola(R_GRID, e, i), mse=float(e[i]), at_cap=bool(i == len(R_GRID) - 1), i=i)


# ---------------------------------------------------------------- isotonic calibration curves
def pav(x, y):
    """isotonic (non-decreasing) regression of y on x; returns unique sorted x and fitted values."""
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    ux = np.unique(x)
    yv = np.array([y[x == u].mean() for u in ux])
    w = np.array([(x == u).sum() for u in ux], float)
    blocks = [[yv[i], w[i], 1] for i in range(len(ux))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, n2 = out.pop()
            v1, w1, n1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    fit = np.concatenate([[v] * n for v, _, n in out])
    return ux, fit


class Curve:
    def __init__(self, x, y, floor):
        ux, fy = pav(np.r_[0.0, x], np.r_[floor, y])          # anchored at request 0 = the K=0 floor
        self.x, self.y = ux, fy

    def __call__(self, r):
        return np.interp(r, self.x, self.y)

    def inverse(self, t):
        """smallest request reaching t; below the floor -> 0; above the curve's top -> its largest request (flagged)"""
        if t <= self.y[0]:
            return 0.0, "below_floor"
        if t > self.y[-1]:
            return float(self.x[-1]), "unreachable"
        j = int(np.argmax(self.y >= t))                       # first index reaching t (j >= 1)
        x0, x1, y0, y1 = self.x[j - 1], self.x[j], self.y[j - 1], self.y[j]
        return float(x0 + (t - y0) / (y1 - y0) * (x1 - x0)) if y1 > y0 else float(x1), "ok"

    def as_dict(self):
        return dict(x=self.x.tolist(), y=self.y.tolist())


def fit_calibrations(dev_rows, floor):
    """dev_rows: non-focus cells of dev scenes (uncorrected, seed 0), with r_nom, r_hat, at_cap, side.
    Above-range fits enter at 48 px (a lower bound)."""
    x = np.array([r["r_nom"] for r in dev_rows])
    y = np.array([48.0 if r["at_cap"] else r["r_hat"] for r in dev_rows])
    unc = np.array([not r["at_cap"] for r in dev_rows])
    c0 = float((x[unc] * y[unc]).sum() / (x[unc] ** 2).sum())
    c1 = Curve(x, y, floor)
    c2 = {s: Curve(x[[r["side"] == s for r in dev_rows]], y[[r["side"] == s for r in dev_rows]], floor)
          for s in ("front", "behind")}
    return c0, c1, c2


def corrected_requests(cells_, method, c0, c1, c2):
    """returns {cell: (requested radius, status)} for non-focus cells, target = their nominal radius"""
    out = {}
    for c in cells_:
        if c["side"] == "focus":
            continue
        t = c["r_nom"]
        if method == "C0":
            out[c["cell"]] = (t / c0, "ok")
        elif method == "C1":
            out[c["cell"]] = c1.inverse(t)
        else:
            out[c["cell"]] = c2[c["side"]].inverse(t)
    return out


def validate(design):
    """estimator check on the round 19 layouts: known per-cell disk radii, degraded like an editor output."""
    rng = np.random.default_rng(0)
    rads = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 40]
    rows = []
    for n in design.SCENES:
        s = design.make(n)
        img = s["image"].astype(np.float32)
        st = m18.stack(img, "disk")
        ints = {k: design.interior(s["cid"], k) for k in range(9)}
        Bs = {k: stack_at(st, ints[k]) for k in range(9)}
        for rep in range(2):
            assign = {k: rads[(5 * rep + 3 * k + int(n[1])) % len(rads)] for k in range(9)}
            y = m18.degrade(m18.render_layers(img, s["cid"], assign), rng)
            for k in range(9):
                f = fit_fast(y[ints[k]], Bs[k])
                rows.append(dict(scene=n, cell=k, r_true=assign[k], r_hat=f["v"],
                                 ok=abs(f["v"] - assign[k]) <= m18.VAL_TOL(assign[k])))
    # one direct check that the closed form equals round 18's least-squares fit
    s = design.make("S1")
    img = s["image"].astype(np.float32)
    st = m18.stack(img, "disk")
    y = m18.degrade(m18.render_layers(img, s["cid"], {k: 7 for k in range(9)}), rng)
    m = design.interior(s["cid"], 4)
    same = abs(fit_fast(y[m], stack_at(st, m))["v"] - m18.fit(y, st, m)["v"]) < 1e-3   # float32 vs float64 rounding
    return dict(rows=rows, passed=bool(all(r["ok"] for r in rows) and same), closed_form_matches=bool(same),
                worst_err=float(max(abs(r["r_hat"] - r["r_true"]) for r in rows)))
