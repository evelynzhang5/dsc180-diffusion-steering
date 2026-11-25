"""Round 45 (CPU): fit and FREEZE the monotone controller on round 44's fp32 DEVELOPMENT curves only.

Data: round 44 score44.json, curves[i]["fp32"] = delivered blur r at K = 4 ... 64 for the 8 round 40 mapping scenes
(all behind tiles, round 40 estimator); unmeasurable points (NaN, beyond the estimator's 24 px cap) are dropped.
x = K * gap, gap = the scene's depth gap (fit42.gap: BokehDiff preprocessing, median over behind tiles), from
rule42.json dev_gaps (inputs only).
Curve (one form, fixed in advance, no selection among curves): isotonic (non-decreasing) least-squares fit of r on x
(pool-adjacent-violators, ties averaged), evaluated at the sorted unique x; between knots, linear interpolation.
Inverse for a target t: the smallest x on the piecewise-linear curve with y(x) >= t (linear within the crossing
segment). Out of range: if t exceeds the curve's maximum, x = the largest knot (flagged); if t is below its minimum,
x = the smallest knot. K = x / gap, clipped to [4, 64] (clipping flagged).
usage (repo root): python quarter2/confirm45/fit45.py   -> quarter2/confirm45/results/rule45.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S44 = os.path.join(HERE, "..", "precision44", "results", "run", "score44.json")
R42 = os.path.join(HERE, "..", "control42", "results", "rule42.json")


def pava(y):
    blocks = [[v, 1] for v in y]
    out = []
    for v, n in blocks:
        out.append([v, n])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, n2 = out.pop()
            v1, n1 = out.pop()
            out.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return np.concatenate([[v] * n for v, n in out])


def fit(x, r):
    o = np.argsort(x, kind="stable")
    x, r = np.asarray(x)[o], np.asarray(r)[o]
    ux = np.unique(x)
    ur = np.array([r[x == u].mean() for u in ux])
    w = np.array([(x == u).sum() for u in ux])
    # weighted PAVA via expansion by counts
    yy = pava(np.repeat(ur, w))
    uy = np.array([yy[np.searchsorted(np.repeat(ux, w), u)] for u in ux])
    return ux, uy


def inverse(kx, ky, t):
    if t > ky.max():
        return float(kx[-1]), "above_curve"
    if t <= ky[0]:
        return float(kx[0]), "below_curve"
    j = int(np.argmax(ky >= t))
    x0, x1, y0, y1 = kx[j - 1], kx[j], ky[j - 1], ky[j]
    return float(x0 + (t - y0) / (y1 - y0) * (x1 - x0)) if y1 > y0 else float(x1), "ok"


def plan_K(rule, t, gap):
    x, flag = inverse(np.array(rule["knots_x"]), np.array(rule["knots_y"]), t)
    k = x / gap
    kc = float(np.clip(k, rule["k_min"], rule["k_max"]))
    return kc, dict(flag=flag, unclipped=float(k), clipped=bool(kc != k))


if __name__ == "__main__":
    s, r42 = json.load(open(S44)), json.load(open(R42))
    gaps = r42["dev_gaps"]
    xs, rs = [], []
    for i, g in gaps.items():
        c = s["curves"][i]["fp32"]
        for K, r in zip(c["K"], c["r"]):
            if np.isfinite(r):
                xs.append(K * g)
                rs.append(r)
    kx, ky = fit(xs, rs)
    rule = dict(form="isotonic", knots_x=kx.tolist(), knots_y=ky.tolist(), k_min=4.0, k_max=64.0, dev_gaps=gaps,
                n_points=len(xs), source="round 44 fp32 dev curves (8 mapping scenes x 12 K)")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(rule, open(os.path.join(HERE, "results", "rule45.json"), "w"), indent=1)
    print(f"{len(xs)} points, {len(kx)} knots; x range {kx[0]:.2f}-{kx[-1]:.2f}; y range {ky[0]:.2f}-{ky[-1]:.2f}")
    for t in [4, 8, 12]:
        x, f = inverse(kx, ky, t)
        b, c = r42["params"]["b"], r42["params"]["c"]
        print(f"target {t}: monotone x = K*gap {x:.2f} ({f}); linear x {(t / b + c):.2f}")
