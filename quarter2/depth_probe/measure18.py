"""Round 18 measurement: blur-radius estimator, its validation on known lens-blur renders, and the frozen analysis.

Estimator: inside a region's interior (>= 48 px from any depth edge, >= 24 px from the image border), find the disk
radius r (anti-aliased disk, grid 0..48 px step 0.5, parabolic refinement) such that the sharp input blurred by
the disk best matches the output, allowing a per-channel gain and offset (least squares). A Gaussian fit (sigma) is
reported alongside. Shift: integer search over +-3 px of the best-blurred input, parabolic refinement.
"""
import json
import os

import numpy as np
from scipy.signal import fftconvolve

R_GRID = np.arange(0, 48.01, 0.5)
S_GRID = np.arange(0, 24.01, 0.25)
R_CAP = 48.0
NOM = 31 * 16 / 51.2 / 10          # = 0.96875: nominal radius in px (512 image) = NOM * K * |d_norm - d_focus|
VAL_TOL = lambda r: max(1.0, 0.10 * r)   # estimator validation tolerance (disk renders)
TOL = lambda r: max(2.0, 0.20 * r)       # agreement tolerance for BokehDiff vs predicted radius
PASS_FRAC = 0.80
MAE_FOCUS, SHIFT_MAX, FOCUS_R_SLACK = 3.0, 1.0, 1.0


def disk(r, ss=8):
    if r <= 1e-6:
        return np.ones((1, 1), np.float32)
    n = int(np.ceil(r))
    t = (np.arange(-n * ss, (n + 1) * ss) + 0.5) / ss - 0.5          # sub-sample centres
    yy, xx = np.meshgrid(t, t, indexing="ij")
    m = (yy ** 2 + xx ** 2 <= r * r).astype(np.float32)
    k = m.reshape(2 * n + 1, ss, 2 * n + 1, ss).mean((1, 3))
    return k / k.sum()


def gauss(s):
    if s <= 1e-6:
        return np.ones((1, 1), np.float32)
    n = int(np.ceil(3.5 * s))
    t = np.arange(-n, n + 1)
    g = np.exp(-t ** 2 / (2 * s * s))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


def blur(x, k):
    """x HxWxC float, reflect padding."""
    p = k.shape[0] // 2
    if p == 0:
        return x.copy()
    xp = np.pad(x, ((p, p), (p, p), (0, 0)), mode="reflect")
    return np.stack([fftconvolve(xp[..., c], k, mode="valid") for c in range(x.shape[2])], -1)


def stack(x, kind="disk"):
    grid = R_GRID if kind == "disk" else S_GRID
    f = disk if kind == "disk" else gauss
    return grid, [blur(x, f(v)) for v in grid]


def _affine_mse(y, b, m):
    """per-channel least-squares y ~ a*b + c on mask m; returns mse (0-255 scale) and fitted image."""
    err = 0.0
    for c in range(y.shape[2]):
        yy, bb = y[..., c][m], b[..., c][m]
        A = np.stack([bb, np.ones_like(bb)], 1)
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        err += np.mean((A @ coef - yy) ** 2)
    return err / y.shape[2]


def _parabola(grid, e, i):
    if 0 < i < len(grid) - 1:
        d = e[i - 1] - 2 * e[i] + e[i + 1]
        if d > 0:
            off = 0.5 * (e[i - 1] - e[i + 1]) / d
            return float(grid[i] + np.clip(off, -1, 1) * (grid[1] - grid[0]))
    return float(grid[i])


def fit(y, st, m):
    grid, bl = st
    e = np.array([_affine_mse(y, b, m) for b in bl])
    i = int(np.argmin(e))
    return dict(v=_parabola(grid, e, i), mse=float(e[i]), mse0=float(e[0]), at_cap=bool(i == len(grid) - 1), i=i)


def shift(y, b, m, s=3):
    # interiors are >= 24 px from the border, so np.roll's wrap-around never reaches them for |shift| <= 3
    Y, L = y.mean(2), b.mean(2)
    E = np.zeros((2 * s + 1, 2 * s + 1))
    for i, dy in enumerate(range(-s, s + 1)):
        for j, dx in enumerate(range(-s, s + 1)):
            B = np.roll(L, (dy, dx), (0, 1))
            yy, bb = Y[m], B[m]
            A = np.stack([bb, np.ones_like(bb)], 1)
            coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
            E[i, j] = np.mean((A @ coef - yy) ** 2)
    i, j = np.unravel_index(np.argmin(E), E.shape)
    g = np.arange(-s, s + 1, dtype=float)
    dy = _parabola(g, E[:, j], i)
    dx = _parabola(g, E[i, :], j)
    return float(dy), float(dx)


# ---------------------------------------------------------------- estimator validation (known lens-blur renders)
def render_layers(img, labels, radii, kind="disk"):
    out = np.zeros_like(img)
    f = disk if kind == "disk" else gauss
    for lab, r in radii.items():
        out[labels == lab] = blur(img, f(r))[labels == lab]
    return out


def degrade(x, rng):
    """the kinds of change an editor's output also has: per-channel gain/offset, noise, 8-bit quantization"""
    g = np.array([0.96, 1.03, 0.98], np.float32)
    o = np.array([2.0, -3.0, 1.0], np.float32)
    return np.clip(np.round(x * g + o + rng.normal(0, 1.5, x.shape)), 0, 255)


def validate(scenes_mod):
    rng = np.random.default_rng(0)
    rads = [0, 1, 2, 4, 8, 12, 16, 24, 32, 40]
    sig = [0.5, 1, 2, 4, 8, 12]
    rows, grows = [], []
    for n in scenes_mod.LAYOUTS:
        s = scenes_mod.make(n)
        img = s["image"].astype(np.float32)
        labs = [r["label"] for r in s["regions"]]
        ints = {l: scenes_mod.interior(s["labels"], l) for l in labs}
        st = stack(img, "disk")
        for rep in range(3):                                   # rotate radii over regions so each sees several
            assign = {l: rads[(3 * rep + 2 * k + len(n)) % len(rads)] for k, l in enumerate(labs)}
            y = degrade(render_layers(img, s["labels"], assign), rng)
            for l in labs:
                f = fit(y, st, ints[l])
                rows.append(dict(scene=n, label=l, r_true=assign[l], r_hat=f["v"], err=f["v"] - assign[l],
                                 ok=abs(f["v"] - assign[l]) <= VAL_TOL(assign[l])))
            sh = shift(y, st[1][int(round(assign[labs[0]] / 0.5))], ints[labs[0]])
            rows[-1]["shift_check"] = sh
        # Gaussian renders: report what the disk fit returns (kernel-shape bias), no pass criterion
        assign = {l: sig[k % len(sig)] for k, l in enumerate(labs)}
        y = degrade(render_layers(img, s["labels"], assign, "gauss"), rng)
        for l in labs:
            f = fit(y, st, ints[l])
            grows.append(dict(scene=n, label=l, sigma=assign[l], disk_r_hat=f["v"]))
    ok = all(r["ok"] for r in rows)
    # known-shift check: shift a disk render by (1, -2) px, expect that back
    s = scenes_mod.make("A")
    img = s["image"].astype(np.float32)
    b = blur(img, disk(8))
    y = np.roll(degrade(b, rng), (1, -2), (0, 1))
    sh = shift(y, b, scenes_mod.interior(s["labels"], 0))
    sh_ok = abs(sh[0] - 1) <= 0.25 and abs(sh[1] + 2) <= 0.25
    return dict(disk=rows, gauss=grows, disk_ok=ok, worst_err=float(max(abs(r["err"]) for r in rows)),
                shift_known=(1, -2), shift_hat=sh, shift_ok=sh_ok, passed=bool(ok and sh_ok))


# ---------------------------------------------------------------- analysis of BokehDiff outputs
def dnorm(disp):
    return (disp - disp.min()) / (disp.max() - disp.min())


def analyze(out_dir, scenes_mod, conds):
    """conds: list of dict(scene, K, focus, seed, file). Returns per-region rows."""
    from PIL import Image
    rows = []
    for n in scenes_mod.LAYOUTS:
        s = scenes_mod.make(n)
        img = s["image"].astype(np.float32)
        dn = dnorm(s["disparity"])
        st, stg = stack(img, "disk"), stack(img, "gauss")
        ref0 = [c for c in conds if c["scene"] == n and c["K"] == 0 and c["seed"] == 0 and not c.get("repeat")][0]
        y0 = np.asarray(Image.open(os.path.join(out_dir, ref0["file"]))).astype(np.float32)
        st0 = stack(y0, "disk")
        for c in [c for c in conds if c["scene"] == n and not c.get("repeat")]:
            y = np.asarray(Image.open(os.path.join(out_dir, c["file"]))).astype(np.float32)
            zf = s["focus"][c["focus"]] if c["focus"] else None
            df = float(dn[s["depth"] == zf][0]) if zf else None
            for reg in s["regions"]:
                m = scenes_mod.interior(s["labels"], reg["label"])
                dl = float(dn[s["labels"] == reg["label"]][0])
                r_nom = NOM * c["K"] * abs(dl - df) if df is not None else 0.0
                f = fit(y, st, m)
                g = fit(y, stg, m)
                f0 = fit(y, st0, m)
                sh = shift(y, st[1][f["i"]], m)
                rows.append(dict(scene=n, K=c["K"], focus=c["focus"], seed=c["seed"], label=reg["label"],
                                 texture=reg["texture"], z=reg["z"], d_norm=dl, d_focus=df,
                                 in_focus=bool(zf is not None and reg["z"] == zf),
                                 side=("none" if zf is None else "focus" if reg["z"] == zf else
                                       "front" if zf is not None and reg["z"] < zf else "behind"),
                                 r_nom=r_nom, r_hat=f["v"], at_cap=f["at_cap"], fit_mse=f["mse"], mse_r0=f["mse0"],
                                 sigma_hat=g["v"], gauss_mse=g["mse"], r_hat_vs_k0=f0["v"],
                                 shift_dy=sh[0], shift_dx=sh[1],
                                 mae_vs_input=float(np.abs(y - img)[m].mean()),
                                 mae_vs_k0=float(np.abs(y - y0)[m].mean())))
    return rows


def decide(rows, dev="A"):
    """Frozen decision rules (see README)."""
    meas = [r for r in rows if r["K"] > 0 and r["seed"] == 0]
    blurred = [r for r in meas if not r["in_focus"]]
    k0 = {(r["scene"], r["label"]): r for r in rows if r["K"] == 0}
    within = lambda r, c: abs(r["r_hat"] - c * r["r_nom"]) <= TOL(c * r["r_nom"])
    frac = lambda rs, c: float(np.mean([within(r, c) for r in rs])) if rs else float("nan")
    nominal_frac = frac(blurred, 1.0)
    d = [r for r in blurred if r["scene"] == dev]
    c_hat = float(sum(r["r_hat"] * r["r_nom"] for r in d) / sum(r["r_nom"] ** 2 for r in d))
    held = [r for r in blurred if r["scene"] != dev]
    calib_frac = frac(held, c_hat)
    focus_rows = [r for r in meas if r["in_focus"]]
    focus_ok = [r["r_hat"] <= k0[(r["scene"], r["label"])]["r_hat"] + FOCUS_R_SLACK and r["mae_vs_k0"] <= MAE_FOCUS
                for r in focus_rows]
    shift_ok = [max(abs(r["shift_dy"]), abs(r["shift_dx"])) <= SHIFT_MAX for r in meas]
    if nominal_frac >= PASS_FRAC:
        outcome = "1: behaves as specified (keep BokehDiff as a baseline; no demonstrated problem here)"
    elif calib_frac >= PASS_FRAC:
        outcome = "2: consistently too strong/weak; one global gain fitted on scene A fixes held-out scenes"
    else:
        outcome = "3: errors not fixed by one global gain (candidate problem; confirm on more scenes)"
    return dict(n_blurred=len(blurred), nominal_within_frac=nominal_frac, c_hat_dev=c_hat,
                calibrated_within_frac_heldout=calib_frac, n_heldout=len(held),
                any_at_cap=any(r["at_cap"] for r in meas),
                focus_preserved_frac=float(np.mean(focus_ok)), n_focus=len(focus_rows),
                shift_ok_frac=float(np.mean(shift_ok)), outcome=outcome)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scenes
    v = validate(scenes)
    print(json.dumps({k: v[k] for k in ("disk_ok", "worst_err", "shift_hat", "shift_ok", "passed")}))
    for r in v["gauss"]:
        print("gauss", r)
