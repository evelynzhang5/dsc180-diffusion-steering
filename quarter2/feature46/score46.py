"""Round 46 (CPU, on saved outputs): score the controlled transparency test. FROZEN before any round 46 render.

Reference (a controlled compositing reference using BokehDiff-rendered backgrounds, NOT physical ground truth):
  R = alpha * F + (1 - alpha) * lin(REF_bg), REF_bg = BokehDiff(B alone) with the same map d_B, focus, K, resolution,
  seed. Everything in linear light (0-1), errors reported x255.
Methods: A (original map), S (see-through = blur everything, negative control), M2 (surface-aware single map),
  O (pixel output blend alpha * A + (1 - alpha) * S), C_pisa (PRIMARY feature blend), C_latent and C_dec (exploratory).
Regions inside the pane: surface-dominant (alpha >= 0.8), transmission-dominant (alpha <= 0.2), overlap
  (0.2 < alpha < 0.8); < 2,000 px -> unavailable. Errors in these regions are not independent measurements of the two
  underlying layers. Also: whole pane, and outside the pane (sanity: should be equal for all methods).
Per composite: mean absolute error vs R (x255) per region; surface-pattern agreement = correlation between the
  high-pass (x - Gaussian(2 px)) of the output and of R over the pane.
Independent units: the 6 backgrounds. Each background's value = mean over its 4 surface types; paired differences
  are bootstrapped over backgrounds (10,000, seed 0); results are also shown per surface type.
Controls (must pass before quality is interpreted): zero intervention (2-channel disp_coc vs 3-channel) and alpha := 1 /
  alpha := 0 endpoints of every C_* reproduce A / S (max diff <= 1/255); flat colour: the ADDITIONAL brightness change
  of each C_* relative to the alpha-weighted endpoint renders, mean over the pane, |.| <= 1/255.
Decision (frozen):
  ADVANTAGE if C_pisa lowers whole-pane error vs O by >= 10% (relative, background-level mean) with the bootstrap CI of
  the difference excluding 0, and no available region is worse than O by more than 5% (point estimate).
  Otherwise: "No advantage is established for the tested feature blends; output blending remains the practical
  baseline." O's absolute error against R is reported either way. C_latent and C_dec are exploratory (no decision).
usage (repo root): python quarter2/feature46/score46.py <R46 data> <R46 out> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

DATA, OUT, RES = sys.argv[1:4]
os.makedirs(RES, exist_ok=True)
META = json.load(open(f"{DATA}/compose46.json"))
run = json.load(open(f"{OUT}/bd46.json"))
rng = np.random.default_rng(0)
pane = np.load(f"{DATA}/maps/pane.npy") > 0.5
MIN_PX = 2000
METHODS = ["A", "S", "M2", "O", "C_pisa", "C_latent", "C_dec"]


def lin(path):
    x = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def hp(x):
    return x - np.stack([gaussian_filter(x[..., c], 2.0) for c in range(3)], -1)


files = {(r["kind"], r.get("comp", r["bg"])): r["file"] for r in run["rows"]}
rows = []
for c, m in META["composites"].items():
    if m["control"] or ("A", c) not in files:
        continue
    a = np.load(f"{DATA}/alpha/{c}.npy")[..., None].astype(np.float64)
    Fs = np.load(f"{DATA}/surf/{c}.npy").astype(np.float64)
    R = a * Fs + (1 - a) * lin(f"{OUT}/{files[('ref', m['bg'])]}")
    out = {k: lin(f"{OUT}/{files[(k, c)]}") for k in ["A", "S", "M2", "C_pisa", "C_latent", "C_dec"]}
    out["O"] = a * out["A"] + (1 - a) * out["S"]
    a2 = a[..., 0]
    regs = dict(pane=pane, surface=pane & (a2 >= 0.8), transmission=pane & (a2 <= 0.2),
                overlap=pane & (a2 > 0.2) & (a2 < 0.8), outside=~pane)
    hR = hp(R)
    for k, y in out.items():
        e = np.abs(y - R).mean(-1) * 255
        hy = hp(y)
        corr = float(np.corrcoef(hy[pane].ravel(), hR[pane].ravel())[0, 1])
        rows.append(dict(comp=c, bg=m["bg"], type=m["type"], method=k, corr=corr,
                         **{f"err_{n}": (float(e[g].mean()) if g.sum() >= MIN_PX else float("nan"))
                            for n, g in regs.items()}))
    print(c, "scored", flush=True)


def val(method, key, bg=None, typ=None):
    v = [r[key] for r in rows if r["method"] == method and (bg is None or r["bg"] == bg)
         and (typ is None or r["type"] == typ)]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


bgs = sorted({r["bg"] for r in rows})
types = sorted({r["type"] for r in rows})
KEYS = ["err_pane", "err_surface", "err_transmission", "err_overlap", "err_outside", "corr"]


def boot(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, est=float("nan"), lo=float("nan"), hi=float("nan"))
    b = [np.mean(rng.choice(v, len(v))) for _ in range(10000)]
    return dict(n=int(len(v)), est=float(np.mean(v)), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)))


summary = {m: {k: boot([val(m, k, bg=b) for b in bgs]) for k in KEYS} for m in METHODS}
by_type = {t: {m: {k: val(m, k, typ=t) for k in KEYS} for m in METHODS} for t in types}
paired = {}
for m in ["C_pisa", "C_latent", "C_dec", "M2", "A", "S"]:
    paired[f"{m}-O"] = {k: boot([val(m, k, bg=b) - val("O", k, bg=b) for b in bgs]) for k in KEYS}
    paired[f"{m}-O"]["n_backgrounds_better_pane"] = int(sum(val(m, "err_pane", bg=b) < val("O", "err_pane", bg=b)
                                                            for b in bgs))

# controls
chk = run["checks"]
ctrl = dict(endpoints_and_zero=dict(chk), endpoints_ok=bool(chk) and max(chk.values()) <= 1)
if ("A", "flat__tinted") in files:
    a = np.load(f"{DATA}/alpha/flat__tinted.npy")[..., None]
    A, S = lin(f"{OUT}/{files[('A', 'flat__tinted')]}"), lin(f"{OUT}/{files[('S', 'flat__tinted')]}")
    expect = (a * A + (1 - a) * S)[pane].mean()
    ctrl["flat_extra_brightness_x255"] = {loc: float((lin(f"{OUT}/{files[(loc, 'flat__tinted')]}")[pane].mean()
                                                       - expect) * 255) for loc in ["C_pisa", "C_latent", "C_dec"]}
    ctrl["flat_ok"] = {k: abs(v) <= 1 for k, v in ctrl["flat_extra_brightness_x255"].items()}
p = paired["C_pisa-O"]
rel = p["err_pane"]["est"] / summary["O"]["err_pane"]["est"]
worse = {k: paired["C_pisa-O"][k]["est"] / summary["O"][k]["est"]
         for k in ["err_surface", "err_transmission", "err_overlap"] if np.isfinite(summary["O"][k]["est"])}
adv = rel <= -0.10 and p["err_pane"]["hi"] < 0 and all(v <= 0.05 for v in worse.values())
interpretable = ctrl["endpoints_ok"] and ctrl.get("flat_ok", {}).get("C_pisa", False)
decision = dict(
    verdict=("ADVANTAGE" if adv else "No advantage is established for the tested feature blends; output blending "
             "remains the practical baseline.") if interpretable else "NOT INTERPRETABLE: a control failed",
    C_pisa_relative_pane_change=float(rel), region_relative_change=worse,
    output_blend_abs_err_pane_x255=summary["O"]["err_pane"]["est"], controls=ctrl, pisa_calls=run["calls"])
json.dump(dict(decision=decision, summary=summary, by_type=by_type, paired=paired, rows=rows),
          open(f"{RES}/score46.json", "w"), indent=1, default=float)
print(f"\n{len(bgs)} backgrounds x {len(types)} surface types")
for m in METHODS:
    s = summary[m]
    print(f"{m:9s} " + " ".join(f"{k[4:] if k.startswith('err') else k} {s[k]['est']:.2f}" for k in KEYS))
for t in types:
    print(t, {m: round(by_type[t][m]["err_pane"], 2) for m in METHODS})
for k, v in paired.items():
    print(k, f"pane {v['err_pane']['est']:+.2f} [{v['err_pane']['lo']:+.2f},{v['err_pane']['hi']:+.2f}]",
          f"better in {v['n_backgrounds_better_pane']}/{len(bgs)}")
print("decision:", json.dumps(decision, indent=1))
