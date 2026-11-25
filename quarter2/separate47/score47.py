"""Round 47 (CPU, on saved outputs): score the separation pipeline. FROZEN before training.

Reference (as round 46; a controlled compositing reference using BokehDiff-rendered backgrounds, not physical ground
truth): R = alpha F + (1 - alpha) lin(REF_bg), REF_bg = round 46's render of the background alone.
Methods. T1: A, S, M2, O from round 46's renders (unchanged); SEP_full = alpha_hat F_hat + (1 - alpha_hat) lin(render of
B_hat); SEP_oracle = alpha F_hat + (1 - alpha) lin(render of B_hat). T2: A, S, O (round 47 renders) and SEP_*.
Measures (round 46 definitions): linear-light MAE x255 vs R over the pane, surface (alpha >= 0.8), transmission
(alpha <= 0.2) and overlap regions (< 2,000 px -> unavailable), outside the pane; high-pass correlation. Separation
quality: |alpha_hat - alpha| (pane), |F_hat - F| (pane, alpha > 0.2), |B_hat - B| (pane), x255 where colour.
Units: the 6 backgrounds (mean over surface types), paired bootstrap over backgrounds (10,000, seed 0); per type shown.
Controls: pipeline check (true B renders equal round 46's reference, <= 1/255); no-glass control C1 (separator on
background-only images): mean alpha_hat in the pane <= 0.05 (otherwise flagged), |B_hat - I| reported.
Decision on T1 (frozen): SUCCESS if SEP_full lowers the whole-pane error vs O by >= 25% (relative, background-level
means) with the paired-difference CI excluding 0 and no available region worse than O by > 5%; PARTIAL if only
SEP_oracle meets this; otherwise "No advantage is established for the tested separator; output blending remains the
practical baseline." T2 and T3 (real rb_48: round 43's see-through ratio rho on its frozen tiles, next to the real
f/2.0 capture and round 43's conditions) are reported, not decided.
usage (repo root): python score47.py <R46 data> <R46 run> <R47 T2> <R47 sep> <R47 out> <R43 data> <R43 run> <res dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

D46, R46, T2, SEP, OUT, D43, R43, RES = sys.argv[1:9]
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(0)
m46, m47 = json.load(open(f"{D46}/compose46.json")), json.load(open(f"{T2}/compose47.json"))
run46, run47 = json.load(open(f"{R46}/bd46.json")), json.load(open(f"{OUT}/bd47.json"))
f46 = {(r["kind"], r.get("comp", r["bg"])): r["file"] for r in run46["rows"]}
sep = {(r["set"], r["name"], r["variant"]): r["file"] for r in run47["rows"] if r["kind"] == "sep"}
base47 = {(r["kind"], r["name"]): r["file"] for r in run47["rows"] if r["kind"] in ("A", "S")}
pane = np.load(f"{D46}/maps/pane.npy") > 0.5
MIN_PX = 2000


def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin(path):
    return srgb2lin(np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0)


def npy(tag, k):
    return np.load(f"{SEP}/{tag}_{k}.npy").astype(np.float64)


def hp(x):
    return x - np.stack([gaussian_filter(x[..., c], 2.0) for c in range(3)], -1)


def measures(y, R, a2):
    e = np.abs(y - R).mean(-1) * 255
    regs = dict(pane=pane, surface=pane & (a2 >= 0.8), transmission=pane & (a2 <= 0.2),
                overlap=pane & (a2 > 0.2) & (a2 < 0.8), outside=~pane)
    out = {f"err_{n}": (float(e[g].mean()) if g.sum() >= MIN_PX else float("nan")) for n, g in regs.items()}
    out["corr"] = float(np.corrcoef(hp(y)[pane].ravel(), hp(R)[pane].ravel())[0, 1])
    return out


rows = []
for setn, D, comps in [("T1", D46, {c: m for c, m in m46["composites"].items() if not m["control"]}),
                       ("T2", T2, m47["composites"])]:
    for c, m in comps.items():
        if ("T1" if setn == "T1" else "T2", c, "full") not in sep:
            continue
        a = np.load(f"{D}/alpha/{c}.npy").astype(np.float64)
        a3 = a[..., None]
        Fs = np.load(f"{D}/surf/{c}.npy").astype(np.float64)
        B = lin(f"{D46}/input/{m['bg']}__bgonly.png")
        R = a3 * Fs + (1 - a3) * lin(f"{R46}/{f46[('ref', m['bg'])]}")
        outs = {}
        if setn == "T1":
            for k in ["A", "S", "M2"]:
                outs[k] = lin(f"{R46}/{f46[(k, c)]}")
        else:
            for k in ["A", "S"]:
                outs[k] = lin(f"{OUT}/{base47[(k, c)]}")
        outs["O"] = a3 * outs["A"] + (1 - a3) * outs["S"]
        for v in ["full", "oracle"]:
            tag = f"{setn}/{v}/{c}"
            ah, Fh, Bh = npy(tag, "alpha")[..., None], npy(tag, "F"), npy(tag, "B")
            r = lin(f"{OUT}/{sep[(setn, c, v)]}")
            outs[f"SEP_{v}"] = (ah if v == "full" else a3) * Fh + (1 - (ah if v == "full" else a3)) * r
            rows.append(dict(set=setn, comp=c, bg=m["bg"], type=m["type"], method=f"sepq_{v}",
                             alpha_err=float(np.abs(ah[..., 0] - a)[pane].mean()),
                             F_err=float((np.abs(Fh - Fs).mean(-1))[pane & (a > 0.2)].mean() * 255)
                             if (pane & (a > 0.2)).any() else float("nan"),
                             B_err=float(np.abs(Bh - B).mean(-1)[pane].mean() * 255)))
        for k, y in outs.items():
            rows.append(dict(set=setn, comp=c, bg=m["bg"], type=m["type"], method=k, **measures(y, R, a)))
        print(setn, c, "scored", flush=True)


def val(setn, method, key, bg=None, typ=None):
    v = [r[key] for r in rows if r["set"] == setn and r["method"] == method and (bg is None or r["bg"] == bg)
         and (typ is None or r["type"] == typ) and key in r]
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def boot(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, est=float("nan"), lo=float("nan"), hi=float("nan"))
    b = [np.mean(rng.choice(v, len(v))) for _ in range(10000)]
    return dict(n=int(len(v)), est=float(np.mean(v)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)))


KEYS = ["err_pane", "err_surface", "err_transmission", "err_overlap", "err_outside", "corr"]
res = {}
for setn in ["T1", "T2"]:
    bgs = sorted({r["bg"] for r in rows if r["set"] == setn})
    meths = sorted({r["method"] for r in rows if r["set"] == setn and not r["method"].startswith("sepq")})
    if not bgs:
        continue
    summ = {m: {k: boot([val(setn, m, k, bg=b) for b in bgs]) for k in KEYS} for m in meths}
    sepq = {v: {k: boot([val(setn, f"sepq_{v}", k, bg=b) for b in bgs]) for k in ["alpha_err", "F_err", "B_err"]}
            for v in ["full", "oracle"]}
    types = sorted({r["type"] for r in rows if r["set"] == setn})
    by_type = {t: {m: val(setn, m, "err_pane", typ=t) for m in meths} for t in types}
    paired = {}
    for m in ["SEP_full", "SEP_oracle"]:
        paired[f"{m}-O"] = {k: boot([val(setn, m, k, bg=b) - val(setn, "O", k, bg=b) for b in bgs]) for k in KEYS}
        paired[f"{m}-O"]["n_better"] = int(sum(val(setn, m, "err_pane", bg=b) < val(setn, "O", "err_pane", bg=b)
                                               for b in bgs))
    res[setn] = dict(summary=summ, separation=sepq, by_type=by_type, paired=paired, n_backgrounds=len(bgs))


def meets(setn, m):
    s, p = res[setn]["summary"], res[setn]["paired"][f"{m}-O"]
    rel = p["err_pane"]["est"] / s["O"]["err_pane"]["est"]
    regions = {k: p[k]["est"] / s["O"][k]["est"] for k in ["err_surface", "err_transmission", "err_overlap"]
               if np.isfinite(s["O"][k]["est"]) and np.isfinite(p[k]["est"])}
    return bool(rel <= -0.25 and p["err_pane"]["hi"] < 0 and all(v <= 0.05 for v in regions.values())), rel, regions


# controls
checks = run47["checks"]
c1 = []
for bg in m46["bgs"]:
    tag = f"C1/full/{bg}"
    if os.path.exists(f"{SEP}/{tag}_alpha.npy"):
        ah, Bh = npy(tag, "alpha"), npy(tag, "B")
        Ilin = lin(f"{D46}/input/{bg}__bgonly.png")
        c1.append(dict(bg=bg, mean_alpha_pane=float(ah[pane].mean()),
                       B_minus_I_x255=float(np.abs(Bh - Ilin).mean(-1)[pane].mean() * 255)))
controls = dict(pipeline=checks, pipeline_ok=bool(checks) and max(checks.values()) <= 1, no_glass=c1,
                no_glass_ok=bool(c1) and max(r["mean_alpha_pane"] for r in c1) <= 0.05)
okf, relf, regf = meets("T1", "SEP_full")
oko, relo, rego = meets("T1", "SEP_oracle")
verdict = ("SUCCESS" if okf else "PARTIAL" if oko else
           "No advantage is established for the tested separator; output blending remains the practical baseline.")
if not controls["pipeline_ok"]:
    verdict = "NOT INTERPRETABLE: pipeline check failed"

# T3 (descriptive): round 43's see-through ratio on rb_48's frozen tiles
t3 = {}
if ("T3", "rb_48", "full") in sep:
    reg = json.load(open(f"{D43}/regions43.json"))["rb_48"]
    see = [c for c in reg["see_candidates"] if c["k"] in (8, 10, 11)]                      # review43.json
    x = np.asarray(Image.open(f"{D43}/input/rb_48.png").convert("RGB")).astype(np.float64)
    H, W = x.shape[:2]

    def tmask(tiles):
        m = np.zeros((H, W), bool)
        for t in tiles:
            m[t["y"]:t["y"] + 64, t["x"]:t["x"] + 64] = True
        return m

    def band(z):
        return np.stack([gaussian_filter(z[..., c], 2.0) - gaussian_filter(z[..., c], 5.0) for c in range(3)], -1)

    ms, mb, bx = tmask(see), tmask(reg["tiles"]), band(x)

    def rho(y):
        by = band(y)
        ls = 1 - (by[ms] ** 2).mean() / (bx[ms] ** 2).mean()
        lb = 1 - (by[mb] ** 2).mean() / (bx[mb] ** 2).mean()
        return float(ls / lb) if lb >= 0.3 else float("nan")

    ah, Fh = npy("T3/full/rb_48", "alpha")[..., None], npy("T3/full/rb_48", "F")
    r = lin(f"{OUT}/{sep[('T3', 'rb_48', 'full')]}")
    out = ah * Fh + (1 - ah) * r
    out_srgb = np.where(out <= 0.0031308, 12.92 * out, 1.055 * np.clip(out, 0, 1) ** (1 / 2.4) - 0.055)
    Image.fromarray((np.clip(out_srgb, 0, 1) * 255).round().astype(np.uint8)).save(f"{RES}/T3_rb_48_sep_full.png")
    y = np.clip(out_srgb, 0, 1) * 255
    t3 = dict(SEP_full=rho(y), mean_alpha_hat_in_mask=float(ah[..., 0].mean()),
              real_f2=rho(np.asarray(Image.open(f"{D43}/targets/rb_48_f2.0.png").convert("RGB")).astype(np.float64)))
    for cond in ["original", "seethrough", "seethrough_rim"]:
        t3[f"r43_{cond}"] = rho(np.asarray(Image.open(f"{R43}/img/rb_48_{cond}_12.png").convert("RGB")).astype(float))

decision = dict(verdict=verdict, SEP_full_relative_pane_change=float(relf), SEP_full_region_change=regf,
                SEP_oracle_relative_pane_change=float(relo), SEP_oracle_region_change=rego,
                output_blend_abs_err_pane_x255=res["T1"]["summary"]["O"]["err_pane"]["est"], controls=controls,
                T3=t3, pisa_calls=run47["calls"])
json.dump(dict(decision=decision, results=res, rows=rows), open(f"{RES}/score47.json", "w"), indent=1, default=float)
for setn, r in res.items():
    print(f"\n{setn} ({r['n_backgrounds']} backgrounds)")
    for m, s in r["summary"].items():
        print(f"{m:11s} " + " ".join(f"{k.replace('err_', '')} {s[k]['est']:.2f}" for k in KEYS))
    print("separation:", {v: {k: round(q['est'], 3) for k, q in s.items()} for v, s in r["separation"].items()})
    for t, v in r["by_type"].items():
        print(" ", t, {m: round(x, 2) for m, x in v.items()})
    for k, v in r["paired"].items():
        print(" ", k, f"pane {v['err_pane']['est']:+.2f} [{v['err_pane']['lo']:+.2f},{v['err_pane']['hi']:+.2f}]",
              f"better {v['n_better']}/{r['n_backgrounds']}")
print("decision:", json.dumps(decision, indent=1, default=float))
