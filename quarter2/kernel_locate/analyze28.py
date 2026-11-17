"""Round 28 analysis. Usage: python analyze28.py <R28_OUT> <results dir>"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402

out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
GAMMA = 2.2
TARGETS = [3, 4, 5, 6, 8, 11.625, 15.5]
PT_T = [5, 8, 11.625]
VALID_T = [5, 6, 8, 11.625, 15.5]          # targets used for the frozen decisions
lib = PT.library()
groups = [[p for p in lib if p["split"] == "test"][0::2], [p for p in lib if p["split"] == "test"][1::2]]
med = lambda x: float(np.median(x))
ld = lambda f: np.asarray(Image.open(f)).astype(np.float32)


def lin_stack(img):
    x = (img.astype(np.float64) / 255.0) ** GAMMA
    return M.R_GRID, [(np.clip(m18.blur(x, m18.disk(r)), 0, 1) ** (1 / GAMMA) * 255.0).astype(np.float32) for r in M.R_GRID]


def lin_blur(img, r):
    x = (img.astype(np.float64) / 255.0) ** GAMMA
    return np.clip(m18.blur(x, m18.disk(r)), 0, 1) ** (1 / GAMMA) * 255.0


rng = np.random.default_rng(0)
rows, val = [], []
for gi, grp in enumerate(groups):
    for rot in (0, 2, 4):
        img, cid, where = PT.layout(grp, rot)
        im = img.astype(np.float32)
        ints = {k: D.interior(cid, k) for k in PT.FREE}
        st_s = m18.stack(im, "disk")
        Bs = {k: M.stack_at(st_s, ints[k]) for k in PT.FREE}
        del st_s
        st_l = lin_stack(img)
        Bl = {k: M.stack_at(st_l, ints[k]) for k in PT.FREE}
        del st_l
        def measure(y, label, t):
            for k in PT.FREE:
                rows.append(dict(set=label, target=t, patch=where[k], rot=rot,
                                 r_srgb=M.fit_fast(y[ints[k]], Bs[k])["v"], r_lin=M.fit_fast(y[ints[k]], Bl[k])["v"]))
        if rot == 0:                                       # validation of the linear-light estimator
            for r in (5, 8, 12):
                y = m18.degrade(lin_blur(img, r), rng)
                for k in PT.FREE:
                    v = M.fit_fast(y[ints[k]], Bl[k])["v"]
                    val.append(dict(r=r, patch=where[k], r_hat=v, ok=abs(v - r) <= max(1.0, 0.1 * r)))
        for t in TARGETS:
            tag = f"g{gi}_r{rot}_t{t:.1f}"
            measure(ld(f"{out}/img/{tag}_ft.png"), "real_ft", t)
            measure(ld(f"{out}/img/{tag}_orig.png"), "real_orig", t)
        for t in PT_T:
            measure(np.clip(m18.blur(im, m18.disk(t)), 0, 255), "pixel_srgb", t)
            measure(lin_blur(img, t), "pixel_lin", t)
            for kind in ("srgb", "lin"):
                for dec in ("ft", "orig"):
                    measure(ld(f"{out}/img/path_g{gi}_r{rot}_{kind}{t:.1f}_{dec}.png"), f"path_{kind}_{dec}", t)
    print("measured group", gi, flush=True)

patches = sorted({r["patch"] for r in rows})
def cv(label, key, ts):
    out_ = []
    for t in ts:
        v = [np.mean([r[key] for r in rows if r["set"] == label and r["target"] == t and r["patch"] == p]) for p in patches]
        if all(np.isfinite(v)) and len(v) == len(patches):
            out_.append(float(np.std(v) / np.mean(v)))
    return med(out_) if out_ else float("nan")
def pattern(label, key, ts):
    refs = {t: med([r[key] for r in rows if r["set"] == label and r["target"] == t]) for t in ts}
    return {p: med([r[key] / refs[r["target"]] for r in rows if r["set"] == label and r["patch"] == p and r["target"] in ts]) for p in patches}
sp = lambda a, b: float(spearmanr([a[p] for p in patches], [b[p] for p in patches]).correlation)

lin_val_cv = {r: float(np.std(x := [v["r_hat"] for v in val if v["r"] == r]) / np.mean(x)) for r in (5, 8, 12)}
gate = dict(reproduces_round25=meta["repro_max_diff"] <= 1.0, linear_estimator_valid=all(v["ok"] for v in val) and max(lin_val_cv.values()) <= 0.03)
gate["passed"] = all(gate.values())

sets = ["real_ft", "real_orig", "pixel_srgb", "pixel_lin", "path_srgb_ft", "path_srgb_orig", "path_lin_ft", "path_lin_orig"]
cvs = {s: dict(srgb=cv(s, "r_srgb", VALID_T if s.startswith("real") else PT_T), lin=cv(s, "r_lin", VALID_T if s.startswith("real") else PT_T)) for s in sets}
pats = {s: pattern(s, "r_srgb", VALID_T if s.startswith("real") else PT_T) for s in sets}
rho_phys = sp(pats["real_ft"], pats["pixel_lin"])
rho_dec = sp(pats["real_ft"], pats["real_orig"])
cv_ratio_phys = cvs["real_ft"]["lin"] / cvs["real_ft"]["srgb"]
cv_ratio_dec = cvs["real_orig"]["srgb"] / cvs["real_ft"]["srgb"]
if cv_ratio_phys <= 0.5 and rho_phys >= 0.7:
    phys = "PHYS: the per-patch pattern is largely what linear-light lens blur produces (target definition, not a model failure)"
elif cv_ratio_phys >= 0.8 and rho_phys < 0.4:
    phys = "NOTPHYS: linear-light lens blur does not explain the pattern"
else:
    phys = "PARTIAL: linear-light blur explains part of the pattern"
if cv_ratio_dec <= 0.5:
    loc = "DEC: the fine-tuned decoder creates most of the pattern"
elif cv_ratio_dec >= 0.8:
    loc = "UNET: the pattern is already in the latents (decided before the decoder)"
else:
    loc = "MIXED: decoder and latents both contribute"
summary = dict(gate=gate, physics=phys, localization=loc, cv=cvs, cv_ratio_lin_over_srgb=cv_ratio_phys,
               cv_ratio_orig_over_ft=cv_ratio_dec, spearman_real_vs_pixel_lin=rho_phys, spearman_ft_vs_orig=rho_dec,
               spearman_real_vs_pixel_srgb=sp(pats["real_ft"], pats["pixel_srgb"]), patterns=pats,
               linear_estimator_validation=dict(cv=lin_val_cv, all_within=all(v["ok"] for v in val)),
               vae_changed=meta["vae_changed"], vae_total=meta["vae_total"],
               real_lin_by_target={f"{t:g}": med([r["r_lin"] / t for r in rows if r["set"] == "real_ft" and r["target"] == t]) for t in TARGETS},
               real_srgb_by_target={f"{t:g}": med([r["r_srgb"] / t for r in rows if r["set"] == "real_ft" and r["target"] == t]) for t in TARGETS})
json.dump(summary, open(f"{res}/locate.json", "w"), indent=1)
json.dump(dict(rows=rows, validation=val), open(f"{res}/locate_rows.json", "w"))
print("gate", gate)
print("PHYSICS", phys, "| CV real lin/srgb %.3f / %.3f" % (cvs["real_ft"]["lin"], cvs["real_ft"]["srgb"]), "| Spearman real vs pixel-lin %.2f" % rho_phys)
print("LOCALIZATION", loc, "| CV orig/ft %.3f / %.3f" % (cvs["real_orig"]["srgb"], cvs["real_ft"]["srgb"]), "| Spearman ft vs orig %.2f" % rho_dec)
print("all CVs", {k: {a: round(b, 3) for a, b in v.items()} for k, v in cvs.items()})
for s in ("real_ft", "real_orig", "pixel_lin", "path_srgb_ft", "path_lin_ft"):
    print("pattern", s, {p: round(v, 2) for p, v in pats[s].items()})
print("VAE tensors changed", meta["vae_changed"], "of", meta["vae_total"], "| linear estimator CV", {k: round(v, 3) for k, v in lin_val_cv.items()})
print("real/target by target: srgb", {k: round(v, 2) for k, v in summary["real_srgb_by_target"].items()}, "lin", {k: round(v, 2) for k, v in summary["real_lin_by_target"].items()})
