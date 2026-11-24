"""Round 40 stage 3 (CPU, between the dev and test phases): fit and FREEZE each method's settings.

Generated images (targets 4, 8, 12 px effective additional blur):
  - scenes: C.mapping_scenes (development, initially-sharp categories, blur measurable, no mask failure, no
    measurement flag); tiles: regions40 "tiles" only (behind the subject; checked: every tile's median depth is
    behind the focus depth). Front tiles never enter.
  - response of a scene to a setting = its delivered blur (blur40, median over good tiles).
  - objective, EQUAL WEIGHT PER SCENE: the mean response over scenes (each scene one value, whatever its tile count),
    piecewise-linear in the setting (K linear; f-number in log, ordered from weak to strong blur). The frozen setting is
    the first crossing of the target. If the target is outside the mean curve's range, the nearest grid end is used and
    the target is a CONTROL FAILURE for that method (reported; its outputs are never treated as matched unless the
    delivered blur itself is within tolerance).
  - reported: per-scene responses at every grid setting, each scene's residual at the frozen setting, mean |residual|.
  A scene unmeasurable at some setting is left out of that setting's mean; the counts are reported.
RealBokeh real-photo control (f/2.0 and the second aperture):
  - scenes: review40 realbokeh_dev_after_review (3 of the original 5; no test scene moved in).
  - objective: linear-light mean |output - real capture| over the whole image, equal weight per scene; the grid
    setting with the lowest mean is frozen. Per-scene curves are reported (3 scenes can disagree).
usage: python map40.py <data root> <stage 3 out root>   -> <out>/mapping.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blur40 as BL  # noqa: E402
import common40 as C  # noqa: E402

DATA, OUT = sys.argv[1], sys.argv[2]
scenes, regions, review = C.load(DATA)
SMOKE = os.environ.get("SMOKE", "0") == "1"
METHODS = {"bokehdiff": ("bd", "K", lambda v: v, False), "bokehlicious": ("bkl", "av", np.log, True)}


def check_behind(i):
    d = C.depth(DATA, i, regions[i]["size"])
    for t in regions[i]["tiles"]:
        assert np.median(d[t["y"]:t["y"] + 64, t["x"]:t["x"] + 64]) > regions[i]["focus_disp"], (i, t)


def crossing(settings, curve, target, xform, descending):
    """settings ordered weak -> strong blur; returns (setting, failure)."""
    s, c = np.array(settings, float), np.array(curve, float)
    ok = np.isfinite(c)
    s, c = s[ok], c[ok]
    for k in range(len(s) - 1):
        lo, hi = c[k], c[k + 1]
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            a, b = xform(s[k]), xform(s[k + 1])
            v = a + (target - lo) / (hi - lo) * (b - a)
            return float(np.exp(v) if descending else v), False
    j = int(np.argmin(np.abs(c - target)))
    return float(s[j]), True


def interp(settings, curve, v, xform):
    s, c = np.array([xform(x) for x in settings]), np.array(curve, float)
    ok = np.isfinite(c)
    if ok.sum() < 2:
        return float("nan")
    order = np.argsort(s[ok])
    return float(np.interp(xform(v), s[ok][order], c[ok][order]))


maps = C.mapping_scenes(scenes, regions, review)
bad = set(review["mask_failure"]) | set(review["measurement_flag"])
assert not set(maps) & bad and all(scenes[i]["category"] != "existing_blur" for i in maps)
rbdev = C.rb_dev_scenes(review)
assert not set(rbdev) & set(review["mask_failure"]) and all(scenes[i]["split"] == "dev" for i in rbdev)
out = dict(mapping_scenes=maps, rb_dev_scenes=rbdev, targets=C.TARGETS, tol="max(2, 0.2 r)", r2_min=C.R2_MIN)
for i in maps:
    check_behind(i)
stacks = {i: BL.TileStack(C.img(f"{DATA}/input/{i}.png"), regions[i]["tiles"]) for i in maps}

for meth, (pre, key, xform, desc) in METHODS.items():
    rows = json.load(open(f"{OUT}/{pre}_dev.json"))["rows"]
    settings = sorted({r[key] for r in rows}, reverse=desc)          # weak -> strong blur
    resp = {}
    for i in maps:
        resp[i] = {}
        for r in rows:
            if r["id"] == i and "upstreamconv" not in r["file"]:
                m = stacks[i].measure(C.img(f"{OUT}/{r['file']}"))
                resp[i][r[key]] = dict(r=m["r"], n_good=m["n_good"], n_tiles=m["n_tiles"])
    curve = [float(np.nanmean([resp[i][s]["r"] for i in maps if s in resp[i]])) if any(
        np.isfinite(resp[i][s]["r"]) for i in maps if s in resp[i]) else float("nan") for s in settings]
    n_per = [int(sum(np.isfinite(resp[i][s]["r"]) for i in maps if s in resp[i])) for s in settings]
    gen = {}
    for t in C.TARGETS:
        s, fail = crossing(settings, curve, t, xform, desc)
        per = {i: interp(settings, [resp[i][x]["r"] if x in resp[i] else np.nan for x in settings], s, xform)
               for i in maps}
        res = {i: v - t for i, v in per.items()}
        fin = [abs(v) for v in res.values() if np.isfinite(v)]
        gen[f"{t:g}"] = dict(setting=s, control_failure=fail, per_scene_at_setting=per,
                             mean_abs_residual=float(np.mean(fin)) if fin else float("nan"),
                             within_tol=int(sum(abs(v) <= C.TOL(t) for v in fin)), n=len(fin))
        print(f"{meth:13s} target {t:4g}px -> {key} {s:6.2f}{'  CONTROL FAILURE' if fail else ''}   mean |res| "
              f"{gen[f'{t:g}']['mean_abs_residual']:.2f}  within tol {gen[f'{t:g}']['within_tol']}/{len(fin)}",
              flush=True)
    # RealBokeh: linear MAE to the real captures
    rbm = {}
    for a in ["2.0", "second"]:
        per = {}
        for i in rbdev:
            avs = scenes[i]["apertures"]
            av = 2.0 if a == "2.0" else [v for v in avs if v != 2.0][0]
            tgt = C.to_lin(C.img(f"{DATA}/targets/{i}_f{av:g}.png" if os.path.exists(f"{DATA}/targets/{i}_f{av:g}.png")
                                 else f"{DATA}/targets/{i}_f{av}.png"))
            per[i] = {}
            for r in rows:
                if r["id"] == i and "upstreamconv" not in r["file"]:
                    per[i][r[key]] = float(np.abs(C.to_lin(C.img(f"{OUT}/{r['file']}")) - tgt).mean() * 255)
            per[i]["input"] = float(np.abs(C.to_lin(C.img(f"{DATA}/input/{i}.png")) - tgt).mean() * 255)
        mean = {s: float(np.mean([per[i][s] for i in rbdev if s in per[i]])) for s in settings}
        best = min(mean, key=mean.get)
        rbm[a] = dict(setting=float(best), mean_mae_lin=mean, per_scene=per)
        print(f"{meth:13s} RealBokeh {a:6s} -> {key} {best:g}  mean linear MAE {mean[best]:.2f}/255  per scene "
              + " ".join(f"{i}:{min((v for k, v in per[i].items() if k != 'input')):.2f}@"
                         f"{min((k for k in per[i] if k != 'input'), key=lambda k: per[i][k]):g}" for i in rbdev),
              flush=True)
    out[meth] = dict(settings=settings, mean_curve=curve, n_scenes_per_setting=n_per, responses=resp, gen=gen, rb=rbm)

json.dump(out, open(f"{OUT}/mapping.json", "w"), indent=1, default=float)
print("mapping frozen:", f"{OUT}/mapping.json", flush=True)
