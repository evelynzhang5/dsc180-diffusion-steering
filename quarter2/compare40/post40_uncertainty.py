"""Round 40, POST HOC (after review #6; not part of the frozen protocol): scene-level uncertainty and the head-to-head
RealBokeh comparison. Everything is aggregated per scene first (outputs of one scene are not independent), then a
percentile bootstrap over scenes (10,000 resamples, seed 0) gives 95% intervals.
usage (repo root): python quarter2/compare40/post40_uncertainty.py
"""
import json

import numpy as np

S = json.load(open("quarter2/compare40/results/stage3/score40.json"))
rng = np.random.default_rng(0)


def boot(v, stat=np.median, n=10000):
    v = np.asarray([x for x in v if np.isfinite(x)])
    b = [stat(rng.choice(v, len(v))) for _ in range(n)]
    return len(v), float(stat(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def show(name, r, unit=""):
    print(f"{name:62s} n={r[0]:2d}  {r[1]:+.3f}{unit}  [{r[2]:+.3f}, {r[3]:+.3f}]")
    return dict(n=r[0], est=r[1], lo=r[2], hi=r[3])


out = {}
print("RealBokeh, paired per scene: BokehDiff MAE - Bokehlicious MAE (linear, /255; positive = Bokehlicious closer)")
rb = S["realbokeh"]
for a in ["2.0", "second"]:
    for k in ["mae_lin", "mae_lin_gain"]:
        m = {(r["id"], r["method"]): r[k] for r in rb if r["aperture"] == a}
        ids = sorted({i for i, _ in m})
        d = [m[(i, "bokehdiff")] - m[(i, "bokehlicious")] for i in ids]
        r = show(f"  f/{a} {k} median difference", boot(d))
        out[f"rb/{a}/{k}"] = dict(r, bkl_closer=int(sum(x > 0 for x in d)), n_scenes=len(d))
        print(f"{'':64s}Bokehlicious closer in {sum(x > 0 for x in d)}/{len(d)} scenes")

rows = [r for r in S["rows"] if r["category"] != "existing_blur"]
ids = sorted({r["id"] for r in rows})
print("\nGenerated initially-sharp test scenes (scene = mean over its 3 targets)")
for meth in ["bokehdiff", "bokehlicious"]:
    ok = [np.mean([r["control_ok"] for r in rows if r["id"] == i and r["method"] == meth]) for i in ids]
    out[f"control_ok_rate/{meth}"] = show(f"  {meth} control within tolerance (fraction of targets)", boot(ok, np.mean))
    fo = [np.mean([r["focus_ok"] for r in rows if r["id"] == i and r["method"] == meth and "focus_ok" in r])
          for i in ids if any("focus_ok" in r for r in rows if r["id"] == i)]
    out[f"focus_ok_rate/{meth}"] = show(f"  {meth} focus check passed (fraction of targets)", boot(fo, np.mean))


def paired(metric, matched):
    d = []
    for i in ids:
        v = {}
        for meth in ["bokehdiff", "bokehlicious"]:
            x = [r.get(metric, np.nan) for r in rows if r["id"] == i and r["method"] == meth
                 and (r.get("matched") or not matched)]
            v[meth] = np.nanmean(x) if x and np.isfinite(x).any() else np.nan
        d.append(v["bokehdiff"] - v["bokehlicious"])
    return d


for metric, matched in [("hf_ratio", False), ("hf_ratio", True), ("halo_sharp", True)]:
    out[f"paired/{metric}/{'matched' if matched else 'all'}"] = show(
        f"  BokehDiff - Bokehlicious {metric} ({'matched cells' if matched else 'all cells'})", boot(paired(metric, matched)))
json.dump(out, open("quarter2/compare40/results/stage3/uncertainty40.json", "w"), indent=1)
