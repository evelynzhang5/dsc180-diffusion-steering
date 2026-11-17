"""Round 29 analysis. Usage: python analyze29.py <R29_OUT> <results dir>"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import metrics29 as Q  # noqa: E402
import patches as PT  # noqa: E402

out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
img, cid, where = Q.scene()
im = img.astype(np.float32)
ints = {k: D.interior(cid, k) for k in PT.FREE + [0, 8]}
fm = Q.focus_mask(cid)
st = m18.stack(im, "disk")
B = {k: M.stack_at(st, ints[k]) for k, _ in Q.CASES}
del st
tg = {t: Q.target(img, t) for t in Q.TARGETS}
ld = lambda tag: np.asarray(Image.open(f"{out}/img/{tag}.png")).astype(np.float32)
base = {t: ld(f"base_t{t:.1f}") for t in Q.TARGETS}
foc0 = {t: Q.raw_mae(base[t], im, fm) for t in Q.TARGETS}
med = lambda x: float(np.median(x))


def describe(tag, k, t):
    y = ld(tag)
    others = [c for c in PT.FREE + [0, 8] if c != k]
    return dict(err=Q.aff_mae(y, tg[t], ints[k]), raw=Q.raw_mae(y, tg[t], ints[k]),
                r_hat=M.fit_fast(y[ints[k]], B[k])["v"], focus_add=Q.raw_mae(y, im, fm) - foc0[t],
                others_vs_base=float(np.mean([Q.raw_mae(y, base[t], ints[c]) for c in others])))


g = meta["gate"]
gate = dict(repro_r25=max(g["repro_r25"].values()) <= 1.0, repeat=max(g["repeat"].values()) == 0.0,
            zero_edit=max(g["zero_edit"].values()) == 0.0)
gate["passed"] = all(gate.values())

cases = {}
for k, p in Q.CASES:
    for t in Q.TARGETS:
        key = f"{p}@{t:g}"
        b = describe(f"base_t{t:.1f}", k, t)
        ins = [r for r in meta["rows"] if r["kind"] == "input" and r["cell"] == k and r["target"] == t]
        bi = min(ins, key=lambda r: r["err"])
        cases[key] = dict(cell=k, patch=p, target=t, base=b, input=dict(describe(bi["tag"], k, t), factor=bi["factor"],
                          at_edge=bi["factor"] <= 0.3 or bi["factor"] >= 2.0), loc={})
        for loc in Q.LOCS:
            fr = [r for r in meta["rows"] if r["kind"] == "feature" and r["loc"] == loc and r["cell"] == k and r["target"] == t]
            cand = [dict(describe(r["tag"], k, t), width=r["width"], alpha=r["alpha"], tag=r["tag"]) for r in fr]
            ok = [c for c in cand if c["focus_add"] <= Q.FOCUS_ADD_MAX]
            best = min(ok + [dict(b, width=0, alpha=0.0, tag=f"base_t{t:.1f}")], key=lambda c: c["err"])
            cases[key]["loc"][loc] = dict(best=best, all=cand, n_admissible=len(ok))

within = lambda c, t: abs(c["r_hat"] - t) <= m18.TOL(t)
locs = {}
for loc in Q.LOCS:
    ratio_in = [c["loc"][loc]["best"]["err"] / c["input"]["err"] for c in cases.values()]
    ratio_base = [c["loc"][loc]["best"]["err"] / c["base"]["err"] for c in cases.values()]
    wins = sum(r < 1 for r in ratio_in)
    tol_loc = sum(within(c["loc"][loc]["best"], c["target"]) for c in cases.values())
    tol_base = sum(within(c["base"], c["target"]) for c in cases.values())
    edge = sum(abs(c["loc"][loc]["best"]["alpha"]) >= 0.1 for c in cases.values())
    passed = med(ratio_in) <= 0.80 and wins >= 6 and tol_loc >= tol_base
    locs[loc] = dict(median_ratio_vs_input=med(ratio_in), median_ratio_vs_base=med(ratio_base), wins_vs_input=wins,
                     within_tol=tol_loc, within_tol_base=tol_base, at_alpha_edge=edge, passed=passed,
                     unresolved=(not passed) and edge >= 4,
                     max_focus_add=max(c["loc"][loc]["best"]["focus_add"] for c in cases.values()),
                     median_others_vs_base=med([c["loc"][loc]["best"]["others_vs_base"] for c in cases.values()]),
                     median_raw_ratio_vs_base=med([c["loc"][loc]["best"]["raw"] / c["base"]["raw"] for c in cases.values()]))
passing = [l for l in Q.LOCS if locs[l]["passed"]]
if passing:
    best_loc = min(passing, key=lambda l: locs[l]["median_ratio_vs_input"])
    outcome = f"LOC({best_loc}): small edits at this location correct output errors that changing the blur request does not"
else:
    best_loc = min(Q.LOCS, key=lambda l: locs[l]["median_ratio_vs_input"])
    unres = [l for l in Q.LOCS if locs[l]["unresolved"]]
    outcome = "NONE: this smoothing/sharpening edit family did not demonstrate useful control at these locations" + \
        (f" (search range unresolved at {unres})" if unres else "")

summary = dict(gate=gate, gate_values=g, outcome=outcome, best_location=best_loc, locations=locs,
               input_only=dict(median_ratio_vs_base=med([c["input"]["err"] / c["base"]["err"] for c in cases.values()]),
                               factors={k: c["input"]["factor"] for k, c in cases.items()},
                               at_edge=[k for k, c in cases.items() if c["input"]["at_edge"]]),
               shapes=meta["shapes"], module_names=meta["module_names"], sec_per_gen=meta["sec_per_gen"],
               n_generations=meta["n_generations"])
json.dump(summary, open(f"{res}/steer_locate.json", "w"), indent=1)
json.dump({k: dict(c, loc={l: dict(v, all=v["all"]) for l, v in c["loc"].items()}) for k, c in cases.items()},
          open(f"{res}/steer_locate_cases.json", "w"))

print("gate", gate, g)
print("shapes", meta["shapes"])
print("OUTCOME", outcome)
print("case                 base   input(f)        " + "  ".join(f"{l:>10}" for l in Q.LOCS))
for key, c in cases.items():
    print(f"{key:20s} {c['base']['err']:5.2f}  {c['input']['err']:5.2f} ({c['input']['factor']:.3f})  " +
          "  ".join(f"{c['loc'][l]['best']['err']:5.2f}{'*' if abs(c['loc'][l]['best']['alpha']) >= 0.1 else ' '}    " for l in Q.LOCS))
for l, v in locs.items():
    print(l, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print("input-only median err ratio vs baseline %.3f" % summary["input_only"]["median_ratio_vs_base"])

# paired crops at 11.625 px: target | baseline | input-only best | best edit at the best location  (16 tiles)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
t = 11.625
fig, ax = plt.subplots(4, 4, figsize=(10, 10.6))
for i, (k, p) in enumerate(Q.CASES):
    c = cases[f"{p}@{t:g}"]
    ys, xs = np.where(cid == k)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    be = c["loc"][best_loc]["best"]
    tiles = [(tg[t], "target (disk)"), (base[t], f"baseline {c['base']['err']:.2f}"),
             (ld(f"in_c{k}_t{t:.1f}_f{c['input']['factor']:.4f}"), f"input x{c['input']['factor']:.2f} {c['input']['err']:.2f}"),
             (ld(be["tag"]), f"{best_loc} w{be['width']} a{be['alpha']:+.2f} {be['err']:.2f}")]
    for j, (a, lab) in enumerate(tiles):
        ax[i, j].imshow(np.clip(a[sl], 0, 255).astype(np.uint8))
        ax[i, j].set_title(f"{p}: {lab}" if j == 0 else lab, fontsize=8)
        ax[i, j].axis("off")
fig.suptitle(f"Round 29, target {t} px: error = gain/offset-adjusted MAE vs disk target (lower is better)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{res}/crops_t11.6.png", dpi=110)
print("wrote", f"{res}/crops_t11.6.png")
