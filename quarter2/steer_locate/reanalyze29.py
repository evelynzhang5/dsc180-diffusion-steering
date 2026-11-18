"""Round 29 exploratory reanalysis (saved outputs only; the frozen outcome is unchanged).

Same selection rule for every method: among a case's candidates, keep only those with
  - delivered blur within tolerance: |r_hat - target| <= max(2 px, 20 %), and
  - focused-region MAE vs the source added over the baseline <= 0.5/255,
then take the lowest primary error. No valid candidate = attainment failure (kept, not dropped).
Methods: input-only request adjustment (24 requests incl. the baseline), and each feature location
(8 edits + the baseline, i.e. alpha = 0).
Also splits each selected output's error into size vs shape: err_size = error of an ideal disk of the
output's own measured radius against the target; err_shape = error of the output against that ideal disk.
Usage: python reanalyze29.py <R29_OUT> <results dir>
"""
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

out, res = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{out}/conds.json"))
cases = json.load(open(f"{res}/steer_locate_cases.json"))
img, cid, _ = Q.scene()
im = img.astype(np.float32)
fm = Q.focus_mask(cid)
st = m18.stack(im, "disk")
ld = lambda tag: np.asarray(Image.open(f"{out}/img/{tag}.png")).astype(np.float32)
base = {t: ld(f"base_t{t:.1f}") for t in Q.TARGETS}
foc0 = {t: Q.raw_mae(base[t], im, fm) for t in Q.TARGETS}
tg = {t: Q.target(img, t) for t in Q.TARGETS}
med = lambda x: float(np.median(x)) if len(x) else float("nan")
valid = lambda c, t: abs(c["r_hat"] - t) <= m18.TOL(t) and c["focus_add"] <= Q.FOCUS_ADD_MAX

sel = {}
for k, p in Q.CASES:
    m = D.interior(cid, k)
    B = M.stack_at(st, m)
    for t in Q.TARGETS:
        key = f"{p}@{t:g}"
        c = cases[key]
        b = dict(c["base"], tag=f"base_t{t:.1f}", alpha=0.0, width=0)
        inp = []
        for r in meta["rows"]:
            if r["kind"] == "input" and r["cell"] == k and r["target"] == t:
                y = ld(r["tag"])
                inp.append(dict(tag=r["tag"], factor=r["factor"], err=r["err"], r_hat=M.fit_fast(y[m], B)["v"],
                                focus_add=Q.raw_mae(y, im, fm) - foc0[t]))
        methods = {"input": inp}
        for loc in Q.LOCS:
            methods[loc] = c["loc"][loc]["all"] + [b]
        sel[key] = {}
        for name, cand in methods.items():
            ok = [x for x in cand if valid(x, t)]
            if not ok:
                sel[key][name] = dict(valid=False, n_valid=0, n=len(cand))
                continue
            x = min(ok, key=lambda x: x["err"])
            ideal = np.clip(m18.blur(im, m18.disk(x["r_hat"])), 0, 255)
            y = ld(x["tag"])
            sel[key][name] = dict(valid=True, n_valid=len(ok), n=len(cand), err=x["err"], r_hat=x["r_hat"], tag=x["tag"],
                                  alpha=x.get("alpha"), width=x.get("width"), factor=x.get("factor"),
                                  err_size=Q.aff_mae(ideal, tg[t], m), err_shape=Q.aff_mae(y, ideal, m))
        sel[key]["baseline_valid"] = valid(b, t)
        sel[key]["baseline_err"] = b["err"]

summ = {}
for loc in Q.LOCS:
    both = [k for k in sel if sel[k]["input"]["valid"] and sel[k][loc]["valid"]]
    ratios = [sel[k][loc]["err"] / sel[k]["input"]["err"] for k in both]
    summ[loc] = dict(attained=sum(sel[k][loc]["valid"] for k in sel), compared=len(both),
                     median_ratio_vs_input=med(ratios), wins=sum(r < 1 for r in ratios),
                     ties=sum(r == 1 for r in ratios),
                     chose_baseline=sum(sel[k][loc]["valid"] and sel[k][loc]["alpha"] == 0.0 for k in sel),
                     at_alpha_edge=sum(sel[k][loc]["valid"] and abs(sel[k][loc]["alpha"]) >= 0.1 for k in sel),
                     signs=[(k, sel[k][loc].get("alpha"), sel[k][loc].get("width")) for k in sel])
summary = dict(note="exploratory reanalysis; round 29's frozen outcome (NONE) is unchanged",
               input_attained=sum(sel[k]["input"]["valid"] for k in sel),
               baseline_valid=sum(sel[k]["baseline_valid"] for k in sel), locations=summ, cases=sel)
json.dump(summary, open(f"{res}/reanalysis.json", "w"), indent=1)

print("baseline valid in", summary["baseline_valid"], "of 8 | input-only attains", summary["input_attained"], "of 8")
print(f"{'case':20s} {'base':>6s} {'input':>16s} " + " ".join(f"{l:>17s}" for l in Q.LOCS))
fmt = lambda s: (f"{s['err']:5.2f} r{s['r_hat']:4.1f}" + (f" {s['alpha']:+.2f}" if s.get("alpha") is not None and s.get("factor") is None else f" x{s['factor']:.2f}" if s.get("factor") else "")) if s["valid"] else "   FAIL"
for k, s in sel.items():
    print(f"{k:20s} {s['baseline_err']:5.2f}{'v' if s['baseline_valid'] else ' '} {fmt(s['input']):>16s} " + " ".join(f"{fmt(s[l]):>17s}" for l in Q.LOCS))
for l, v in summ.items():
    print(l, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "signs"})
print("size vs shape error of the selected valid outputs (median over cases):")
for name in ["input"] + Q.LOCS:
    v = [sel[k][name] for k in sel if sel[k][name]["valid"]]
    print(f"  {name:10s} err {med([x['err'] for x in v]):.2f}  size part {med([x['err_size'] for x in v]):.2f}  shape part {med([x['err_shape'] for x in v]):.2f}")
