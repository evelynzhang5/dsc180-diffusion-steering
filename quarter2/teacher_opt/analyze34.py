"""Round 34 analysis: frozen decision for the optimised (student-input) activation edit.

Checks G1-G3, C1, C2 are enforced in run34.py before any optimisation (no optimisation output without them).
Error = revised metric (metrics34.off_mae). Request-only = round 31's request search re-selected with it.
Per location (post_pisa, mid, up0, up1), one combined image per target with all 4 case cells edited jointly; per case:
  valid  = |r_hat - t| <= max(2 px, 20 %) and focus guard <= 0.5/255
  contrast guard: contrast(edit) >= contrast(request) - 0.05 (detail/contrast not removed)
  failing validity or the guard -> ratio inf; request invalid and edit valid -> ratio 0.
FEASIBLE at a location if median(edit / request) <= 0.80, wins >= 6/8, and its unedited-cell change
(rocket_0, text_0; MAE vs baseline, max over targets) <= the largest request-only other-cell change + 0.5/255.
This is a privileged upper bound (the edit is optimised against the teacher's latent): FEASIBLE means an edit at
that location exists; not feasible means even an optimised edit at this budget did not reach it.
Convergence: if the last 10 % of steps still lowered the loss by > 5 %, the location is reported "not converged"
(a non-feasible result there is unresolved rather than negative).
usage: python analyze34.py <R34_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure18 as m18  # noqa: E402
import metrics34 as Q  # noqa: E402

RATIO_MAX, WINS_MIN, PRES_MARGIN, CON_MARGIN = 0.80, 6, 0.5, 0.05
out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
assert meta["gate_ok"], meta["gate"]
req, opt = meta["request"], meta["opt"]
valid = lambda d, t: abs(d["r_hat"] - t) <= m18.TOL(t) and d["focus_add"] <= Q.FOCUS_ADD_MAX
req_others = max(v["others"] for v in req.values() if v)

locs = {}
for loc in meta["locs"]:
    ratios, wins, cases, others, conv = [], 0, {}, 0.0, {}
    for t in Q.TARGETS:
        o = opt[f"{loc}@{t:g}"]
        L = o["info"]["losses"]
        n = max(1, len(L) // 10)
        conv[f"{t:g}"] = dict(loss0=L[0], best=o["info"]["best_loss"], late_drop=(L[-n - 1] - L[-1]) / max(L[-n - 1], 1e-12),
                              delta_over_h=o["info"]["delta_over_h"], energy_top1=o["info"]["energy_top1"],
                              energy_top5=o["info"]["energy_top5"])
        for k, p in Q.CASES:
            d, r = o["cells"][p], req[f"{p}@{t:g}"]
            others = max(others, d["others_vs_base"])
            ok = valid(d, t) and (r is None or d["contrast"] >= r["contrast"] - CON_MARGIN)
            ratio = (d["err"] / r["err"] if r else 0.0) if ok else float("inf")
            ratios.append(ratio)
            wins += ratio < 1
            cases[f"{p}@{t:g}"] = dict(ratio=ratio, err=d["err"], request_err=r and r["err"], r_hat=d["r_hat"],
                                       contrast=d["contrast"], request_contrast=r and r["contrast"], valid=valid(d, t),
                                       shape=d["shape"], err_old=d["err_old"])
    med = float(np.median(ratios))
    converged = all(c["late_drop"] <= 0.05 for c in conv.values())
    feasible = med <= RATIO_MAX and wins >= WINS_MIN and others <= req_others + PRES_MARGIN
    locs[loc] = dict(median_ratio=med, wins=int(wins), others_max=others, feasible=feasible, converged=converged,
                     convergence=conv, cases=cases)

summary = dict(feasible=[l for l, v in locs.items() if v["feasible"]],
               unresolved=[l for l, v in locs.items() if not v["feasible"] and not v["converged"]],
               gate=meta["gate"], request_others_max=req_others, steps=meta["steps"],
               locations={l: {k: v for k, v in d.items() if k != "cases"} for l, d in locs.items()})
json.dump(summary, open(f"{res}/teacher_opt.json", "w"), indent=1)
json.dump({l: d["cases"] for l, d in locs.items()}, open(f"{res}/teacher_opt_cases.json", "w"), indent=1)
print("gate", meta["gate"])
print("FEASIBLE:", summary["feasible"], "| unresolved (not converged):", summary["unresolved"])
for l, d in locs.items():
    c = d["convergence"]
    print(f"{l:9s} median {d['median_ratio']:.3f} wins {d['wins']}/8 others {d['others_max']:.2f} (req {req_others:.2f}) "
          f"converged {d['converged']} | " + " ".join(f"t{t}: loss {v['loss0']:.3f}->{v['best']:.3f} late {v['late_drop']:.2f} "
                                                      f"|d|/|h| {v['delta_over_h']:.2f} top1 {v['energy_top1']:.2f}"
                                                      for t, v in c.items()))
    print("   " + " ".join(f"{k}:{v['err']:.2f}/{v['request_err']:.2f}(r{v['r_hat']:.1f},c{v['contrast']:.2f})"
                           for k, v in d["cases"].items()))
