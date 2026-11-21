"""Round 37 analysis (rules fixed before the run).

Protected-region change = mean |y - baseline| /255 in each of focus, rocket_0, text_0, anchor_tl, anchor_br interiors;
P(arm) = mean over those five. In-region error = revised metric (metrics34.off_mae) over the 4 case cells.
Controls (both targets):
  K0 zero_free: max |zero edit, unfrozen - baseline| <= 1 (uint8)  (inner passes replay the real call exactly; added after
               run 1, which failed K1/K2 because PISA's dropout/occlusion jitter drew new random numbers per pass)
  K1 zero_frz: every protected region changes <= 0.1/255  (freezing the statistics alone is harmless)
  K2 repro:    max |repro - free| <= 1 (uint8)             (the saved edit reproduces round 35's optimum)
MECHANISM SUPPORTED if, at both targets, P(d35_frz) <= 0.5 x P(free) AND median in-region error(d35_frz) <= 1.2 x
  median in-region error(free). (Not supported -> move on from this mechanism.)
PRESERVING CORRECTION FOUND if opt_frz passes round 34's FEASIBLE rule: valid (|r_hat - t| <= max(2, 20 %), focus guard
  <= 0.5/255), contrast >= request - 0.05, median ratio to request-only <= 0.80, wins >= 6/8, unedited-cell change
  <= largest request-only other-cell change + 0.5/255.
usage: python analyze37.py <R37_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure18 as m18  # noqa: E402
import metrics34 as Q  # noqa: E402

REG = ["focus", "rocket_0", "text_0", "anchor_tl", "anchor_br"]
out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
assert meta["gate_ok"], meta["gate"]
req = meta["request"]
req_others = max(v["others"] for v in req.values() if v)
valid = lambda d, t: abs(d["r_hat"] - t) <= m18.TOL(t) and d["focus_add"] <= Q.FOCUS_ADD_MAX
R, K0, K1, K2, mech = {}, True, True, True, True
feas_ratios, feas_wins, feas_others = [], 0, 0.0
for tk, arms in meta["arms"].items():
    t = float(tk)
    P = {a: float(np.mean([v["regions"][r] for r in REG])) for a, v in arms.items()}
    E = {a: float(np.median([d["err"] for d in v["cells"].values()])) for a, v in arms.items()}
    k0 = arms["zero_free"]["maxdiff_vs_base"] <= 1
    K0 = K0 and k0
    k1 = max(arms["zero_frz"]["regions"][r] for r in REG) <= 0.1
    k2 = arms["repro"]["maxdiff_vs_free"] <= 1
    m = P["d35_frz"] <= 0.5 * P["free"] and E["d35_frz"] <= 1.2 * E["free"]
    K1, K2, mech = K1 and k1, K2 and k2, mech and m
    o = arms["opt_frz"]
    for k, p in Q.CASES:
        d, r = o["cells"][p], req[f"{p}@{t:g}"]
        ok = valid(d, t) and (r is None or d["contrast"] >= r["contrast"] - 0.05)
        ratio = (d["err"] / r["err"] if r else 0.0) if ok else float("inf")
        feas_ratios.append(ratio)
        feas_wins += ratio < 1
        feas_others = max(feas_others, d["others_vs_base"])
    R[tk] = dict(protected_mean=P, inregion_median_err=E, K0=k0, K0_maxdiff=arms["zero_free"]["maxdiff_vs_base"],
                 K1_maxdiff=arms["zero_frz"]["maxdiff_vs_base"], K2_maxdiff=arms["repro"]["maxdiff_vs_free"], K1=k1, K2=k2, mechanism=m,
                 regions={a: v["regions"] for a, v in arms.items()},
                 focus_add={a: max(d["focus_add"] for d in v["cells"].values()) for a, v in arms.items()},
                 cases={a: {p: dict(err=d["err"], r_hat=d["r_hat"], contrast=d["contrast"], request=req[f"{p}@{t:g}"]["err"])
                            for p, d in v["cells"].items()} for a, v in arms.items()},
                 opt_info={a: {k: v["info"][k] for k in ("loss0", "best_loss", "best_in", "best_out", "steps", "delta_over_h",
                                                          "energy_top1")} for a, v in arms.items() if v.get("info")})
med = float(np.median(feas_ratios))
feasible = med <= 0.80 and feas_wins >= 6 and feas_others <= req_others + 0.5
summary = dict(controls_ok=bool(K0 and K1 and K2), K0=bool(K0), K1=bool(K1), K2=bool(K2), mechanism_supported=bool(mech),
               preserving_correction_found=bool(feasible),
               opt_frz=dict(median_ratio=med, wins=int(feas_wins), others_max=feas_others, request_others_max=req_others),
               targets=R)
json.dump(summary, open(f"{res}/norm_freeze.json", "w"), indent=1)
for tk, r in R.items():
    print(f"t={tk}: K0 {r['K0']} ({r['K0_maxdiff']:.0f}) K1 {r['K1']} (max {r['K1_maxdiff']:.0f}) K2 {r['K2']} ({r['K2_maxdiff']:.0f}) "
          f"mechanism {r['mechanism']}")
    for a in r["protected_mean"]:
        rg = r["regions"][a]
        print(f"   {a:9s} protected {r['protected_mean'][a]:.3f} in-region err {r['inregion_median_err'][a]:.2f} "
              f"focus_add {r['focus_add'][a]:.2f} | " + " ".join(f"{n} {rg[n]:.2f}" for n in REG)
              + " | dist " + " ".join(f"{rg[k]:.2f}" for k in rg if k.startswith("dist_")))
print("opt_frz: median ratio %.3f wins %d/8 others %.2f (req %.2f)" % (med, feas_wins, feas_others, req_others))
print("DECISIONS: controls_ok", summary["controls_ok"], "| mechanism_supported", mech, "| preserving_correction_found", feasible)
