"""Round 39 step 1 analysis (decision rule frozen before generation).

Per cell (setting x level), per scene: 'adjusted' = the lowest-E VALID generation among the global and edge request
adjustments (base included). A scene shows a boundary error if adjusted E >= 2.0/255 and its band error is
>= 1.5x that side's interior error (measure39.qualifies). No valid generation -> the scene counts as not showing one
and is reported as invalid.
  ESTABLISHED (cell)  >= 18/24 scenes show a boundary error after adjustment.
  Round outcome       ESTABLISHED in >= 1 cell -> a repeatable boundary error survives request adjustment;
                      otherwise NOT ESTABLISHED for these scenes.
Bokehlicious (setting A only): the same rule with the f-stop as its adjustment; reported as a baseline, not part of
the decision.
usage: python analyze39.py <R39_OUT> <results dir>
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure39 as Q  # noqa: E402

OUT, RES = sys.argv[1], sys.argv[2]
os.makedirs(RES, exist_ok=True)
C = json.load(open(f"{OUT}/conds.json"))
print("checks", C["checks"], "OK" if C["checks_ok"] else "FAILED")
res = dict(checks=C["checks"], checks_ok=C["checks_ok"], cells={})
if not C["checks_ok"]:
    json.dump(res, open(f"{RES}/boundary39.json", "w"), indent=1)
    sys.exit(0)
rows = C["rows"]
PROF_AT = [-24, -16, -12, -8, -4, -2, 0, 2, 4, 8, 12, 16, 24]


def summarise(R, base_fn, label):
    scenes = sorted({r["scene"] for r in R})
    out, n_q, n_inv, bests, bases = {}, 0, 0, [], []
    for s in scenes:
        Rs = [r for r in R if r["scene"] == s]
        v = [r for r in Rs if r["valid"]]
        base = base_fn(Rs)
        best = min(v, key=lambda r: r["E"]) if v else None
        q = bool(best is not None and Q.qualifies(best))
        n_q += q
        n_inv += best is None
        bases.append(base)
        if best is not None:
            bests.append(best)
        out[s] = dict(base_E=base["E"] if base else None, base_valid=base["valid"] if base else None,
                      best_E=best["E"] if best else None, best_band=best["E_band"] if best else None,
                      best_ratio=best["ratio"] if best else None, best_knob=(best.get("mg"), best.get("me"), best.get("av")) if best else None,
                      qualifies=q)
    med = lambda xs: float(np.median(xs)) if xs else None
    prof = {d: med([r["profile"].get(str(d), r["profile"].get(d)) for r in bests]) for d in PROF_AT}
    summ = dict(n=len(scenes), n_qualify=n_q, n_invalid=n_inv, established=n_q >= Q.FRAC_MIN * len(scenes),
                base_E_median=med([b["E"] for b in bases if b]), best_E_median=med([b["E"] for b in bests]),
                best_ratio_median=med([b["ratio"] for b in bests]),
                interior_in_median=med([b["interior"]["in"] for b in bests]),
                interior_out_median=med([b["interior"]["out"] for b in bests]),
                band_mode=Counter(b["E_band"] for b in bests).most_common(2),
                band_medians={n: med([b["band"][n] for b in bests]) for n in bests[0]["band"]} if bests else {},
                profile_median=prof, per_scene=out)
    print(f"{label:22s} qualify {n_q}/{len(scenes)} (invalid {n_inv}) -> {'ESTABLISHED' if summ['established'] else 'no'}; "
          f"E base {summ['base_E_median']} -> adjusted {summ['best_E_median']}; ratio {summ['best_ratio_median']}; "
          f"bands {summ['band_mode']}")
    print("   profile (signed luminance residual by distance, + = inside fg):",
          {d: (round(v, 2) if v is not None else None) for d, v in prof.items()})
    return summ


for st in ("A", "B", "C"):
    for lv in ("L1", "L2"):
        R = [r for r in rows if r["setting"] == st and r["level"] == lv]
        if not R:
            continue
        res["cells"][f"{st}_{lv}"] = summarise(
            R, lambda Rs: next((r for r in Rs if r["kind"] == "global" and abs(r["mg"] - 1.0) < 1e-9), None),
            f"BokehDiff {st} {lv}")
res["established_any"] = any(c["established"] for c in res["cells"].values())
print("ROUND OUTCOME:", "ESTABLISHED" if res["established_any"] else "NOT ESTABLISHED")
if os.path.exists(f"{OUT}/bkl.json"):
    B = json.load(open(f"{OUT}/bkl.json"))
    res["bokehlicious"] = dict(latency=B["latency_s"], cells={})
    for lv in ("L1", "L2"):
        R = [r for r in B["rows"] if r["level"] == lv]
        if R:
            res["bokehlicious"]["cells"][f"A_{lv}"] = summarise(
                R, lambda Rs: min(Rs, key=lambda r: r["E"]), f"Bokehlicious A {lv}")
    print("Bokehlicious latency", B["latency_s"], "| BokehDiff minutes/generation",
          C.get("minutes", 0) / max(C.get("n_generations", 1), 1))
json.dump(res, open(f"{RES}/boundary39.json", "w"), indent=1, default=float)
