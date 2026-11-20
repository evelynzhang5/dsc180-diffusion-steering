"""Round 36 analysis: three separate claims, each with a rule fixed before the run (test scenes only).

A. The inner product reads depth (probe d_hat = w.h + b at up_blocks.0, fitted on development scenes).
   Cell-level (mean over a cell's interior tokens), 72 test cells:
     A1 tracks depth:        Pearson r(d_hat, d) >= 0.7 and MAE <= 0.10 (disparity units; gap step 0.1)
     A2 not content/position: share of residual variance explained by texture source <= 0.2, and by position <= 0.2
     A3 not the blur request: front/behind pairs with the same requested blur in the same scene (36 pairs):
                             d_hat orders them correctly in >= 29/36 (p < 0.001 under chance)
   READS DEPTH if A1-A3 all hold. Reported: partial r(d_hat, d | requested blur), partial r(d_hat, d | signed gap)
   (i.e. does it see the focus setting), calibration slope, and the |gap| / signed-gap probes' test R^2.
B. The edit controls blur (u = round 32's size direction, w_d = probe weight direction, random; equal norms;
   a in {-2,-1,0,1,2} on test scenes 0-5, 36 cells). Per direction:
     B1 consistent:  the sign of r_hat(+2) - r_hat(-2) agrees in >= 80 % of cells
     B2 sizeable:    median |r_hat(+2) - r_hat(-2)| >= 1 px and >= 2 x the random direction's
     B3 preserving:  focus + anchor pixel change vs the unedited output <= 0.5/255 at |a| = 2 (every scene)
   CONTROLS BLUR if B1-B3 hold. (The probe's own output is not used: it moves along w_d by construction.)
C. Adapting the strength adds value (72 test cells; ratio = revised error / calibrated request's; invalid or failing
   the contrast or preservation guard -> inf; calibrated invalid and method valid -> 0):
     methods: fixed a (dev-chosen), depth-map g(gap), proxy g(d_hat - d_f), shuffled proxy; oracle per-cell a (ceiling)
     GATE (smoke-test level, as rounds 29-35): median ratio <= 0.80 and wins >= 54/72 (75 %)
   ADAPTATION HELPS if the proxy passes the gate AND beats the fixed strength (wins > 36/72 head-to-head and
   median ratio below fixed's) AND beats the shuffled proxy the same way. A pass needs fresh-image confirmation.
usage: python analyze36.py <R36_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure18 as m18  # noqa: E402

PRESERVE_MAX, CON_MARGIN = 0.5, 0.05
out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
frozen = json.load(open(f"{out}/frozen36.json"))
P = np.load(f"{out}/probe36_test.npz")
rows = meta["rows"]
R = {}


# ---------------------------------------------------------------- A
def cellmean(pred):
    keys = sorted(set(zip(P["scene"], P["cell"])))
    out_ = []
    for s, k in keys:
        m = (P["scene"] == s) & (P["cell"] == k)
        out_.append(dict(scene=str(s), cell=int(k), pred=float(pred[m].mean()), d=float(P["disp"][m][0]),
                         gap=float(P["gap"][m][0]), abs_gap=float(P["abs_gap"][m][0]), src=str(P["src"][m][0]),
                         d_f=float(P["d_f"][m][0])))
    return out_


def eta2(resid, groups):
    resid, groups = np.asarray(resid), np.asarray(groups)
    tot = ((resid - resid.mean()) ** 2).sum()
    between = sum((groups == g).sum() * (resid[groups == g].mean() - resid.mean()) ** 2 for g in set(groups))
    return float(between / max(tot, 1e-12))


def partial(x, y, groups):
    groups = np.asarray(groups)
    rx, ry = np.array(x, float), np.array(y, float)
    for g in set(groups):
        m = groups == g
        rx[m] -= rx[m].mean()
        ry[m] -= ry[m].mean()
    return float(np.corrcoef(rx, ry)[0, 1])


cm = cellmean(P["pred_disp"])
pred, d = np.array([c["pred"] for c in cm]), np.array([c["d"] for c in cm])
resid = pred - d
pairs, correct = 0, 0
for s in sorted({c["scene"] for c in cm}):
    for g in sorted({c["abs_gap"] for c in cm}):
        pr = [c for c in cm if c["scene"] == s and abs(c["abs_gap"] - g) < 1e-9]
        if len(pr) == 2:
            fr, be = (pr[0], pr[1]) if pr[0]["gap"] > 0 else (pr[1], pr[0])
            pairs += 1
            correct += fr["pred"] > be["pred"]
r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
A = dict(r=float(np.corrcoef(pred, d)[0, 1]), mae=float(np.abs(resid).mean()), r2=r2(pred, d),
         slope=float(np.polyfit(d, pred, 1)[0]), eta2_source=eta2(resid, [c["src"] for c in cm]),
         eta2_position=eta2(resid, [c["cell"] for c in cm]), pairs_correct=int(correct), pairs=int(pairs),
         partial_r_given_blur=partial(pred, d, [c["abs_gap"] for c in cm]),
         partial_r_given_signed_gap=partial(pred, d, [c["gap"] for c in cm]),
         token_r=float(np.corrcoef(P["pred_disp"], P["disp"])[0, 1]),
         token_mae=float(np.abs(P["pred_disp"] - P["disp"]).mean()),
         dev_r=float(np.corrcoef(P["dev_pred_disp"], P["dev_disp"])[0, 1]),
         abs_gap_probe_r2=r2(np.array([c["pred"] for c in cellmean(P["pred_abs_gap"])]), np.array([c["abs_gap"] for c in cm])),
         gap_probe_r2=r2(np.array([c["pred"] for c in cellmean(P["pred_gap"])]), np.array([c["gap"] for c in cm])),
         lambdas={n: p["lam"] for n, p in frozen["probes"].items()})
A.update(A1=A["r"] >= 0.7 and A["mae"] <= 0.10, A2=A["eta2_source"] <= 0.2 and A["eta2_position"] <= 0.2,
         A3=A["pairs_correct"] >= 29)
A["reads_depth"] = bool(A["A1"] and A["A2"] and A["A3"])
R["A"] = A
R["A_cells"] = cm
d0 = {k: v for k, v in meta["gate"].items() if k.startswith("D0_")}
R["D0_disparity_flip"] = {s: dict(pix_median=float(np.median([c["pix"] for c in v.values()])),
                                  feat_rel_median=float(np.median([c["feat_rel"] for c in v.values()])),
                                  r_hat_change_max=float(max(abs(c["r_hat_flip"] - c["r_hat0"]) for c in v.values())),
                                  probe_shift=[round(c["probe_shift"], 3) for c in v.values()]) for s, v in d0.items()}

# ---------------------------------------------------------------- B
test = [r for r in rows if r["split"] == "test"]
s3_scenes = sorted({r["scene"] for r in test if r["kind"] == "step3"})


def get(scene, kind, a, dir_):
    if a == 0:
        kind, dir_ = "grid", "size"
    return next((r for r in test if r["scene"] == scene and r["kind"] == kind and r.get("a") == a and r.get("dir") == dir_), None)


B = {}
for dir_, kind in (("size", "grid"), ("wd", "step3"), ("random", "step3")):
    rng_, slopes, dcon, pres, derr = [], [], [], [], []
    for s in s3_scenes:
        lo, hi = get(s, kind, -2.0, dir_), get(s, kind, 2.0, dir_)
        z = get(s, kind, 0.0, dir_)
        if lo is None or hi is None:
            continue
        pres += [lo["preserve"], hi["preserve"]]
        for k in hi["cells"]:
            rng_.append(hi["cells"][k]["r_hat"] - lo["cells"][k]["r_hat"])
            dcon.append(max(abs(hi["cells"][k]["contrast"] - z["cells"][k]["contrast"]),
                            abs(lo["cells"][k]["contrast"] - z["cells"][k]["contrast"])))
            derr.append(min(hi["cells"][k]["err"], lo["cells"][k]["err"]) - z["cells"][k]["err"])
    rng_ = np.array(rng_)
    B[dir_] = dict(n_cells=len(rng_), sign_consistency=float(max((rng_ > 0).mean(), (rng_ < 0).mean())) if len(rng_) else 0,
                   median_abs_range=float(np.median(np.abs(rng_))) if len(rng_) else 0,
                   median_range=float(np.median(rng_)) if len(rng_) else 0,
                   preserve_max_at_2=float(max(pres)) if pres else None, contrast_change_median=float(np.median(dcon)) if dcon else None,
                   err_change_best_side_median=float(np.median(derr)) if derr else None)
for dir_ in B:
    b = B[dir_]
    b.update(B1=b["sign_consistency"] >= 0.8,
             B2=b["median_abs_range"] >= 1.0 and b["median_abs_range"] >= 2 * B["random"]["median_abs_range"],
             B3=b["preserve_max_at_2"] is not None and b["preserve_max_at_2"] <= PRESERVE_MAX)
    b["controls_blur"] = bool(b["B1"] and b["B2"] and b["B3"]) if dir_ != "random" else None
R["B"] = B


# ---------------------------------------------------------------- C
def valid(d, p):
    return abs(d["r_hat"] - d["r_star"]) <= m18.TOL(d["r_star"]) and p <= PRESERVE_MAX


def ratio(r, r0, k):
    d, d0 = r["cells"][k], r0["cells"][k]
    if not (valid(d, r["preserve"]) and d["contrast"] >= d0["contrast"] - CON_MARGIN):
        return float("inf")
    return d["err"] / d0["err"] if valid(d0, 0.0) else 0.0


a_fix = frozen["rule"]["a_fixed"]
grid = meta["grid"]
ratios = {m: [] for m in ["fixed", "depth", "proxy", "shuffled", "oracle"]}
base_valid, keys = 0, []
for s in sorted({r["scene"] for r in test}):
    r0 = get(s, "grid", 0.0, "size")
    rf = get(s, "grid", a_fix, "size") if a_fix != 0 else r0
    meth = {r["method"]: r for r in test if r["scene"] == s and r["kind"] == "method"}
    gr = [get(s, "grid", a, "size") for a in grid]
    for k in r0["cells"]:
        keys.append(f"{s}/{k}")
        base_valid += valid(r0["cells"][k], 0.0)
        ratios["fixed"].append(ratio(rf, r0, k) if a_fix != 0 else 1.0)
        for m in ("depth", "proxy", "shuffled"):
            ratios[m].append(ratio(meth[m], r0, k) if m in meth else float("inf"))
        ratios["oracle"].append(min(ratio(g, r0, k) if g is not r0 else 1.0 for g in gr if g is not None))
n = len(keys)
C = dict(n_cells=n, calibrated_valid=int(base_valid), a_fixed=a_fix, rule=frozen["rule"], methods={})
for m, v in ratios.items():
    v = np.array(v)
    C["methods"][m] = dict(median_ratio=float(np.median(v)), wins=int((v < 1).sum()), invalid=int(np.isinf(v).sum()),
                           gate=bool(np.median(v) <= 0.80 and (v < 1).sum() >= int(np.ceil(0.75 * n))))


def h2h(a, b):
    x, y = np.array(ratios[a]), np.array(ratios[b])
    return dict(wins=int((x < y).sum()), losses=int((x > y).sum()), ties=int((x == y).sum()))


C["proxy_vs_fixed"], C["proxy_vs_shuffled"], C["proxy_vs_depth"], C["depth_vs_fixed"] = (
    h2h("proxy", "fixed"), h2h("proxy", "shuffled"), h2h("proxy", "depth"), h2h("depth", "fixed"))
pm = C["methods"]
C["adaptation_helps"] = bool(pm["proxy"]["gate"]
                             and C["proxy_vs_fixed"]["wins"] > n / 2 and pm["proxy"]["median_ratio"] < pm["fixed"]["median_ratio"]
                             and C["proxy_vs_shuffled"]["wins"] > n / 2 and pm["proxy"]["median_ratio"] < pm["shuffled"]["median_ratio"])
C["cells"] = {k: {m: ratios[m][i] for m in ratios} for i, k in enumerate(keys)}
R["C"] = C
R["gate"] = {k: v for k, v in meta["gate"].items() if not k.startswith("D0_")}
R["n_generations"], R["minutes"] = meta.get("n_generations"), meta.get("minutes")
json.dump(R, open(f"{res}/depth_probe.json", "w"), indent=1, default=float)

print("gate", R["gate"])
print("A probe:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in A.items() if k != "lambdas"})
print("D0 disparity flip:", R["D0_disparity_flip"])
for dname, b in B.items():
    print("B", dname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})
print("C calibrated valid", base_valid, "/", n, "| a_fixed", a_fix, "| rule", frozen["rule"]["knots_x"], frozen["rule"]["knots_y"])
for m, v in pm.items():
    print(f"  {m:9s} median {v['median_ratio']:.3f} wins {v['wins']}/{n} invalid {v['invalid']} gate {v['gate']}")
print("  proxy vs fixed", C["proxy_vs_fixed"], "| vs shuffled", C["proxy_vs_shuffled"], "| vs depth", C["proxy_vs_depth"],
      "| depth vs fixed", C["depth_vs_fixed"])
print("DECISIONS: reads_depth", A["reads_depth"], "| controls_blur", {k: b["controls_blur"] for k, b in B.items() if k != "random"},
      "| adaptation_helps", C["adaptation_helps"])
