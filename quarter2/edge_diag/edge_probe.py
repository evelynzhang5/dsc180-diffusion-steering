"""Round 38 D1 (CPU): is disparity readable at boundaries, where occlusion acts, rather than in interiors?

Data: round 36's saved up_blocks.0 activations (1280 x 32 x 32) of the unedited calibrated-request outputs,
12 development + 12 test scenes (scenes36). Tokens are 16 x 16 px.
Token types
  interior: inside a free cell's interior mask (round 36's definition)
  boundary: exactly two cells cover the token, each >= 25 % of its pixels (no corners); side A = the left/top cell.
Labels (a boundary token mixes two depths, so no single depth label is assigned):
  disparity of each side (d_A, d_B); difference d_A - d_B; closer side (A if d_A > d_B; no ties occur);
  controls: requested blur (CoC map value) on each side, c_A - c_B.
Probes: standardised ridge, lambda by leave-one-scene-out CV on development scenes; evaluated on test scenes.
  interior  -> d                       (round 36 repeated with scene groups)
  boundary  -> d_A - d_B, closer side (ridge on +-1, sign), d_A, d_B
  boundary  -> c_A - c_B               (control: requested blur difference)
Blur-only baseline for the closer side: logistic regression on (c_A, c_B, c_A c_B, c_A^2, c_B^2), dev-fitted.
Evaluation unit for the closer side: an edge (scene, cell pair), by the mean token score; 144 test edges, of which 12
have the same requested blur on both sides (the blur-only baseline is at chance there by construction).
Interpretation (fixed before running):
  DEPTH READABLE AT BOUNDARIES if, on test edges, (i) the probe's closer-side accuracy exceeds the blur-only baseline's
  by >= 10 points, and (ii) the same-blur edges are ordered correctly in >= 11/12 (one-sided binomial p = 0.003).
  The same-blur subset is small (12 edges); (i) is the better-powered comparison.
  Readable depth would show where depth information is; demonstrating useful steering is a separate step.
usage: python edge_probe.py <R36_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402
import scenes36 as SC  # noqa: E402

R36, RES = sys.argv[1], sys.argv[2]
os.makedirs(RES, exist_ok=True)
T = 16
G = D.S // T
LAMBDAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
c25 = json.load(open(os.path.join(os.path.dirname(R36), "q2_pisa_gate", "calib.json")))["C1"]["G0"]
curve = M.Curve.__new__(M.Curve)
curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])


def pool(mask):
    return mask.reshape(G, T, G, T).mean((1, 3))


def scene_tokens(sc):
    f = np.load(f"{R36}/feats/{sc['name']}.npy").astype(np.float32)          # C x G x G
    cid = sc["cid"]
    coc_free = {k: curve.inverse(SC.rstar(g))[0] / (D.NOM * 10) for k, g in sc["gaps"].items()}
    disp_map, coc_map = SC.maps(sc, coc_free)
    dcell = {k: float(disp_map[cid == k][0]) for k in range(9)}
    ccell = {k: float(coc_map[cid == k][0]) for k in range(9)}
    frac = np.stack([pool((cid == k).astype(np.float32)) for k in range(9)])  # 9 x G x G
    inter, bnd = [], []
    for k in PT.FREE:
        t = pool(D.interior(cid, k).astype(np.float32)) > 0.999
        for i, j in zip(*np.nonzero(t)):
            inter.append(dict(x=f[:, i, j], d=dcell[k], c=ccell[k], scene=sc["name"], cell=k))
    for i in range(G):
        for j in range(G):
            cov = [k for k in range(9) if frac[k, i, j] > 1e-6]
            if len(cov) == 2 and min(frac[cov[0], i, j], frac[cov[1], i, j]) >= 0.25:
                a, b = sorted(cov)                                            # a = left/top (smaller id)
                bnd.append(dict(x=f[:, i, j], dA=dcell[a], dB=dcell[b], cA=ccell[a], cB=ccell[b],
                                scene=sc["name"], edge=f"{sc['name']}/{a}-{b}", fa=float(frac[a, i, j])))
    return inter, bnd


def ridge(X, y, groups):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    cv = {}
    for lam in LAMBDAS:
        e = []
        for g in sorted(set(groups)):
            tr, te = groups != g, groups == g
            ym = y[tr].mean()
            al = np.linalg.solve(Z[tr] @ Z[tr].T + lam * np.eye(tr.sum()), y[tr] - ym)
            e.append(((Z[te] @ (Z[tr].T @ al) + ym - y[te]) ** 2).mean())
        cv[lam] = float(np.mean(e))
    lam = min(cv, key=cv.get)
    ym = y.mean()
    al = np.linalg.solve(Z @ Z.T + lam * np.eye(len(Z)), y - ym)
    w = (Z.T @ al) / sd
    return (lambda Xn: Xn @ w + ym - mu @ w), lam, cv


def collect(split):
    I, B = [], []
    for sc in SC.plan(split):
        i, b = scene_tokens(sc)
        I += i
        B += b
    return I, B


Id, Bd = collect("dev")
It, Bt = collect("test")
arr = lambda L, k: np.array([r[k] for r in L])
X = lambda L: np.stack([r["x"] for r in L])
out = dict(n_tokens=dict(dev_interior=len(Id), dev_boundary=len(Bd), test_interior=len(It), test_boundary=len(Bt)))


def r_mae(p, y):
    return dict(r=float(np.corrcoef(p, y)[0, 1]), mae=float(np.abs(p - y).mean()),
                r2=float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


# interior (round 36 repeated, scene groups)
f, lam, cv = ridge(X(Id), arr(Id, "d"), arr(Id, "scene"))
out["interior_disp"] = dict(lam=lam, cv_mse=cv[lam], label_var=float(arr(Id, "d").var()), **r_mae(f(X(It)), arr(It, "d")))
f, lam, cv = ridge(X(Id), arr(Id, "c"), arr(Id, "scene"))
out["interior_blur_control"] = dict(lam=lam, **r_mae(f(X(It)), arr(It, "c")))

# boundary probes
gd = arr(Bd, "scene")
targets = {"diff_disp": lambda L: arr(L, "dA") - arr(L, "dB"), "dA": lambda L: arr(L, "dA"), "dB": lambda L: arr(L, "dB"),
           "diff_blur_control": lambda L: arr(L, "cA") - arr(L, "cB"),
           "closer": lambda L: np.where(arr(L, "dA") > arr(L, "dB"), 1.0, -1.0)}
pred = {}
for n, fn in targets.items():
    f, lam, cv = ridge(X(Bd), fn(Bd), gd)
    pred[n] = f(X(Bt))
    out[f"boundary_{n}"] = dict(lam=lam, cv_mse=cv[lam], label_var=float(fn(Bd).var()), **r_mae(pred[n], fn(Bt)))

# closer side per edge: probe vs blur-only baseline
feat = lambda L: np.stack([arr(L, "cA"), arr(L, "cB"), arr(L, "cA") * arr(L, "cB"), arr(L, "cA") ** 2, arr(L, "cB") ** 2], 1)
ed = {}
for r, s in zip(Bt, pred["closer"]):
    ed.setdefault(r["edge"], dict(score=[], dA=r["dA"], dB=r["dB"], cA=r["cA"], cB=r["cB"]))["score"].append(s)
yd = (arr(Bd, "dA") > arr(Bd, "dB")).astype(int)
dev_edges = {}
for r, yy in zip(Bd, yd):
    dev_edges[r["edge"]] = (r["cA"], r["cB"], yy)
E = np.array([[v[0], v[1]] for v in dev_edges.values()])
lr = LogisticRegression(C=100, max_iter=5000).fit(
    np.stack([E[:, 0], E[:, 1], E[:, 0] * E[:, 1], E[:, 0] ** 2, E[:, 1] ** 2], 1), [v[2] for v in dev_edges.values()])
rows = []
for k, v in ed.items():
    truth = v["dA"] > v["dB"]
    b = lr.predict(np.array([[v["cA"], v["cB"], v["cA"] * v["cB"], v["cA"] ** 2, v["cB"] ** 2]]))[0] == 1
    rows.append(dict(edge=k, probe_ok=bool((np.mean(v["score"]) > 0) == truth), blur_ok=bool(b == truth),
                     same_blur=abs(v["cA"] - v["cB"]) < 1e-6, n_tokens=len(v["score"]), dA=v["dA"], dB=v["dB"]))
n = len(rows)
pa = np.mean([r["probe_ok"] for r in rows])
ba = np.mean([r["blur_ok"] for r in rows])
sb = [r for r in rows if r["same_blur"]]
k_sb = sum(r["probe_ok"] for r in sb)
out["closer_side_edges"] = dict(n_edges=n, probe_acc=float(pa), blur_only_acc=float(ba),
                                probe_only_right=int(sum(r["probe_ok"] and not r["blur_ok"] for r in rows)),
                                blur_only_right=int(sum(r["blur_ok"] and not r["probe_ok"] for r in rows)),
                                same_blur_n=len(sb), same_blur_correct=int(k_sb),
                                same_blur_p=float(binom.sf(k_sb - 1, len(sb), 0.5)) if sb else None)
out["depth_readable_at_boundaries"] = bool(pa - ba >= 0.10 and k_sb >= 11)
json.dump(dict(summary=out, edges=rows), open(f"{RES}/edge_probe.json", "w"), indent=1)
for k, v in out.items():
    print(k, v)

# ---- post hoc (added after the first run; reported as exploratory): does the boundary depth-difference prediction
# carry depth beyond the requested blur? Edge-level partial correlation given a quadratic in (c_A, c_B), and the
# same-blur edges listed.
ped = {}
for r, s in zip(Bt, pred["diff_disp"]):
    ped.setdefault(r["edge"], dict(p=[], y=r["dA"] - r["dB"], cA=r["cA"], cB=r["cB"]))["p"].append(s)
P_ = np.array([np.mean(v["p"]) for v in ped.values()])
Y_ = np.array([v["y"] for v in ped.values()])
C_ = np.array([[1, v["cA"], v["cB"], v["cA"] * v["cB"], v["cA"] ** 2, v["cB"] ** 2] for v in ped.values()])
res = lambda z: z - C_ @ np.linalg.lstsq(C_, z, rcond=None)[0]
post = dict(edge_r=float(np.corrcoef(P_, Y_)[0, 1]),
            edge_partial_r_given_blur=float(np.corrcoef(res(P_), res(Y_))[0, 1]),
            blur_explains_true_diff_r2=float(1 - (res(Y_) ** 2).sum() / ((Y_ - Y_.mean()) ** 2).sum()),
            same_blur_edges=[dict(edge=r["edge"], dA=r["dA"], dB=r["dB"], ok=r["probe_ok"]) for r in sb])
print("post hoc", post)
d = json.load(open(f"{RES}/edge_probe.json"))
d["post_hoc"] = post
json.dump(d, open(f"{RES}/edge_probe.json", "w"), indent=1)

# ---- post hoc: position-only baseline for the closer side (dev majority per cell pair) and probe + blur combined
maj = {}
for k, (cA, cB, yy) in dev_edges.items():
    maj.setdefault(k.split("/")[1], []).append(yy)
pos_ok = [(np.mean(maj[r["edge"].split("/")[1]]) > 0.5) == (r["dA"] > r["dB"]) for r in rows]
post2 = dict(position_only_acc=float(np.mean(pos_ok)),
             test_base_rate_A_closer=float(np.mean([r["dA"] > r["dB"] for r in rows])),
             same_blur_distinct_configs=sorted({(r["edge"].split("/")[1], round(r["dA"], 2), round(r["dB"], 2)) for r in sb}),
             same_blur_position_only_correct=int(sum((np.mean(maj[r["edge"].split("/")[1]]) > 0.5) == (r["dA"] > r["dB"]) for r in sb)))
print("post hoc 2", post2)
d = json.load(open(f"{RES}/edge_probe.json"))
d["post_hoc_position"] = post2
json.dump(d, open(f"{RES}/edge_probe.json", "w"), indent=1, default=str)

# ---- post hoc: partial correlation given blur AND cell-pair position (dummies)
pairs_ = sorted({k.split("/")[1] for k in ped})
C2 = np.concatenate([C_, np.array([[1.0 * (k.split("/")[1] == p) for p in pairs_[1:]] for k in ped])], 1)
res2 = lambda z: z - C2 @ np.linalg.lstsq(C2, z, rcond=None)[0]
free_free = np.array([all(int(c) in PT.FREE for c in k.split("/")[1].split("-")) for k in ped])
post3 = dict(edge_partial_r_given_blur_and_position=float(np.corrcoef(res2(P_), res2(Y_))[0, 1]),
             blur_and_position_explain_true_diff_r2=float(1 - (res2(Y_) ** 2).sum() / ((Y_ - Y_.mean()) ** 2).sum()),
             free_free_edges=int(free_free.sum()),
             free_free_partial_r_given_blur=float(np.corrcoef(res(P_)[free_free], res(Y_)[free_free])[0, 1]))
print("post hoc 3", post3)
d = json.load(open(f"{RES}/edge_probe.json"))
d["post_hoc_blur_position"] = post3
json.dump(d, open(f"{RES}/edge_probe.json", "w"), indent=1, default=str)
