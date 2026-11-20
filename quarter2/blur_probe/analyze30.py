"""Round 30 analysis: frozen decision on whether a linear readout of activations predicts content-specific blur error.

Label per free cell: e = log(delivered radius / target), delivered = disk-equivalent radius (measure19.fit_fast).
Content part: e minus the mean e of that generation's 6 free cells (removes the per-scene/target offset that a
global curve already handles). Predictions are centred the same way.
Probe per layer: standardise pooled features (dev statistics), ridge regression on dev cells only.
Ridge strength and the layer are chosen by leave-one-source-out CV on dev (6 sources); the test set is used once,
for the chosen layer. Baselines (same ridge + CV): contrast only (log_std); all three image stats.
Pass: chosen layer's held-out Spearman >= 0.70 and >= contrast baseline + 0.20.
usage: python analyze30.py <R30_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402

ALPHAS = [10.0 ** k for k in range(-2, 6)]
RHO_MIN, MARGIN = 0.70, 0.20
out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/meta.json"))
F = np.load(f"{out}/feats.npz")
lib = PT.library()
stats = {p["id"]: p["stats"] for p in lib}

# measure delivered radius for every free cell (rows[0] is the gate generation; it is a test layout too)
rows = meta["rows"]
lab, keep = [], []
for i, row in enumerate(rows):
    if row["tag"].startswith("gate"):
        continue
    grp = [p for p in lib if p["split"] == row["split"]][row["group"]::2]
    img, cid, where = PT.layout(grp, row["rot"])
    y = np.asarray(Image.open(f"{out}/img/{row['tag']}.png")).astype(np.float32)
    st = m18.stack(img.astype(np.float32), "disk")
    for j, k in enumerate(row["cells"][:6]):
        m = D.interior(cid, k)
        r = M.fit_fast(y[m], M.stack_at(st, m))["v"]
        pid = where[k]
        lab.append(dict(gen=i, j=j, split=row["split"], patch=pid, source=pid.rsplit("_", 1)[0], target=row["target"],
                        rot=row["rot"], delivered=float(r), e=float(np.log(max(r, 0.5) / row["target"]))))
gen = np.array([d["gen"] for d in lab])
e = np.array([d["e"] for d in lab])
ec = e - np.array([e[gen == g].mean() for g in gen])
split = np.array([d["split"] for d in lab])
src = np.array([d["source"] for d in lab])
dev, test = split == "dev", split == "test"


def centre(p):
    return p - np.array([p[gen == g].mean() for g in gen])


def ridge_fit(X, y, a):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - mu) / sd) @ w + y.mean()


def cv_score(X, a):
    """leave-one-source-out on dev: Spearman of centred predictions vs centred labels"""
    pred = np.zeros(len(e))
    for s in np.unique(src[dev]):
        tr, te = dev & (src != s), dev & (src == s)
        pred[te] = ridge_fit(X[tr], ec[tr], a)(X[te])
    pc = centre(np.where(dev, pred, 0.0))
    return spearmanr(pc[dev], ec[dev])[0]


def evaluate(X):
    cvs = {a: cv_score(X, a) for a in ALPHAS}
    a = max(cvs, key=cvs.get)
    p = centre(ridge_fit(X[dev], ec[dev], a)(X))
    rho = spearmanr(p[test], ec[test])[0]
    boot = []                                      # bootstrap over the 6 held-out sources
    rng = np.random.default_rng(0)
    ts = np.unique(src[test])
    for _ in range(2000):
        pick = rng.choice(ts, len(ts))
        idx = np.concatenate([np.where(test & (src == s))[0] for s in pick])
        boot.append(spearmanr(p[idx], ec[idx])[0])
    return dict(alpha=a, cv=float(cvs[a]), test_rho=float(rho), ci=[float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5))]), p


cell_feats = {n: F[n][[d["gen"] for d in lab], [d["j"] for d in lab]] for n in meta["layers"]}
out_layers = {n: evaluate(X)[0] for n, X in cell_feats.items()}
chosen = max(out_layers, key=lambda n: out_layers[n]["cv"])
st1 = np.array([[stats[d["patch"]]["log_std"]] for d in lab])
st3 = np.array([[stats[d["patch"]][k] for k in ("log_std", "log_grad", "hf_ratio")] for d in lab])
base = {"contrast": evaluate(st1)[0], "three_stats": evaluate(st3)[0]}
c = out_layers[chosen]
passed = c["test_rho"] >= RHO_MIN and c["test_rho"] >= base["contrast"]["test_rho"] + MARGIN
summary = dict(decision="PASS" if passed else "FAIL", chosen_layer=chosen, chosen=c, baselines=base,
               all_layers_exploratory=out_layers, n_dev=int(dev.sum()), n_test=int(test.sum()),
               label_sd_centred=dict(dev=float(ec[dev].std()), test=float(ec[test].std())), rule=dict(rho_min=RHO_MIN, margin=MARGIN))
json.dump(summary, open(f"{res}/probe.json", "w"), indent=1)
json.dump(lab, open(f"{res}/labels.json", "w"))
print(json.dumps(summary, indent=1))
