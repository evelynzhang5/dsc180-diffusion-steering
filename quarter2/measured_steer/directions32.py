"""Round 32 candidate steering directions, from round 30's saved activations (development cells only).

For every development free cell of round 30 (72 generations x 6 cells, sources grass/brick/camera/coffee/
immunohistochemistry/page; none of round 32's test cases), two measured labels:
  size  = log(delivered radius / target)                       (round 30's label)
  shape = primary error of the output vs an ideal disk blur at the DELIVERED radius (error left after size)
Both labels and the cell-pooled features are centred within their generation (removes scene/target offsets).
Direction at a layer = mean centred feature of cells above the label median minus cells below, unit norm
(a difference of means, not the probe readout w). Positive = towards over-blur (size) / larger shape error.
Control: a fixed-seed Gaussian random unit vector per layer.
usage: python directions32.py <R30_OUT> <out .npz>
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import metrics29 as Q  # noqa: E402
import patches as PT  # noqa: E402

LAYERS = ["mid", "up0"]
r30, dst = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{r30}/meta.json"))
F = np.load(f"{r30}/feats.npz")
lib = PT.library()
gi, cj, size, shape = [], [], [], []
for i, row in enumerate(meta["rows"]):
    if row["split"] != "dev" or row["tag"].startswith("gate"):
        continue
    grp = [p for p in lib if p["split"] == "dev"][row["group"]::2]
    img, cid, where = PT.layout(grp, row["rot"])
    im = img.astype(np.float32)
    y = np.asarray(Image.open(f"{r30}/img/{row['tag']}.png")).astype(np.float32)
    st = m18.stack(im, "disk")
    for j, k in enumerate(row["cells"][:6]):
        m = D.interior(cid, k)
        r = float(M.fit_fast(y[m], M.stack_at(st, m))["v"])
        ideal = np.clip(m18.blur(im, m18.disk(r)), 0, 255)
        gi.append(i), cj.append(j)
        size.append(np.log(max(r, 0.5) / row["target"]))
        shape.append(Q.aff_mae(y, ideal, m))
gi, cj = np.array(gi), np.array(cj)
cen = lambda v: v - np.array([v[gi == g].mean(0) for g in gi])
labels = dict(size=cen(np.array(size)), shape=cen(np.array(shape)))
out, info = {}, dict(n_cells=len(gi), n_generations=len(set(gi.tolist())))
rng = np.random.default_rng(32)
for L in LAYERS:
    X = cen(F[L][gi, cj].astype(np.float64))
    norm_tok = float(np.linalg.norm(F[L][gi, cj], axis=1).mean())
    for name, lab in labels.items():
        hi = lab > np.median(lab)
        v = X[hi].mean(0) - X[~hi].mean(0)
        info[f"{L}_{name}"] = dict(raw_norm=float(np.linalg.norm(v)), pooled_feature_norm=norm_tok,
                                   proj_spearman=float(np.corrcoef(np.argsort(np.argsort(X @ v)), np.argsort(np.argsort(lab)))[0, 1]))
        out[f"{L}_{name}"] = (v / np.linalg.norm(v)).astype(np.float32)
    r = rng.standard_normal(X.shape[1])
    out[f"{L}_random"] = (r / np.linalg.norm(r)).astype(np.float32)
    info[f"{L}_cos_size_shape"] = float(out[f"{L}_size"] @ out[f"{L}_shape"])
info["label_corr_size_shape"] = float(np.corrcoef(labels["size"], labels["shape"])[0, 1])
np.savez(dst, **out)
json.dump(info, open(dst.replace(".npz", ".json"), "w"), indent=1)
print(json.dumps(info, indent=1))
