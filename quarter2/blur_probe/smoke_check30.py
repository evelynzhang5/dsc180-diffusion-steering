"""Round 30 smoke check (on the pod): feature file layout, and delivered radius of each saved generation."""
import json, sys, os
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D, measure18 as m18, measure19 as M, patches as PT

out = sys.argv[1]
meta = json.load(open(f"{out}/meta.json"))
f = np.load(f"{out}/feats.npz")
print("gate", meta["gate"], "sec/gen", round(meta["sec_per_gen"], 2))
for n in meta["layers"]:
    a = f[n]
    print(f"{n:10s} shape {meta['shapes'][n]} pooled {a.shape} finite {np.isfinite(a).all()} "
          f"cell-spread {a[:, :6].std(1).mean():.4f}")
lib = PT.library()
for row in meta["rows"]:
    grp = [p for p in lib if p["split"] == row["split"]][row["group"]::2]
    img, cid, where = PT.layout(grp, row["rot"])
    y = np.asarray(Image.open(f"{out}/img/{row['tag']}.png")).astype(np.float32)
    st = m18.stack(img.astype(np.float32), "disk")
    r = {where[k]: round(M.fit_fast(y[D.interior(cid, k)], M.stack_at(st, D.interior(cid, k)))["v"], 2) for k in PT.FREE}
    print(row["tag"], "target", row["target"], "request", round(row["request"], 2), r)
