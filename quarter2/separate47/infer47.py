"""Round 47 (GPU): apply both separators. FROZEN before training.
T1 = round 46 composites (non-control), T2 = round 47 held-out-family composites, T3 = RealBokeh rb_48 (round 43 data;
pane region = round 43's BiRefNet mask), C1 = no-glass control: round 46's background-only images with the pane mask.
Saves, per item and variant, alpha_hat / F_hat / B_hat (linear light, float16 .npy) and B_hat as an sRGB png (the
BokehDiff input). oracle on T3 is skipped (true alpha unknown).
usage: python infer47.py <full.pt> <oracle.pt> <R46 data> <R47 T2 data> <R43 data> <out dir>   env: SMOKE
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth47 as S  # noqa: E402
import train47 as T  # noqa: E402

FULL, ORACLE, D46, D47, D43, OUT = sys.argv[1:7]
SMOKE = os.environ.get("SMOKE", "0") == "1"
models = {}
for v, p in [("full", FULL), ("oracle", ORACLE)]:
    ck = torch.load(p, map_location="cuda")
    m = T.make_model(v).cuda().eval()
    m.load_state_dict(ck["state"])
    models[v] = m
os.makedirs(f"{OUT}", exist_ok=True)
items = []
m46, m47 = json.load(open(f"{D46}/compose46.json")), json.load(open(f"{D47}/compose47.json"))
pane = np.load(f"{D46}/maps/pane.npy")
for c, m in m46["composites"].items():
    if not m["control"]:
        items.append(("T1", c, f"{D46}/input/{c}.png", pane, np.load(f"{D46}/alpha/{c}.npy")))
for c in m47["composites"]:
    items.append(("T2", c, f"{D47}/input/{c}.png", pane, np.load(f"{D47}/alpha/{c}.npy")))
for bg in m46["bgs"]:
    items.append(("C1", bg, f"{D46}/input/{bg}__bgonly.png", pane, np.zeros_like(pane)))
x = Image.open(f"{D43}/input/rb_48.png")
m43 = np.asarray(Image.open(f"{D43}/mask/rb_48.png").convert("L").resize(x.size)) / 255.0
items.append(("T3", "rb_48", f"{D43}/input/rb_48.png", (m43 >= 0.5).astype(np.float32), None))
if SMOKE:
    items = [it for k, it in enumerate(items) if it[0] in ("T3",) or k in (0, len(m46["composites"]) - 1)]
rows = []
with torch.no_grad():
    for setn, name, path, pn, alpha in items:
        I = np.asarray(Image.open(path).convert("RGB")) / 255.0
        t = lambda a, c=None: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).cuda()
        Ii = t(I).permute(2, 0, 1)[None]
        Pi = t(pn)[None, None]
        for v in (["full"] if alpha is None else ["full", "oracle"]):
            Ai = t(alpha)[None, None] if alpha is not None else None
            ah, Fh, Bh = T.predict(models[v], v, Ii, Pi, Ai)
            tag = f"{setn}/{v}/{name}"
            os.makedirs(f"{OUT}/{setn}/{v}", exist_ok=True)
            for k, arr in [("alpha", ah[0, 0]), ("F", Fh[0].permute(1, 2, 0)), ("B", Bh[0].permute(1, 2, 0))]:
                np.save(f"{OUT}/{tag}_{k}.npy", arr.float().cpu().numpy().astype(np.float16))
            Image.fromarray((S.to_srgb(Bh[0].permute(1, 2, 0).cpu().numpy()) * 255).round().astype(np.uint8)) \
                .save(f"{OUT}/{tag}_B.png")
            rows.append(dict(set=setn, name=name, variant=v, tag=tag))
            print(tag, f"mean alpha_hat in pane {float(ah[0, 0][Pi[0, 0] > 0.5].mean()):.3f}", flush=True)
json.dump(dict(rows=rows), open(f"{OUT}/infer47.json", "w"), indent=1)
