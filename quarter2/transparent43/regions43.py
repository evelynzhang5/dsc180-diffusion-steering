"""Round 43 stage 1 (CPU, before any method runs): measurement regions, eligibility and the K plan.

Per image (768-px scale; BokehDiff's own Depth Anything V2 disparity, larger = farther; BiRefNet subject mask):
  behind tiles   round 40 regions40.py: 64x64 (stride 32), entirely >= 24 px outside the subject, >= 90% of
                 pixels >= 0.08 from the focus disparity (round 40 used 0.15; lowered before any round 43 image existed
                 because in the strongest real example, rb_48, the shelves behind the sign are only 0.11-0.15 from the
                 sign and round 40 found no tile there), mean gradient >= 4/255, median depth behind the focus, up to
                 8 non-overlapping, most textured first. Reference blur of the real background.
  see candidates 64x64 (stride 16) entirely INSIDE the subject at >= 6 px from its edge, mean gradient >= 4/255, up to 12
                 non-overlapping, most textured first, numbered on the sheets. Texture alone also picks the object's
                 own features (logos, water, facets), so the SEE TILES are chosen from these candidates by eye on the
                 INPUT sheets, before any output is viewed (review43.json; criterion in the README).
  rim band       subject pixels within 6 px of the subject edge: the object's own outline, which should stay sharp.
  focus          median disparity inside the subject (as BokehDiff's preprocessing).
  depth_sees_through   median |disparity - focus| over the see candidates >= 0.15 (round 40's gap): the depth model already places the
                 through-content far (then the original depth is not the cause in that scene; reported).
Rendered (frozen): subject covers 1-70% of the image, >= 2 behind tiles and >= 2 see candidates.
K plan: round 42's frozen depth rule (rule42.json, offset form) for 8 and 12 px of background blur, from this scene's
depth gap (fit42.gap definition), clipped to [4, 64]; plus the sweep K = 36 and 64 for the original depth.
Contact sheets: input with subject outline (red), behind tiles (cyan), see tiles (yellow); disparity; f/2.0 capture.
usage: python regions43.py <data root>   -> <root>/regions43.json, <root>/sheets/*.png
"""
import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as sw
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1]
RULE = json.load(open(os.path.join(HERE, "..", "control42", "results", "rule42.json")))
os.makedirs(f"{ROOT}/sheets", exist_ok=True)
TILE, STRIDE, MAXT, GAP, TEX, BG_MARGIN = 64, 32, 8, 0.08, 4.0, 24      # regions40 but GAP 0.15 -> 0.08 (see doc)
S_STRIDE, S_MAX, S_MARGIN, RIM = 16, 12, 6, 6
TARGETS, SWEEP = [8.0, 12.0], [36.0, 64.0]


def _mm(d, fn):
    return fn(sw(np.pad(d, 1, mode="edge"), (3, 3)), axis=(-2, -1))


def gap(disp_raw, mask, H, W, tiles):                    # = fit42.gap (BokehDiff preprocessing, behind tiles)
    d = disp_raw.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = _mm(_mm(d, np.min), np.max)
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((d.shape[1], d.shape[0]),
                                                                       resample=Image.LANCZOS)) / 255.0
    f = np.median(d[m > 0.5])
    d = np.asarray(Image.fromarray(d).resize((W, H), resample=Image.BILINEAR))
    return float(np.median([np.median(np.abs(d[t["y"]:t["y"] + 64, t["x"]:t["x"] + 64] - f)) for t in tiles]))


def plan_K(t, g):
    b, c = RULE["params"]["b"], RULE["params"]["c"]
    assert RULE["form"] == "L"
    k = (t / b + c) / g
    return float(np.clip(k, RULE["k_min"], RULE["k_max"])), float(k)


ids = sorted(f[:-4] for f in os.listdir(f"{ROOT}/input") if f.endswith(".png"))
out, panels = {}, []
for i in ids:
    img = np.asarray(Image.open(f"{ROOT}/input/{i}.png").convert("RGB")).astype(np.float64)
    H, W = img.shape[:2]
    mask = np.asarray(Image.open(f"{ROOT}/mask/{i}.png").convert("L").resize((W, H))).astype(np.float64) / 255
    raw = np.load(f"{ROOT}/depth/{i}_pred.npy").astype(np.float64)
    disp = raw if raw.shape == (H, W) else np.asarray(
        Image.fromarray(raw.astype(np.float32)).resize((W, H), resample=Image.BILINEAR)).astype(np.float64)
    subj = mask >= 0.5
    din, dout = distance_transform_edt(subj), distance_transform_edt(~subj)
    f_disp = float(np.median(disp[subj])) if subj.any() else float("nan")
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    cands = []
    for y in range(0, H - TILE + 1, STRIDE):
        for x in range(0, W - TILE + 1, STRIDE):
            if dout[y:y + TILE, x:x + TILE].min() < BG_MARGIN:
                continue
            g = float(np.percentile(np.abs(disp[y:y + TILE, x:x + TILE] - f_disp), 10))
            t = float(grad[y:y + TILE, x:x + TILE].mean())
            if g >= GAP and t >= TEX and np.median(disp[y:y + TILE, x:x + TILE]) > f_disp:
                cands.append((t, y, x, g))
    cands.sort(reverse=True)
    tiles = []
    for t, y, x, g in cands:
        if len(tiles) < MAXT and all(abs(y - yy) >= TILE or abs(x - xx) >= TILE for _, yy, xx, _ in tiles):
            tiles.append((t, y, x, g))
    sc = []
    for y in range(0, H - TILE + 1, S_STRIDE):
        for x in range(0, W - TILE + 1, S_STRIDE):
            if din[y:y + TILE, x:x + TILE].min() >= S_MARGIN:
                t = float(grad[y:y + TILE, x:x + TILE].mean())
                if t >= TEX:
                    sc.append((t, y, x))
    sc.sort(reverse=True)
    see = []
    for t, y, x in sc:
        if len(see) < S_MAX and all(abs(y - yy) >= TILE or abs(x - xx) >= TILE for _, yy, xx in see):
            see.append((t, y, x))
    see_gap = float(np.median([np.median(np.abs(disp[y:y + TILE, x:x + TILE] - f_disp)) for _, y, x in see])) \
        if see else float("nan")
    frac = float(subj.mean())
    eligible = bool(0.01 <= frac <= 0.70 and len(tiles) >= 2 and len(see) >= 2)        # = rendered
    tl = [dict(y=y, x=x, texture=t, disp_gap=g) for t, y, x, g in tiles]
    rec = dict(size=[H, W], subject_frac=frac, focus_disp=f_disp, tiles=tl,
               see_candidates=[dict(k=k, y=y, x=x, texture=t) for k, (t, y, x) in enumerate(see)], rim_px=int(((din > 0) & (din <= RIM)).sum()),
               see_depth_gap=see_gap, depth_sees_through=bool(see_gap >= 0.15), eligible=eligible)
    if eligible:
        g = gap(raw, mask if raw.shape == (H, W) else np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                                                                  .resize(raw.shape[::-1])) / 255.0, H, W, tl)
        rec["gap"] = g
        rec["K"] = {f"{t:g}": plan_K(t, g)[0] for t in TARGETS}
        rec["K_unclipped"] = {f"{t:g}": plan_K(t, g)[1] for t in TARGETS}
        rec["K"].update({f"K{k:g}": k for k in SWEEP})
    out[i] = rec
    vis = img.astype(np.uint8).copy()
    vis[subj & (din <= 2)] = [255, 0, 0]
    vis = Image.fromarray(vis)
    d = ImageDraw.Draw(vis)
    for _, y, x, _ in tiles:
        d.rectangle([x, y, x + TILE, y + TILE], outline=(0, 255, 255), width=2)
    for k, (_, y, x) in enumerate(see):
        d.rectangle([x, y, x + TILE, y + TILE], outline=(255, 255, 0), width=2)
        d.rectangle([x + 2, y + 2, x + 20, y + 16], fill=(0, 0, 0))
        d.text((x + 4, y + 3), str(k), fill=(255, 255, 0))
    dv = (disp - disp.min()) / max(np.ptp(disp), 1e-9)
    row = [vis, Image.fromarray((dv * 255).astype(np.uint8)).convert("RGB")]
    tgt = f"{ROOT}/targets/{i}_f2.0.png"
    if os.path.exists(tgt):
        row.append(Image.open(tgt).convert("RGB"))
    big = vis.copy()
    s = 300 / max(H, W)
    row = [r.resize((int(W * s), int(H * s))) for r in row[1:]]
    p = Image.new("RGB", (W + 2 * 306, H + 22), (255, 255, 255))
    p.paste(big, (0, 22))
    for k, r in enumerate(row):
        p.paste(r, (W + 6 + k * 306, 22))
    ImageDraw.Draw(p).text((4, 4), f"{i}  subject {frac:.2f}  behind {len(tiles)}  candidates {len(see)}  see-depth gap "
                                   f"{see_gap:.2f}  eligible {eligible}", fill=(0, 0, 0))
    panels.append(p)
    print(i, f"subject {frac:.2f} behind {len(tiles)} see {len(see)} see_gap {see_gap:.2f} eligible {eligible}",
          rec.get("K", ""), flush=True)
for k in range(0, len(panels), 2):
    ps = panels[k:k + 2]
    sheet = Image.new("RGB", (max(p.width for p in ps), sum(p.height + 6 for p in ps)), (255, 255, 255))
    yy = 0
    for p in ps:
        sheet.paste(p, (0, yy))
        yy += p.height + 6
    sheet.save(f"{ROOT}/sheets/sheet_{k // 2:02d}.png")
json.dump(out, open(f"{ROOT}/regions43.json", "w"), indent=1)
print("regions:", len(out), "images;", sum(v["eligible"] for v in out.values()), "eligible;",
      sum(v["depth_sees_through"] for v in out.values()), "where depth already sees through", flush=True)
