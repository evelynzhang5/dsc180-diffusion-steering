"""Round 40 stage 1: measurement regions, chosen on the ORIGINAL images only (before either method runs), then frozen.

Inputs per image: input/<id>.png, BokehDiff's own preparation (prepare_data.py): depth/<id>_pred.npy (Depth Anything
V2 Base, disparity in [0, 1]) and mask/<id>.png (BiRefNet salient mask).
  subject    mask >= 0.5; focus disparity = median disparity in the subject
  inner      subject pixels >= 8 px inside its contour (preservation region)
  band       pixels within 16 px of the subject contour, either side (boundary region)
  bg tiles   64x64 tiles (stride 32) lying entirely in the background >= 24 px from the subject, with
             |disparity - focus disparity| >= 0.15 for >= 90% of the tile's pixels (10th percentile of the gap;
             revised before any method ran: the first version used the tile median, which admitted tiles up to
             half covered by in-focus structure such as branches at the subject's depth) and texture (mean luminance
             gradient magnitude) >= 4/255; up to 8 non-overlapping tiles, most textured first. These are the only
             places where delivered blur is measured; an image with < 2 tiles has unmeasurable blur.
             Split by side (revised after the mask review, before any method ran; the first version ranked all tiles
             together, so in 11 images most tiles were in front of the subject, e.g. hair_7's fence): BokehDiff's
             depth map is larger = farther, so tiles with median depth > focus are BEHIND the subject (background
             blur: "tiles", the primary blur measure) and tiles with median depth < focus are IN FRONT (foreground
             defocus: "front_tiles", reported separately). Each side gets up to 8 tiles, chosen independently;
             blur_measurable needs >= 2 behind tiles, front_measurable >= 2 front tiles.
Also writes contact sheets (sheets/*.png): input with the subject contour (red) and tiles (cyan), the disparity map,
and for RealBokeh the real f/2.0 capture, for the mask review. No method output exists at this point.
usage: python regions40.py <data root>
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

ROOT = sys.argv[1]
os.makedirs(f"{ROOT}/sheets", exist_ok=True)
TILE, STRIDE, MAXT, GAP, TEX, BG_MARGIN = 64, 32, 8, 0.15, 4.0, 24
ids = sorted(f[:-4] for f in os.listdir(f"{ROOT}/input") if f.endswith(".png"))
out, panels = {}, []
for i in ids:
    img = np.asarray(Image.open(f"{ROOT}/input/{i}.png").convert("RGB")).astype(np.float64)
    H, W = img.shape[:2]
    mask = np.asarray(Image.open(f"{ROOT}/mask/{i}.png").convert("L").resize((W, H))).astype(np.float64) / 255
    disp = np.load(f"{ROOT}/depth/{i}_pred.npy").astype(np.float64)
    if disp.shape != (H, W):
        disp = np.asarray(Image.fromarray(disp.astype(np.float32)).resize((W, H), resample=Image.BILINEAR)).astype(np.float64)
    subj = mask >= 0.5
    din = distance_transform_edt(subj)
    dout = distance_transform_edt(~subj)
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
            if g >= GAP and t >= TEX:
                side = 1 if np.median(disp[y:y + TILE, x:x + TILE]) > f_disp else -1
                cands.append((t, y, x, g, side))
    cands.sort(reverse=True)
    pick = {1: [], -1: []}
    for t, y, x, g, side in cands:
        sel = pick[side]
        if len(sel) < MAXT and all(abs(y - yy) >= TILE or abs(x - xx) >= TILE for _, yy, xx, _ in sel):
            sel.append((t, y, x, g))
    tiles, front = pick[1], pick[-1]
    out[i] = dict(size=[H, W], subject_frac=float(subj.mean()), focus_disp=f_disp, n_inner=int((din >= 8).sum()),
                  n_band=int(((din > 0) & (din <= 16) | (dout > 0) & (dout <= 16)).sum()),
                  tiles=[dict(y=y, x=x, texture=t, disp_gap=g) for t, y, x, g in tiles],
                  front_tiles=[dict(y=y, x=x, texture=t, disp_gap=g) for t, y, x, g in front],
                  blur_measurable=len(tiles) >= 2, front_measurable=len(front) >= 2)
    # panel
    vis = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(vis)
    edge = subj & (distance_transform_edt(subj) <= 2)
    a = np.asarray(vis).copy()
    a[edge] = [255, 0, 0]
    vis = Image.fromarray(a)
    d = ImageDraw.Draw(vis)
    for _, y, x, _ in tiles:
        d.rectangle([x, y, x + TILE, y + TILE], outline=(0, 255, 255), width=2)
    for _, y, x, _ in front:
        d.rectangle([x, y, x + TILE, y + TILE], outline=(255, 0, 255), width=2)
    row = [vis, Image.fromarray((disp * 255).astype(np.uint8)).convert("RGB")]
    tgt = f"{ROOT}/targets/{i}_f2.0.png"
    if os.path.exists(tgt):
        row.append(Image.open(tgt).convert("RGB"))
    s = 256 / max(H, W)
    row = [r.resize((int(W * s), int(H * s))) for r in row]
    p = Image.new("RGB", (3 * 260, int(H * s) + 22), (255, 255, 255))
    for k, r in enumerate(row):
        p.paste(r, (k * 260, 22))
    ImageDraw.Draw(p).text((4, 4), f"{i}   subject {subj.mean():.2f}   behind {len(tiles)} (cyan)  front {len(front)} (magenta)   focus disp {f_disp:.2f}",
                           fill=(0, 0, 0))
    panels.append(p)
    print(i, f"subject {subj.mean():.2f} behind {len(tiles)} front {len(front)} focus {f_disp:.2f}", flush=True)
for k in range(0, len(panels), 8):
    ps = panels[k:k + 8]
    sheet = Image.new("RGB", (max(p.width for p in ps), sum(p.height + 6 for p in ps)), (255, 255, 255))
    yy = 0
    for p in ps:
        sheet.paste(p, (0, yy))
        yy += p.height + 6
    sheet.save(f"{ROOT}/sheets/sheet_{k // 8:02d}.png")
json.dump(out, open(f"{ROOT}/regions40.json", "w"), indent=1)
print("regions:", len(out), "images;", sum(v["blur_measurable"] for v in out.values()), "with measurable background blur;",
      sum(v["front_measurable"] for v in out.values()), "with measurable foreground defocus", flush=True)
