"""Reference-set bias pilot on the SD 1.5 depth-of-field setup (sd_dof.py): at comparable reductions in measured
image sharpness, does the portrait composition of the reference set change the generated scene's framing or content?

    python quarter2/sd_dof/ref_bias.py label [--attr portrait|closeup] [--smoke]

Stage `label`: portrait-composition labels for the 1,200 direction photos (same order and 512 px crop as
sd_dof.py's acts.pt), each photo's log Laplacian variance, the 2 x 2 counts (shallow / deep x portrait / not), and a
blinded audit sheet of 24 photos (6 random per cell) with its key kept separately.

Portrait composition = a person is the main subject and fills a large part of the frame. Labelled by CLIP ViT-L/14
zero-shot: probability mass on portrait prompts against prompts for scenes, objects, animals, and distant people or
crowds (a face detector would count distant faces and miss faceless portraits).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sd_dof import CLIP, ROOT, dof_photos, log

SRC = ROOT / "outputs" / "q2_sd_dof"
ATTRS = {
    # portrait composition: a person is the main subject and fills a large part of the frame
    "portrait": (["a portrait photo of a person", "a close-up photo of a person", "a headshot of a person",
                  "a photo of a person posing as the main subject", "a portrait photo of a woman",
                  "a portrait photo of a man", "a portrait photo of a child"],
                 ["a photo of a landscape", "a photo of a city street", "a photo of a building", "a photo of an animal",
                  "a photo of a plant or flowers", "a photo of food or a drink", "a photo of an object",
                  "a photo of a room", "a photo of a vehicle", "a photo of a small person far away in a large scene",
                  "a photo of a crowd of people", "a photo of a person's hands holding something"]),
    # close-up composition: one subject (any kind) fills a large part of the frame, vs. a wide scene
    "closeup": (["a close-up photo of a single subject filling the frame", "a macro photo", "a close-up photo of an animal",
                 "a close-up photo of a flower", "a close-up photo of an object", "a close-up portrait of a person",
                 "a close-up photo of food"],
                ["a wide photo of a landscape", "a wide shot of a city street", "a photo of a large scene with many things in it",
                 "a photo of a building seen from a distance", "a wide photo of a room", "an aerial photo",
                 "a photo of a road stretching into the distance", "a photo of a forest", "a photo of the sky",
                 "a photo of a small figure far away in a large scene"]),
}
THRESH = [0.3, 0.5, 0.7]
HI, LO = 0.85, 0.15             # confident close-up / wide (blind audit: 11 of 12 correct at these extremes)


def label(out, smoke, attr):
    POS, NEG = ATTRS[attr]
    from scipy.ndimage import laplace
    from transformers import CLIPModel, CLIPProcessor
    imgs, labels, names = dof_photos(30 if smoke else None)
    a = torch.load(str(SRC / "acts.pt"), map_location="cpu", mmap=True) if (SRC / "acts.pt").exists() else None
    shallow = a["shallow"] if a is not None else 0
    if a is not None and not smoke:
        assert np.array_equal(a["labels"].numpy(), labels), "photo order differs from acts.pt"
    proc = CLIPProcessor.from_pretrained(CLIP)
    model = CLIPModel.from_pretrained(CLIP).to("cuda").eval()
    with torch.no_grad():
        t = proc(text=POS + NEG, return_tensors="pt", padding=True).to("cuda")
        tf = model.get_text_features(**t)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        probs = []
        for i in range(0, len(imgs), 32):
            x = proc(images=imgs[i:i + 32], return_tensors="pt").to("cuda")
            f = model.get_image_features(**x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (model.logit_scale.exp() * f @ tf.T).softmax(-1)
            probs.append(p[:, :len(POS)].sum(-1).cpu())
    pp = torch.cat(probs).numpy()
    sharp = np.array([np.log(laplace(np.asarray(im.convert("L")).astype(float)).var() + 1e-6) for im in imgs])
    sh = labels == shallow
    rows = [dict(i=int(i), shallow=bool(sh[i]), p=float(pp[i]), log_sharp=float(sharp[i]))
            for i in range(len(imgs))]
    (out / f"labels_{attr}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    L = [f"# Reference groups: {attr} composition x depth of field", "",
         f"{len(imgs)} photos from svnfs/depth-of-field (label {shallow} = shallow). {attr} = CLIP ViT-L/14 "
         f"zero-shot probability on {len(POS)} {attr} prompts against {len(NEG)} others >= threshold.", "",
         f"| threshold | shallow, {attr} | shallow, not | deep, {attr} | deep, not |", "|---|---|---|---|---|"]
    for th in THRESH:
        po = pp >= th
        L.append(f"| {th} | {int((sh & po).sum())} | {int((sh & ~po).sum())} | {int((~sh & po).sum())} | "
                 f"{int((~sh & ~po).sum())} |")
    hist = np.histogram(pp, bins=np.linspace(0, 1, 11))[0]
    L += ["", f"{attr} probability, counts per 0.1 bin: " + ", ".join(map(str, hist)), "",
          "Mean log Laplacian variance per cell (threshold 0.5):", ""]
    po = pp >= 0.5
    for nm, m in ((f"shallow, {attr}", sh & po), ("shallow, not", sh & ~po), (f"deep, {attr}", ~sh & po),
                  ("deep, not", ~sh & ~po)):
        if m.any():
            q = np.quantile(sharp[m], [0.1, 0.5, 0.9])
            L.append(f"- {nm}: n {int(m.sum())}, mean {sharp[m].mean():.2f}, 10/50/90% {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")

    # blinded audit: 6 random photos per cell (threshold 0.5), shuffled; key saved separately
    rng = np.random.default_rng(0)
    pick = []
    for m in (sh & po, sh & ~po, ~sh & po, ~sh & ~po):
        idx = np.flatnonzero(m)
        pick += list(rng.choice(idx, min(6, len(idx)), replace=False))
    pick = [int(i) for i in rng.permutation(pick)]
    W, H, C = 200, 18, 6
    sheet = Image.new("RGB", (C * W, -(-len(pick) // C) * (W + H)), "white")
    dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(pick):
        x, y = (k % C) * W, (k // C) * (W + H)
        dr.text((x + 4, y + 3), f"#{k}", fill="black")
        sheet.paste(imgs[i].resize((W, W), Image.LANCZOS), (x, y + H))
    sheet.save(out / f"audit_{attr}_blind.png")
    (out / f"audit_{attr}_key.json").write_text(json.dumps(
        [dict(k=k, i=i, shallow=bool(sh[i]), p=round(float(pp[i]), 3)) for k, i in enumerate(pick)], indent=1))
    L += ["", f"`audit_{attr}_blind.png`: 24 photos (6 random per cell), numbered; `audit_{attr}_key.json` holds their labels.", ""]
    (out / f"groups_{attr}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def sheets(out, smoke):
    """Blinded audit sheets: both rare groups in full (shallow wide, deep close-up) and 24 random photos of each common
    group, confident CLIP labels only, shuffled across groups; 24 photos per sheet; the key is a separate file."""
    imgs, labels, names = dof_photos(30 if smoke else None)
    rows = [json.loads(l) for l in (out / "labels_closeup.jsonl").open()]
    cell = {"shallow_wide": [r["i"] for r in rows if r["shallow"] and r["p"] <= LO],
            "deep_closeup": [r["i"] for r in rows if not r["shallow"] and r["p"] >= HI]}
    rng = np.random.default_rng(1)
    for nm, sh, f in (("shallow_closeup", True, lambda p: p >= HI), ("deep_wide", False, lambda p: p <= LO)):
        idx = [r["i"] for r in rows if r["shallow"] == sh and f(r["p"])]
        cell[nm] = [int(i) for i in rng.choice(idx, min(24, len(idx)), replace=False)]
    allk = [(i, nm) for nm, ids in cell.items() for i in ids]
    allk = [allk[j] for j in rng.permutation(len(allk))]
    W, H, C, PER = 200, 18, 6, 24
    for s0 in range(0, len(allk), PER):
        chunk = allk[s0:s0 + PER]
        sheet = Image.new("RGB", (C * W, -(-len(chunk) // C) * (W + H)), "white")
        dr = ImageDraw.Draw(sheet)
        for k, (i, _) in enumerate(chunk):
            x, y = (k % C) * W, (k // C) * (W + H)
            dr.text((x + 4, y + 3), f"#{s0 + k}", fill="black")
            sheet.paste(imgs[i].resize((W, W), Image.LANCZOS), (x, y + H))
        sheet.save(out / f"audit2_sheet{s0 // PER}.png")
    (out / "audit2_key.json").write_text(json.dumps([dict(k=k, i=i, group=nm) for k, (i, nm) in enumerate(allk)]))
    log(f"{len(allk)} photos on {-(-len(allk) // PER)} sheets; groups " + ", ".join(f"{k} {len(v)}" for k, v in cell.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=["label", "sheets"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--attr", choices=list(ATTRS), default="portrait")
    cfg = ap.parse_args()
    out = ROOT / "outputs" / ("q2_ref_bias_smoke" if cfg.smoke else "q2_ref_bias")
    out.mkdir(parents=True, exist_ok=True)
    for st in cfg.stages:
        log(f"stage {st}{' (smoke)' if cfg.smoke else ''}")
        if st == "label":
            label(out, cfg.smoke, cfg.attr)
        elif st == "sheets":
            sheets(out, cfg.smoke)
    log("done")


if __name__ == "__main__":
    main()
