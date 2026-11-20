"""Presentation scene: round 25's group 0 / position 0 layout with the two filler corners replaced by two more held-out
patches (TL = chelsea_1, front; BR = coins_1, behind), so all 8 background cells are real content at the same
|disparity gap| and the same calibrated 11.625 px request. Centre (brick) stays at the focus.

Generations (round 25 baseline config: fp32 PISA, original gate, frozen radius curve C1 G0, seed 0):
  gate_r25      round 25's exact scene/maps -> must reproduce cor_g0_r0_G0_t11.6.png
  scene8_norm   8 real cells; disparity normalised like BokehDiff's loader (front 1, focus 0.5, behind 0)  [figure]
  scene8_raw    8 real cells; disparity 0.5 +- gap (span 0.2..0.8), as a sensitivity check
Run from the BokehDiff repo directory (pinned commit), round 17 venv, like rounds 18-29. Writes $R8_OUT/img + gate.json.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402
import PISA_attn_processor as PA  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402

OUT = os.environ["R8_OUT"]
R25 = os.environ["R25_OUT"]
T = 11.625
CORNERS = {0: ("chelsea_1", +1), 8: ("coins_1", -1)}
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # round 25's fp32 PISA baseline
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def scene8():
    lib = {p["id"]: p for p in PT.library()}
    grp = [p for p in PT.library() if p["split"] == "test"][0::2]
    img, cid, where = PT.layout(grp, 0)
    img = img.copy()
    for k, (pid, _) in CORNERS.items():
        r, c = divmod(k, 3)
        ys, xs = D.cell_slice((r, c))
        img[ys, xs] = lib[pid]["pix"][:ys.stop - ys.start, :xs.stop - xs.start]
        where[k] = pid
    return img, cid, where


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])
    r0, _ = curve.inverse(T)
    gp = T / (D.NOM * D.K)
    coc_v = r0 / (D.NOM * 10)
    sign = {**PT.SIGN, **{k: s for k, (_, s) in CORNERS.items()}}

    def gen(img, disp, coc, tag):
        x = torch.from_numpy(img.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        dc = torch.from_numpy(np.stack([disp, coc]).astype(np.float32)[None]).to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x, disp_coc=dc, timeit=True)
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        return a.astype(np.float32)

    # gate: round 25's exact scene and maps
    grp = [p for p in PT.library() if p["split"] == "test"][0::2]
    img25, cid, _ = PT.layout(grp, 0)
    disp, coc = PT.maps(cid, {k: ("coc", coc_v, PT.SIGN[k] * gp) for k in PT.FREE})
    y = gen(img25, disp, coc, "gate_r25")
    ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{T:.1f}.png")).astype(np.float32)
    gate = dict(repro_r25_maxdiff=float(np.abs(y - ref).max()), request_px=float(r0), coc_value=float(coc_v))
    print("gate", gate, flush=True)

    img8, cid, where = scene8()
    Image.fromarray(img8).save(f"{OUT}/img/input8.png")
    coc8 = np.zeros((D.S, D.S), np.float32)
    dn = np.full((D.S, D.S), 0.5, np.float32)
    dr = np.full((D.S, D.S), 0.5, np.float32)
    for k in range(9):
        if k == 4:
            continue
        m = cid == k
        coc8[m] = coc_v
        dn[m] = 0.5 + 0.5 * sign[k]
        dr[m] = 0.5 + gp * sign[k]
    gen(img8, dn, coc8, "scene8_norm")
    gen(img8, dr, coc8, "scene8_raw")
    gate.update(where={str(k): v for k, v in where.items()}, sign={str(k): v for k, v in sign.items()})
    json.dump(gate, open(f"{OUT}/gate.json", "w"), indent=1)
    print("done", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
