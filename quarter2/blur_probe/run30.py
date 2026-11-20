"""Round 30 driver: can a linear readout of BokehDiff's internal activations predict the blur it will deliver?

Configuration = round 25's baseline (fp32 PISA, original gate, frozen radius curve C1 G0), seed 0, untiled 512^2.
Scenes: round 24/25 3x3 layouts (centre focus, corner anchors, 6 free cells, same |gap| for all free cells).
For each generation, every free cell's activations are mean-pooled over its interior at each candidate layer
(post-PISA through the final latent), and the output image is saved; delivered radii are measured afterwards.

SMOKE=1: gate (reproduce round 25's cor_g0_r0_G0_t11.6 exactly) + 2 dev generations, then stop.
SMOKE=0: dev patches (probe fitting) and test patches (held out), all 6 rotations, targets TARGETS.
Writes $R30_OUT/img/*.png, feats.npz (layer -> [n_gen, 7 cells, C]), meta.json.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402
import PISA_attn_processor as PA  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402

OUT = os.environ["R30_OUT"]
R25 = os.environ["R25_OUT"]
SMOKE = os.environ.get("SMOKE", "1") == "1"
TARGETS = [4.0, 6.0, 8.0, 10.0, 11.625, 14.0]
ROTS = [0, 1, 2, 3, 4, 5]
CELLS = PT.FREE + [4]                      # 6 free cells + the focus cell (control)
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
CALLS, SHAPES, CUR = {}, {}, {}
POOL = {}                                  # layer -> [7, C] for the current generation
PMASK = {}                                 # (H, W) -> [7, H, W] pooled interior-mask weights


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # round 25's fp32 PISA baseline
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def record(h, name):
    CALLS[name] = CALLS.get(name, 0) + 1
    SHAPES.setdefault(name, list(h.shape))
    x = h.float()[0]                                       # [C, H, W]
    C, H, W = x.shape
    if (H, W) not in PMASK:
        m = F.adaptive_avg_pool2d(CUR["imask"], (H, W))[0]  # [7, H, W] area fraction of each cell interior
        PMASK[(H, W)] = m / m.sum((1, 2), keepdim=True).clamp_min(1e-12)
    POOL[name] = torch.einsum("chw,khw->kc", x, PMASK[(H, W)]).cpu().numpy()


def post_hook(name):
    def fn(module, args, out):
        h = out if isinstance(out, torch.Tensor) else out[0] if isinstance(out, tuple) else out.sample
        record(h, name)
    return fn


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    unet, vae = pipeline.unet, pipeline.vae
    layers = {"post_pisa": unet.down_blocks[1].attentions[0], "down1": unet.down_blocks[1],
              "down2": unet.down_blocks[2], "mid": unet.mid_block}
    for i, b in enumerate(unet.up_blocks):
        layers[f"up{i}"] = b
    for n, mod in layers.items():
        mod.register_forward_hook(post_hook(n))
    orig_decode = vae.decode

    def decode(z, *a, **kw):
        record(z, "latent")
        return orig_decode(z, *a, **kw)

    vae.decode = decode
    names = list(layers) + ["latent"]
    print("layers", names, flush=True)

    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])
    lib = PT.library()
    groups = {f"{s}{g}": [p for p in lib if p["split"] == s][g::2] for s in ("test", "dev") for g in (0, 1)}
    feats, rows, t0 = {n: [] for n in names}, [], time.time()

    def gen(split, g, rot, t, tag):
        img, cid, where = PT.layout(groups[f"{split}{g}"], rot)
        CUR["imask"] = torch.from_numpy(np.stack([D.interior(cid, k) for k in CELLS]).astype(np.float32))[None].to(dev)
        PMASK.clear()
        r0, flag = curve.inverse(t)
        gp = t / (D.NOM * D.K)
        disp, coc = PT.maps(cid, {k: ("coc", r0 / (D.NOM * 10), PT.SIGN[k] * gp) for k in PT.FREE})
        x = torch.from_numpy(img.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        CALLS.clear()
        POOL.clear()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x, disp_coc=dc, timeit=True)
        assert all(CALLS.get(n, 0) == 1 for n in names), CALLS
        for n in names:
            assert np.isfinite(POOL[n]).all(), n
            feats[n].append(POOL[n].astype(np.float32))
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        rows.append(dict(tag=tag, split=split, group=g, rot=rot, target=t, request=r0, flag=flag,
                         cells=CELLS, patches=[where.get(k, "focus") for k in CELLS]))
        return a.astype(np.float32)

    # gate: round 25 test group 0, position 0, 11.6 px must reproduce exactly
    y = gen("test", 0, 0, 11.625, "gate_test0_r0_t11.6")
    ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t11.6.png")).astype(np.float32)
    gate = dict(repro_r25_maxdiff=float(np.abs(y - ref).max()))
    print("gate", gate, "shapes", SHAPES, flush=True)
    assert gate["repro_r25_maxdiff"] == 0.0, gate

    if SMOKE:
        plan = [("dev", 0, 0, 8.0), ("dev", 1, 3, 14.0)]
    else:
        plan = [(s, g, r, t) for s in ("dev", "test") for g in (0, 1) for r in ROTS for t in TARGETS]
    for i, (s, g, r, t) in enumerate(plan):
        gen(s, g, r, t, f"{s}{g}_r{r}_t{t:.1f}")
        if i in (4, 24) or (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(plan)} generations, {(time.time() - t0) / (i + 2):.2f} s each", flush=True)

    np.savez_compressed(f"{OUT}/feats.npz", **{n: np.stack(v) for n, v in feats.items()})
    json.dump(dict(smoke=SMOKE, gate=gate, shapes=SHAPES, layers=names, rows=rows, targets=TARGETS,
                   sec_per_gen=(time.time() - t0) / len(rows)), open(f"{OUT}/meta.json", "w"), indent=1)
    print("done:", len(rows), "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
