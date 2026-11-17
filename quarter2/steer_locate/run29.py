"""Round 29 driver: where can a small, restricted feature edit correct BokehDiff's output?

Run from the BokehDiff repo directory (pinned commit), round 17 venv; started like rounds 18-28.
Configuration = round 25's baseline (fp32 PISA, original gate, frozen radius curve), group 0 / position 0 layout.
Cases: gravel_0, astronaut_0, chelsea_0, coins_0 at targets 8 and 11.625 px.

Edit (one location at a time, only inside the selected patch's cell):
    d = G_s * h - h ;  h' = h + alpha * M * d * rms_M(h) / rms_M(d)
Locations: input / output of down_blocks.1.attentions.0 (the PISA block), mid_block output, up_blocks[-1] output,
final latent before vae.decode, vae.decoder.mid_block output.
Input-only correction: search that patch's blur request alone (grid 0.3-2.0 x baseline request, 3 refinements).
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import measure19 as M  # noqa: E402
import metrics29 as Q  # noqa: E402
import patches as PT  # noqa: E402

OUT = os.environ["R29_OUT"]
R25 = os.environ["R25_OUT"]
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
EDIT = None                    # dict(loc, cell, width, alpha) or None
CALLS, SHAPES = {}, {}
CMASK = {}


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # identical to round 25's baseline path
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def smooth(x, sig):
    """separable Gaussian (sigma in feature cells), replicate padding, per channel"""
    B, C, H, W = x.shape
    rad = max(1, int(np.ceil(3 * sig)))
    g = torch.exp(-0.5 * (torch.arange(-rad, rad + 1, device=x.device, dtype=torch.float32) / sig) ** 2)
    g = g / g.sum()
    y = F.conv2d(F.pad(x, (rad, rad, 0, 0), mode="replicate"), g.view(1, 1, 1, -1).expand(C, 1, 1, -1), groups=C)
    return F.conv2d(F.pad(y, (0, 0, rad, rad), mode="replicate"), g.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)


def edit(h, name):
    """count the call, record the shape, apply the active edit if it targets this location"""
    CALLS[name] = CALLS.get(name, 0) + 1
    SHAPES.setdefault(name, list(h.shape))
    E = EDIT
    if E is None or E["loc"] != name:
        return h
    x = h.float()
    B, C, H, W = x.shape
    sig = E["width"] / (D.S / W)
    m = F.adaptive_avg_pool2d(CMASK[E["cell"]], (H, W))        # area fraction of the cell per feature location
    d = smooth(x, sig) - x
    n = m.sum() * B * C
    rms_h = torch.sqrt((m * x ** 2).sum() / n)
    rms_d = torch.sqrt((m * d ** 2).sum() / n).clamp_min(1e-12)
    return (x + E["alpha"] * m * d * (rms_h / rms_d)).to(h.dtype)


def post_hook(name):
    def fn(module, args, out):
        if isinstance(out, torch.Tensor):
            return edit(out, name)
        if isinstance(out, tuple):
            return (edit(out[0], name),) + tuple(out[1:])
        out.sample = edit(out.sample, name)
        return out
    return fn


def pre_hook(name):
    def fn(module, args, kwargs):
        if len(args):
            return (edit(args[0], name),) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = edit(kwargs["hidden_states"], name)
        return args, kwargs
    return fn


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    global EDIT
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    unet, vae = pipeline.unet, pipeline.vae
    pisa = unet.down_blocks[1].attentions[0]
    names = {"pre_pisa": "unet.down_blocks.1.attentions.0 (input)", "post_pisa": "unet.down_blocks.1.attentions.0 (output)",
             "mid": "unet.mid_block (output)", "up_late": f"unet.up_blocks.{len(unet.up_blocks) - 1} (output, {type(unet.up_blocks[-1]).__name__})",
             "latent": "vae.decode input (unscaled final latent)", "dec_early": "vae.decoder.mid_block (output)"}
    print("PISA block processors:", sorted({type(p).__name__ for n, p in unet.attn_processors.items() if "down_blocks.1.attentions.0" in n}), flush=True)
    pisa.register_forward_pre_hook(pre_hook("pre_pisa"), with_kwargs=True)
    pisa.register_forward_hook(post_hook("post_pisa"))
    unet.mid_block.register_forward_hook(post_hook("mid"))
    unet.up_blocks[-1].register_forward_hook(post_hook("up_late"))
    vae.decoder.mid_block.register_forward_hook(post_hook("dec_early"))
    orig_decode = vae.decode

    def decode(z, *a, **kw):
        return orig_decode(edit(z, "latent"), *a, **kw)

    vae.decode = decode

    img, cid, where = Q.scene()
    for k in PT.FREE:
        CMASK[k] = torch.from_numpy((cid == k).astype(np.float32))[None, None].to(dev)
    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])
    gp = lambda r: r / (D.NOM * D.K)
    x_in = torch.from_numpy(img.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
    tgts = {t: Q.target(img, t) for t in Q.TARGETS}
    ints = {k: D.interior(cid, k) for k in PT.FREE}
    rows, t0, n_gen = [], time.time(), [0]

    def gen(t, tag, over=None, ed=None):
        global EDIT
        r0, _ = curve.inverse(t)
        req = {k: (over or {}).get(k, r0) for k in PT.FREE}
        disp, coc = PT.maps(cid, {k: ("coc", req[k] / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE})
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        CALLS.clear()
        EDIT = ed
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x_in, disp_coc=dc, timeit=True)
        EDIT = None
        assert all(CALLS.get(n, 0) == 1 for n in Q.LOCS), CALLS
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        n_gen[0] += 1
        if n_gen[0] in (5, 25) or n_gen[0] % 100 == 0:
            print(f"{n_gen[0]} generations, {(time.time() - t0) / n_gen[0]:.2f} s each", flush=True)
        return a.astype(np.float32)

    # gate: baseline reproduces round 25, repeats agree, zero edit at every location reproduces the baseline
    base, gate = {}, dict(repro_r25={}, repeat={}, zero_edit={})
    for t in Q.TARGETS:
        base[t] = gen(t, f"base_t{t:.1f}")
        ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{t:.1f}.png")).astype(np.float32)
        gate["repro_r25"][f"{t:g}"] = float(np.abs(base[t] - ref).max())
        gate["repeat"][f"{t:g}"] = float(np.abs(gen(t, f"rep_t{t:.1f}") - base[t]).max())
        for loc in Q.LOCS:
            y = gen(t, f"zero_{loc}_t{t:.1f}", ed=dict(loc=loc, cell=Q.CASES[0][0], width=16, alpha=0.0))
            gate["zero_edit"][f"{loc}_{t:g}"] = float(np.abs(y - base[t]).max())
    print("gate", gate, flush=True)
    print("module shapes", SHAPES, flush=True)

    # input-only correction: that patch's request alone, scored on the primary error against the rendered target
    for k, p in Q.CASES:
        for t in Q.TARGETS:
            r0, _ = curve.inverse(t)
            ev = {}

            def score(f):
                f = round(float(f), 4)
                if f not in ev and f > 0:
                    y = gen(t, f"in_c{k}_t{t:.1f}_f{f:.4f}", over={k: f * r0})
                    ev[f] = Q.aff_mae(y, tgts[t], ints[k])
                    rows.append(dict(kind="input", cell=k, patch=p, target=t, factor=f, request=f * r0,
                                     tag=f"in_c{k}_t{t:.1f}_f{f:.4f}", err=ev[f]))
            for f in np.arange(0.3, 2.0001, 0.1):
                score(f)
            step = 0.1
            for _ in range(3):
                b = min(ev, key=ev.get)
                step /= 2
                score(b - step)
                score(b + step)
            b = min(ev, key=ev.get)
            print(f"input-only {p} t={t:g}: best factor {b} err {ev[b]:.3f} (baseline {ev[1.0]:.3f})", flush=True)

    # feature scan: 6 locations x 2 widths x 4 coefficients x 8 cases
    for loc in Q.LOCS:
        for k, p in Q.CASES:
            for t in Q.TARGETS:
                for w in Q.WIDTHS:
                    for al in Q.ALPHAS:
                        tag = f"ft_{loc}_c{k}_t{t:.1f}_w{w}_a{al:+.2f}"
                        y = gen(t, tag, ed=dict(loc=loc, cell=k, width=w, alpha=al))
                        rows.append(dict(kind="feature", loc=loc, cell=k, patch=p, target=t, width=w, alpha=al, tag=tag,
                                         err=Q.aff_mae(y, tgts[t], ints[k])))
        print("location", loc, "done", flush=True)

    meta = dict(gate=gate, shapes=SHAPES, module_names=names, rows=rows, n_generations=n_gen[0],
                sec_per_gen=(time.time() - t0) / n_gen[0], curve_requests={f"{t:g}": curve.inverse(t)[0] for t in Q.TARGETS})
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    print("done:", n_gen[0], "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
