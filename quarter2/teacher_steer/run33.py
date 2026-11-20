"""Round 33 driver: teacher activation difference (privileged feasibility).

Teacher = the same model, same seed, given an input in which the 4 case cells already show the target blur
(sRGB disk blur of the input at t, pasted inside each case cell) and a zero blur request in those cells (pass-through);
the rest of the scene and every other request are the student's. Student = round 25's baseline.
Step 1 (pass-through gate): does the teacher output keep the target in the case cells?
Step 2: inject the teacher-student difference inside the case cells at one location:
    h' = h + alpha * m * (h_teacher - h)          m = area fraction of the 4 case cells on the feature grid
locations: post_pisa (down_blocks.1.attentions.0), down2, mid, up0, up1 (hidden states only, skip connections stay
the student's); up2 (last up block, almost the answer itself) is the ceiling, not a candidate.
alpha in {0.25, 0.5, 1.0, 1.5}. Request-only comparison: round 31's best request-only output per case, regenerated
here (must reproduce exactly).
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
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import metrics29 as Q  # noqa: E402
import patches as PT  # noqa: E402

OUT = os.environ["R33_OUT"]
R25 = os.environ["R25_OUT"]
R31 = os.environ["R31_OUT"]
LOCS = ["post_pisa", "down2", "mid", "up0", "up1", "up2"]
ALPHAS = [0.25, 0.5, 1.0, 1.5]
CELLS = [k for k, _ in Q.CASES]
OTHER = [k for k in PT.FREE if k not in CELLS]
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
MODE = dict(rec=False, inj=None)            # inj: (loc, alpha)
REC, CALLS = {}, {}
MASK = None


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # round 25's fp32 PISA baseline
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def edit(h, name):
    CALLS[name] = CALLS.get(name, 0) + 1
    if MODE["rec"]:
        REC[name] = h.detach().clone()
    inj = MODE["inj"]
    if inj is None or inj[0] != name:
        return h
    x = h.float()
    m = F.adaptive_avg_pool2d(MASK, x.shape[-2:])
    return (x + inj[1] * m * (REC[name].float() - x)).to(h.dtype)


def post_hook(name):
    def fn(module, args, out):
        if isinstance(out, torch.Tensor):
            return edit(out, name)
        if isinstance(out, tuple):
            return (edit(out[0], name),) + tuple(out[1:])
        out.sample = edit(out.sample, name)
        return out
    return fn


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    global MASK
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    unet = pipeline.unet
    mods = {"post_pisa": unet.down_blocks[1].attentions[0], "down2": unet.down_blocks[2], "mid": unet.mid_block,
            "up0": unet.up_blocks[0], "up1": unet.up_blocks[1], "up2": unet.up_blocks[2]}
    assert len(unet.up_blocks) == 3
    for n, mod in mods.items():
        mod.register_forward_hook(post_hook(n))

    img, cid, where = Q.scene()
    im = img.astype(np.float32)
    case = np.isin(cid, CELLS)
    MASK = torch.from_numpy(case.astype(np.float32))[None, None].to(dev)
    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])
    gp = lambda r: r / (D.NOM * D.K)
    tgts = {t: Q.target(img, t) for t in Q.TARGETS}
    ints = {k: D.interior(cid, k) for k in PT.FREE}
    fm = Q.focus_mask(cid)
    st = m18.stack(im, "disk")
    B = {k: M.stack_at(st, ints[k]) for k in PT.FREE}
    om = np.zeros(cid.shape, bool)
    for k in OTHER:
        om |= ints[k]
    t0, n_gen, rows, base, foc0 = time.time(), [0], [], {}, {}

    def gen(t, tag, over=None, teacher=False, inj=None):
        r0, _ = curve.inverse(t)
        g = {k: ("coc", (over or {}).get(k, r0) / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE}
        x = im
        if teacher:                                    # case cells: target pasted in, zero blur requested
            x = np.where(case[..., None], tgts[t], im)
            for k in CELLS:
                g[k] = ("coc", 0.0, PT.SIGN[k] * gp(t))
        disp, coc = PT.maps(cid, g)
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        x_in = torch.from_numpy(x / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        CALLS.clear()
        MODE["rec"], MODE["inj"] = teacher, inj
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x_in, disp_coc=dc, timeit=True)
        MODE["rec"], MODE["inj"] = False, None
        assert all(CALLS.get(n) == 1 for n in LOCS), CALLS
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        n_gen[0] += 1
        if n_gen[0] in (5, 25) or n_gen[0] % 25 == 0:
            print(f"{n_gen[0]} generations, {(time.time() - t0) / n_gen[0]:.2f} s each", flush=True)
        return a.astype(np.float32)

    def measure(y, t, k):
        return dict(err=Q.aff_mae(y, tgts[t], ints[k]), r_hat=float(M.fit_fast(y[ints[k]], B[k])["v"]),
                    focus_add=Q.raw_mae(y, im, fm) - foc0[t], others_vs_base=Q.raw_mae(y, base[t], om))

    # ---- gate
    gate = {}
    for t in Q.TARGETS:
        base[t] = gen(t, f"base_t{t:.1f}")
        foc0[t] = Q.raw_mae(base[t], im, fm)
        ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{t:.1f}.png")).astype(np.float32)
        gate[f"repro_r25_{t:g}"] = float(np.abs(base[t] - ref).max())
    gate["repeat"] = float(np.abs(gen(11.625, "rep_t11.6") - base[11.625]).max())
    r31 = json.load(open(f"{HERE}/results/r31_cases.json"))
    for k, p in Q.CASES:                                # round 31's request-only winners reproduce exactly
        for t in Q.TARGETS:
            c = r31[f"{p}@{t:g}"]["request"]
            y = gen(t, f"req_{p}_t{t:.1f}", over={k: c["factor"] * curve.inverse(t)[0]})
            ref = np.asarray(Image.open(f"{R31}/img/{c['tag']}.png")).astype(np.float32)
            gate[f"req_repro_{p}_{t:g}"] = float(np.abs(y - ref).max())
            rows.append(dict(kind="request", loc=None, alpha=None, cell=k, patch=p, t=t, tag=f"req_{p}_t{t:.1f}",
                             **measure(y, t, k)))
    teach = {}
    for t in Q.TARGETS:
        teach[t] = gen(t, f"teacher_t{t:.1f}", teacher=True)
        for loc in LOCS:                                # alpha = 0 with the teacher recorded reproduces the baseline
            gate[f"zero_{loc}_{t:g}"] = float(np.abs(gen(t, f"zero_{loc}_t{t:.1f}", inj=(loc, 0.0)) - base[t]).max())
        for k, p in Q.CASES:
            rows.append(dict(kind="teacher", loc=None, alpha=None, cell=k, patch=p, t=t, tag=f"teacher_t{t:.1f}",
                             **measure(teach[t], t, k)))
        for k, p in Q.CASES:
            rows.append(dict(kind="base", loc=None, alpha=None, cell=k, patch=p, t=t, tag=f"base_t{t:.1f}",
                             **measure(base[t], t, k)))
        # ---- injection scan (REC holds this target's teacher activations)
        for loc in LOCS:
            for a in ALPHAS:
                tag = f"inj_{loc}_a{a:.2f}_t{t:.1f}"
                y = gen(t, tag, inj=(loc, a))
                for k, p in Q.CASES:
                    rows.append(dict(kind="inject", loc=loc, alpha=a, cell=k, patch=p, t=t, tag=tag, **measure(y, t, k)))
        print(f"target {t:g} done", flush=True)
    print("gate", gate, flush=True)
    bad = {k: v for k, v in gate.items() if v != 0.0}
    meta = dict(gate=gate, gate_ok=not bad, rows=rows, locs=LOCS, alphas=ALPHAS, cells=CELLS, other_cells=OTHER,
                n_generations=n_gen[0], sec_per_gen=(time.time() - t0) / n_gen[0])
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    assert not bad, bad
    print("done:", n_gen[0], "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
