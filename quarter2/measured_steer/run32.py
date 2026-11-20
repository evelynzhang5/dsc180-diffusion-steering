"""Round 32 driver: measured-error adaptive steering vs equally adaptive request adjustment.

Configuration = round 25's baseline (fp32 PISA, original gate, frozen radius curve C1 G0), seed 0, untiled 512^2,
held-out group 0 / position 0 layout; round 29's 4 case cells (gravel_0, astronaut_0, chelsea_0, coins_0) at
targets 8 and 11.625. The two other free cells (rocket_0, text_0) and the focus are never edited.

Edit at layer L (mid_block or up_blocks.0 output), per case cell k with its own strength a_k:
    h' = h + sum_k a_k * m_k * u_L,   u_L = unit direction * dev high-vs-low difference norm (directions32)
m_k = area fraction of cell k per feature location. a = 1 shifts the cell by the difference between
development cells above and below the label median.

Procedure per target (every probe generation applies the same value to all 4 case cells; each cell is measured):
  base    round 25 baseline (pass 1: measure each region's error)
  R1      request round 1: probes factor {0.7,0.85,0.95,1.05,1.15,1.3} x baseline request; per cell choose; 1 final
  R+R     request round 2 on top of R1: probes {0.9,0.95,0.98,1.02,1.05,1.1} x R1 request; choose; 1 final
  S+R     per arm, steering on top of R1 requests: probes a in {-2,-1,-0.5,0.5,1,2}; choose; 1 final
  S       per arm, steering from the baseline request (reported): same probes; choose; 1 final
CHOOSE (per cell, from that round's start generation plus its 6 probes; measured only, no search beyond):
  valid = |r_hat - t| <= max(2, 20 %) and focus_add <= 0.5/255; lowest primary error among valid,
  else the value whose delivered radius is closest to the target.
Arms: {mid, up0} x {size, shape, random}.
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

OUT = os.environ["R32_OUT"]
R25 = os.environ["R25_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
LAYERS = ["mid", "up0"]
DIRS = ["size", "shape", "random"]
R1_F = [0.7, 0.85, 0.95, 1.05, 1.15, 1.3]
R2_F = [0.9, 0.95, 0.98, 1.02, 1.05, 1.1]
ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
CELLS = [k for k, _ in Q.CASES]
OTHER = [k for k in PT.FREE if k not in CELLS]
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
EDIT, CALLS, CMASK = None, {}, {}


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # round 25's fp32 PISA baseline
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def edit(h, name):
    CALLS[name] = CALLS.get(name, 0) + 1
    E = EDIT
    if E is None or E["layer"] != name:
        return h
    x = h.float()
    B, C, H, W = x.shape
    add = torch.zeros((1, 1, H, W), device=x.device)
    for k, a in E["alpha"].items():
        add = add + a * F.adaptive_avg_pool2d(CMASK[k], (H, W))
    return (x + add * E["u"].view(1, C, 1, 1)).to(h.dtype)


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
    global EDIT
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    unet = pipeline.unet
    unet.mid_block.register_forward_hook(post_hook("mid"))
    unet.up_blocks[0].register_forward_hook(post_hook("up0"))
    V = np.load(f"{HERE}/results/directions32.npz")
    vinfo = json.load(open(f"{HERE}/results/directions32.json"))
    U = {}
    for L in LAYERS:
        scale = vinfo[f"{L}_size"]["raw_norm"]            # random keeps the size direction's magnitude
        for d in DIRS:
            s = vinfo[f"{L}_{d}"]["raw_norm"] if d != "random" else scale
            U[(L, d)] = torch.from_numpy(V[f"{L}_{d}"] * s).to(dev)

    img, cid, where = Q.scene()
    im = img.astype(np.float32)
    for k in PT.FREE:
        CMASK[k] = torch.from_numpy((cid == k).astype(np.float32))[None, None].to(dev)
    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])
    gp = lambda r: r / (D.NOM * D.K)
    x_in = torch.from_numpy(im / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
    tgts = {t: Q.target(img, t) for t in Q.TARGETS}
    ints = {k: D.interior(cid, k) for k in PT.FREE}
    fm = Q.focus_mask(cid)
    st = m18.stack(im, "disk")
    B = {k: M.stack_at(st, ints[k]) for k in PT.FREE}
    om = np.zeros(cid.shape, bool)
    for k in OTHER:
        om |= ints[k]
    rows, t0, n_gen, base, foc0 = [], time.time(), [0], {}, {}

    def gen(t, tag, req=None, ed=None):
        global EDIT
        r0, _ = curve.inverse(t)
        rq = {k: (req or {}).get(k, r0) for k in PT.FREE}
        disp, coc = PT.maps(cid, {k: ("coc", rq[k] / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE})
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        CALLS.clear()
        EDIT = ed
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x_in, disp_coc=dc, timeit=True)
        EDIT = None
        assert CALLS.get("mid") == 1 and CALLS.get("up0") == 1, CALLS
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        n_gen[0] += 1
        if n_gen[0] in (5, 25) or n_gen[0] % 50 == 0:
            print(f"{n_gen[0]} generations, {(time.time() - t0) / n_gen[0]:.2f} s each", flush=True)
        return a.astype(np.float32)

    def measure(y, t):
        """per case cell: primary error, delivered radius, focus added; per generation: other-cell change"""
        foc = Q.raw_mae(y, im, fm) - foc0[t] if t in foc0 else 0.0
        oth = Q.raw_mae(y, base[t], om) if t in base else 0.0
        return {k: dict(err=Q.aff_mae(y, tgts[t], ints[k]), r_hat=float(M.fit_fast(y[ints[k]], B[k])["v"]),
                        focus_add=foc, others_vs_base=oth) for k in CELLS}

    valid = lambda d, t: abs(d["r_hat"] - t) <= m18.TOL(t) and d["focus_add"] <= Q.FOCUS_ADD_MAX

    def choose(opts, t):
        """opts: list of (value, per-cell measurement dict); returns {cell: value}"""
        out = {}
        for k in CELLS:
            ok = [(v, m[k]) for v, m in opts if valid(m[k], t)]
            out[k] = (min(ok, key=lambda x: x[1]["err"]) if ok else
                      min(opts, key=lambda x: abs(x[1][k]["r_hat"] - t)))[0]
        return out

    def record(t, stage, arm, tag, meas, setting, kind):
        for k, p in Q.CASES:
            rows.append(dict(t=t, stage=stage, arm=arm, kind=kind, tag=tag, cell=k, patch=p,
                             setting=setting[k] if isinstance(setting, dict) else setting, **meas[k]))

    # ---- gate
    gate = {}
    for t in Q.TARGETS:
        base[t] = gen(t, f"base_t{t:.1f}")
        foc0[t] = Q.raw_mae(base[t], im, fm)
        ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{t:.1f}.png")).astype(np.float32)
        gate[f"repro_r25_{t:g}"] = float(np.abs(base[t] - ref).max())
    gate["repeat"] = float(np.abs(gen(11.625, "rep_t11.6") - base[11.625]).max())
    for L in LAYERS:
        z = dict(layer=L, u=U[(L, "random")], alpha={k: 0.0 for k in CELLS})
        gate[f"zero_{L}"] = float(np.abs(gen(11.625, f"zero_{L}", ed=z) - base[11.625]).max())
        z = dict(layer=L, u=U[(L, "random")], alpha={k: 2.0 for k in CELLS})
        gate[f"random2_{L}_changes"] = float(np.abs(gen(11.625, f"rand2_{L}", ed=z) - base[11.625]).max())
    print("gate", gate, flush=True)
    assert all(v == 0.0 for k, v in gate.items() if "changes" not in k), gate
    assert all(v > 0 for k, v in gate.items() if "changes" in k), gate
    if SMOKE:
        y = gen(11.625, "smoke_size_mid", ed=dict(layer="mid", u=U[("mid", "size")], alpha={k: 1.0 for k in CELLS}))
        print("smoke: mid size a=1 per-cell", {k: (round(d["err"], 2), round(d["r_hat"], 1))
                                               for k, d in measure(y, 11.625).items()},
              "base", {k: (round(d["err"], 2), round(d["r_hat"], 1)) for k, d in measure(base[11.625], 11.625).items()},
              flush=True)
        json.dump(dict(gate=gate, smoke=True), open(f"{OUT}/conds.json", "w"), indent=1)
        print("done: smoke", n_gen[0], "generations", flush=True)
        os._exit(0)

    for t in Q.TARGETS:
        r0, _ = curve.inverse(t)
        mb = measure(base[t], t)
        record(t, "base", "base", f"base_t{t:.1f}", mb, 1.0, "final")

        def request_round(start_req, start_meas, factors, stage):
            opts = [(1.0, start_meas)]
            for f in factors:
                tag = f"{stage}_t{t:.1f}_f{f:.2f}"
                m = measure(gen(t, tag, req={k: start_req[k] * f for k in CELLS}), t)
                record(t, stage, "request", tag, m, f, "probe")
                opts.append((f, m))
            ch = choose(opts, t)
            req = {k: start_req[k] * ch[k] for k in CELLS}
            tag = f"{stage}_t{t:.1f}_final"
            m = measure(gen(t, tag, req=req), t)
            record(t, stage, "request", tag, m, {k: req[k] / r0 for k in CELLS}, "final")
            return req, m

        req1, m1 = request_round({k: r0 for k in CELLS}, mb, R1_F, "R1")
        request_round(req1, m1, R2_F, "RR")

        for L in LAYERS:
            for d in DIRS:
                arm = f"{L}_{d}"
                for stage, req, start in (("SR", req1, m1), ("S", None, mb)):
                    opts = [(0.0, start)]
                    for a in ALPHAS:
                        tag = f"{stage}_{arm}_t{t:.1f}_a{a:+.1f}"
                        m = measure(gen(t, tag, req=req, ed=dict(layer=L, u=U[(L, d)], alpha={k: a for k in CELLS})), t)
                        record(t, stage, arm, tag, m, a, "probe")
                        opts.append((a, m))
                    ch = choose(opts, t)
                    tag = f"{stage}_{arm}_t{t:.1f}_final"
                    m = measure(gen(t, tag, req=req, ed=dict(layer=L, u=U[(L, d)], alpha=ch)), t)
                    record(t, stage, arm, tag, m, ch, "final")
        print(f"target {t:g} done", flush=True)
        json.dump(dict(gate=gate, rows=rows), open(f"{OUT}/conds_partial.json", "w"))

    meta = dict(gate=gate, rows=rows, layers=LAYERS, dirs=DIRS, alphas=ALPHAS, r1=R1_F, r2=R2_F, cells=CELLS,
                other_cells=OTHER, n_generations=n_gen[0], sec_per_gen=(time.time() - t0) / n_gen[0],
                curve_requests={f"{t:g}": curve.inverse(t)[0] for t in Q.TARGETS})
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    print("done:", n_gen[0], "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
