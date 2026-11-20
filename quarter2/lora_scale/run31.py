"""Round 31 driver: does scaling BokehDiff's LoRA in one UNet block improve the output beyond adjusting the request?

Configuration = round 25's baseline (fp32 PISA, original gate, frozen radius curve C1 G0), seed 0, untiled 512^2,
held-out group 0 / position 0 layout; the 8 cases of round 29 (4 patches x targets 8, 11.625).
Intervention: multiply the PEFT LoRA scaling of every LoRA layer inside one block (down0-2, mid, up0-2) by s.
It acts on the whole image (all cells), so one generation measures all 4 cases at a target.

Stages:
  gate     baseline reproduces round 25 exactly; repeat identical; s = 1.0 in every block identical; s = 1.5
           in every block changes the output (the hook works); LoRA layers found in every block.
  scan     7 blocks x SCALES x 2 targets at the baseline request.
  input    request-only search per case (that cell's request only), procedure SEARCH below.
  retune   per case: the TOP_K scan variants with the lowest primary error among those passing the focus guard,
           each with the same request search for that cell.
SEARCH: factors 0.3-2.0 step 0.1 x baseline request, then 3 refinements (+-0.05, +-0.025, +-0.0125) around the
current best valid candidate (within tolerance and focus guard; lowest primary error), or, if none is valid,
the candidate whose delivered radius is closest to the target.
Writes $R31_OUT/img/*.png and conds.json (one row per generation and case, with err, r_hat, focus_add, others).
"""
import json
import os
import sys
import time

import numpy as np
import torch
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

OUT = os.environ["R31_OUT"]
R25 = os.environ["R25_OUT"]
BLOCKS = ["down0", "down1", "down2", "mid", "up0", "up1", "up2"]
SCALES = [0.5, 0.75, 1.25, 1.5]
TOP_K = 3
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # round 25's fp32 PISA baseline
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def block_of(name):
    p = name.split(".")
    if p[0] in ("down_blocks", "up_blocks"):
        return ("down" if p[0] == "down_blocks" else "up") + p[1]
    return "mid" if p[0] == "mid_block" else None


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    from peft.tuners.lora import LoraLayer
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    unet = pipeline.unet
    lora = {b: [] for b in BLOCKS}
    other = []
    for n, mod in unet.named_modules():
        if isinstance(mod, LoraLayer):
            b = block_of(n)
            (lora[b] if b in lora else other).append(mod)
    orig = {id(m): dict(m.scaling) for b in BLOCKS for m in lora[b]}
    counts = {b: len(v) for b, v in lora.items()}
    print("LoRA layers per block", counts, "outside blocks", len(other),
          "adapters", sorted({a for b in BLOCKS for m in lora[b] for a in m.scaling}), flush=True)
    assert all(counts[b] > 0 for b in BLOCKS) and not other, (counts, len(other))

    def set_scale(sc):                                    # sc: {block: s}; blocks not listed are restored
        for b in BLOCKS:
            s = sc.get(b, None)
            for m in lora[b]:
                for a, v in orig[id(m)].items():
                    m.scaling[a] = v if s is None else v * s

    img, cid, where = Q.scene()
    im = img.astype(np.float32)
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
    rows, t0, n_gen, base, foc0 = [], time.time(), [0], {}, {}

    def gen(t, tag, over=None, sc=None):
        r0, _ = curve.inverse(t)
        req = {k: (over or {}).get(k, r0) for k in PT.FREE}
        disp, coc = PT.maps(cid, {k: ("coc", req[k] / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE})
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        set_scale(sc or {})
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x_in, disp_coc=dc, timeit=True)
        set_scale({})
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        n_gen[0] += 1
        if n_gen[0] in (5, 25) or n_gen[0] % 100 == 0:
            print(f"{n_gen[0]} generations, {(time.time() - t0) / n_gen[0]:.2f} s each", flush=True)
        return a.astype(np.float32)

    def measure(y, t, k):
        d = dict(err=Q.aff_mae(y, tgts[t], ints[k]), r_hat=float(M.fit_fast(y[ints[k]], B[k])["v"]))
        if t in base:
            d["focus_add"] = Q.raw_mae(y, im, fm) - foc0[t]
            om = np.zeros(cid.shape, bool)
            for j in PT.FREE:
                if j != k:
                    om |= ints[j]
            d["others_vs_base"] = Q.raw_mae(y, base[t], om)
        return d

    valid = lambda d, t: abs(d["r_hat"] - t) <= m18.TOL(t) and d["focus_add"] <= Q.FOCUS_ADD_MAX

    # ---- gate (smoke stage: aborts the run if anything is off)
    gate = {}
    for t in Q.TARGETS:
        base[t] = gen(t, f"base_t{t:.1f}")
        foc0[t] = Q.raw_mae(base[t], im, fm)
        ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{t:.1f}.png")).astype(np.float32)
        gate[f"repro_r25_{t:g}"] = float(np.abs(base[t] - ref).max())
    gate["repeat"] = float(np.abs(gen(11.625, "rep_t11.6") - base[11.625]).max())
    gate["scale1_all"] = float(np.abs(gen(11.625, "s1_all_t11.6", sc={b: 1.0 for b in BLOCKS}) - base[11.625]).max())
    gate["scale1p5_all_changes"] = float(np.abs(gen(11.625, "s1.5_all_t11.6", sc={b: 1.5 for b in BLOCKS}) - base[11.625]).max())
    print("gate", gate, flush=True)
    assert all(v == 0.0 for k, v in gate.items() if k != "scale1p5_all_changes") and gate["scale1p5_all_changes"] > 0, gate
    for k, p in Q.CASES:
        for t in Q.TARGETS:
            rows.append(dict(stage="base", cell=k, patch=p, target=t, block=None, scale=1.0, factor=1.0,
                             tag=f"base_t{t:.1f}", **measure(base[t], t, k)))

    def search(k, p, t, stage, sc, prefix):
        r0, _ = curve.inverse(t)
        ev = {}

        def score(f):
            f = round(float(f), 4)
            if f > 0 and f not in ev:
                tag = f"{prefix}_c{k}_t{t:.1f}_f{f:.4f}"
                d = measure(gen(t, tag, over={k: f * r0}, sc=sc), t, k)
                ev[f] = d
                blk, s = (next(iter(sc.items())) if sc else (None, 1.0))
                rows.append(dict(stage=stage, cell=k, patch=p, target=t, block=blk, scale=s, factor=f, tag=tag, **d))
        for f in np.arange(0.3, 2.0001, 0.1):
            score(f)
        step = 0.1
        for _ in range(3):
            ok = [f for f in ev if valid(ev[f], t)]
            b = min(ok, key=lambda f: ev[f]["err"]) if ok else min(ev, key=lambda f: abs(ev[f]["r_hat"] - t))
            step /= 2
            score(b - step)
            score(b + step)

    # ---- scan: one block at a time, baseline request
    scan = {}
    for b in BLOCKS:
        for s in SCALES:
            for t in Q.TARGETS:
                tag = f"scan_{b}_s{s:.2f}_t{t:.1f}"
                y = gen(t, tag, sc={b: s})
                for k, p in Q.CASES:
                    d = measure(y, t, k)
                    scan[(k, t, b, s)] = d
                    rows.append(dict(stage="scan", cell=k, patch=p, target=t, block=b, scale=s, factor=1.0, tag=tag, **d))
    print("scan done", flush=True)
    json.dump(dict(gate=gate, rows=rows), open(f"{OUT}/conds_partial.json", "w"))

    # ---- request-only search per case
    for k, p in Q.CASES:
        for t in Q.TARGETS:
            search(k, p, t, "input", None, "in")
    print("input-only done", flush=True)
    json.dump(dict(gate=gate, rows=rows), open(f"{OUT}/conds_partial.json", "w"))

    # ---- retune the TOP_K promising variants per case
    promising = {}
    for k, p in Q.CASES:
        for t in Q.TARGETS:
            cand = [(bs, d) for (kk, tt, *bs), d in scan.items() if kk == k and tt == t and d["focus_add"] <= Q.FOCUS_ADD_MAX]
            top = sorted(cand, key=lambda x: x[1]["err"])[:TOP_K]
            promising[f"{p}@{t:g}"] = [list(bs) for bs, _ in top]
            for (b, s), _ in top:
                search(k, p, t, "retune", {b: s}, f"rt_{b}_s{s:.2f}")
    print("retune done", flush=True)

    meta = dict(gate=gate, lora_layers=counts, blocks=BLOCKS, scales=SCALES, top_k=TOP_K, promising=promising,
                rows=rows, n_generations=n_gen[0], sec_per_gen=(time.time() - t0) / n_gen[0],
                curve_requests={f"{t:g}": curve.inverse(t)[0] for t in Q.TARGETS})
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    print("done:", n_gen[0], "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
