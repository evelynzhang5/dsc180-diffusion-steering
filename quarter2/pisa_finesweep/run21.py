"""Round 21 driver: fine 1x sweep of blur strength K around the onset (requested 4.8-6.8 px in 0.1 px steps).
One scene (S3), fixed geometry (d_f = 0.5, anchors +-0.5, swept pairs at gap 0.15); only K changes.
PISA attention in the shipped fp16 path and (secondary) fp32, wrapper from round 20.

(Round 20 notes follow.)

Run from the BokehDiff repo directory (pinned commit), round 17 venv; started like rounds 18-19 (main() builds the
pipeline, its first log_validation call is intercepted).

PISA's attention function (PISA_attn_processor.customized_scaled_dot_product_attention: sigmoid gate, exp,
per-key normalization, dropout, accumulation) is wrapped:
  mode "fp16": the original call, unchanged (as in rounds 17-19)
  mode "fp32": the same original function called with query, key, value, distance matrix, blur map and occlusion
               mask cast to float32 and autocast disabled; output cast back. Everything else stays fp16.
Known limit: the distance matrix and blur map are already fp16 when PISA hands them over (rounding ~5e-4 relative),
so fp32 mode starts from fp16-rounded distances; gate arguments shift by < 0.02, small next to the 0-vs-nonzero
underflow under test.
The wrapper also records, without using the random generator: call count, tensor shapes and dtypes, time, and
(dropout-free recomputation) the share of each key's attention mass that leaves its own token, by blur-map value.
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
import PISA_attn_processor as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402

OUT = os.environ["R21_OUT"]
SCENE = "S3"
GAP = 0.15
KS = [round(33.0 + 0.7 * i, 1) for i in range(21)]     # 33.0 ... 47.0; r_nom = NOM * K * 0.15 = 4.79 ... 6.83 px
MODES, SEEDS = ["fp16", "fp32"], [0, 1]
_orig_val = I.log_validation
_orig_attn = P.customized_scaled_dot_product_attention
STATE = dict(mode="fp16", log=[])


def _spread_stats(q, k, w, dc, occ, hard, dtype):
    """dropout-free recomputation of PISA's normalized attention; per key: 1 - weight kept on its own token."""
    with torch.no_grad():
        q, k, w, dc, occ = (t.to(dtype) for t in (q, k, w, dc, occ))
        gate = torch.sigmoid(hard * (dc[:, [1], None, :] - w))
        qk = q @ k.transpose(-2, -1) * (1 / np.sqrt(q.size(-1)))
        qk = torch.exp(qk - qk.max(dim=-2, keepdim=True)[0])
        a = qk * gate
        a = a / a.sum(dim=-2, keepdim=True)
        a = a * (1 - occ.repeat(1, a.shape[1], 1, 1))
        leave = 1 - torch.diagonal(a, dim1=-2, dim2=-1).float().mean(1)[0]          # (S,) averaged over heads
        nz = (a != 0).float().sum(-2).mean(1)[0] - 1                                   # nonzero targets besides self
        coc = dc[0, 1].float()
        out = {}
        for v in torch.unique(torch.round(coc * 1000) / 1000).tolist():
            m = torch.isclose(coc, torch.tensor(v, device=coc.device), atol=6e-4)
            out[f"{v:.3f}"] = dict(n=int(m.sum()), leave=float(leave[m].mean()), nonzero_targets=float(nz[m].mean()))
        return out


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    t0 = time.time()
    rec = dict(mode=STATE["mode"], q_shape=list(query.shape), q_dtype=str(query.dtype), w_dtype=str(weight_matrix.dtype),
               coc_dtype=str(disp_coc.dtype), occ_dtype=str(occ_map.dtype), autocast=torch.is_autocast_enabled())
    if STATE["mode"] == "fp16":
        out = _orig_attn(query, key, value, weight_matrix, disp_coc, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, scale=scale, hard=hard, train=train, occ_map=occ_map)
        sdt = query.dtype
    else:
        with torch.autocast("cuda", enabled=False):
            out = _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                             attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                             train=train, occ_map=occ_map.float()).to(value.dtype)
        sdt = torch.float32
    torch.cuda.synchronize()
    rec["seconds"] = time.time() - t0
    rec["out_dtype"] = str(out.dtype)
    if STATE.get("stats"):
        with torch.autocast("cuda", enabled=(sdt != torch.float32)):
            rec["spread"] = _spread_stats(query, key, weight_matrix, disp_coc, occ_map, hard, sdt)
    STATE["log"].append(rec)
    return out


P.customized_scaled_dot_product_attention = attn


def sweep_maps(cid, gap, K):
    """d_f = 0.5; anchors (pair 0) at +-0.5 keep the map spanning [0, 1]; pairs 1-3 all at +-gap (P1 orientation)."""
    disp = np.full((D.S, D.S), 0.5, np.float32)
    coc = np.zeros((D.S, D.S), np.float32)
    for p, (a, b) in enumerate(D.PAIRS):
        g = 0.5 if p == 0 else gap
        for rc, s in ((a, +1), (b, -1)):
            m = cid == 3 * rc[0] + rc[1]
            disp[m] = 0.5 + s * g
            coc[m] = g * K / 10
    return disp, coc


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    conds = []

    def run(scene, tag, disp, coc, mode, sd=0, stats=False, **info):
        STATE["mode"], STATE["stats"] = mode, stats
        n0 = len(STATE["log"])
        s = D.make(scene)
        img = torch.from_numpy(s["image"].astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, sd, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=img, disp_coc=dc, timeit=True)
        torch.cuda.synchronize()
        sec = time.time() - t0
        f = f"img/{scene}_{tag}.png"
        image.save(f"{OUT}/{f}")
        a = np.asarray(image).astype(np.float32)
        calls = STATE["log"][n0:]
        conds.append(dict(scene=scene, tag=tag, file=f, mode=mode, seed=sd, seconds=sec, pisa_calls=len(calls),
                          pisa_seconds=float(sum(c["seconds"] for c in calls)), finite=bool(np.isfinite(a).all()),
                          black=bool(a.max() < 5), **info))
        return a

    # gate: fp16 mode at K = 40 must reproduce round 20's S3 gap 0.15 fp16 seed 0 output exactly
    cid = D.make(SCENE)["cid"]
    disp, coc = sweep_maps(cid, GAP, 40.0)
    a = run(SCENE, "gate_K40_fp16", disp, coc, "fp16", kind="gate")
    ref = f"{os.environ['R20_IMG']}/S3_g0.1500_fp16_s0.png"
    gate_diff = float(np.abs(a - np.asarray(Image.open(ref)).astype(np.float32)).max()) if os.path.exists(ref) else None
    print("gate: K 40 fp16 vs round 20 output, max |diff|", gate_diff, flush=True)

    for mode in MODES:
        for sd in SEEDS:
            d0, c0 = sweep_maps(cid, GAP, 0.0)
            run(SCENE, f"K0_{mode}_s{sd}", d0, c0, mode, sd, kind="K0", K=0.0)
            for K in KS:
                disp, coc = sweep_maps(cid, GAP, K)
                run(SCENE, f"K{K:.1f}_{mode}_s{sd}", disp, coc, mode, sd, stats=(sd == 0), kind="sweep", K=K,
                    r_nom=D.NOM * K * GAP)
            print("sweep", mode, "seed", sd, flush=True)
    disp, coc = sweep_maps(cid, GAP, 40.0)
    a = run(SCENE, "K40.0_fp16_s0_repeat", disp, coc, "fp16", 0, kind="repeat", K=40.0)
    b = np.asarray(Image.open(f"{OUT}/img/{SCENE}_K40.0_fp16_s0.png")).astype(np.float32)
    meta = dict(conds=conds, gate_max_abs_diff=gate_diff, repeat_max_abs_diff=float(np.abs(a - b).max()),
                pisa_log=[{k: v for k, v in c.items()} for c in STATE["log"]],
                peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 1e9, scene=SCENE, gap=GAP, Ks=KS)
    json.dump(meta, open(f"{OUT}/conds.json", "w"))
    print("done:", len(conds), "outputs; PISA calls per image", conds[-1]["pisa_calls"], "; shapes",
          {(c["q_shape"].__str__(), c["q_dtype"], c["mode"]) for c in STATE["log"]}, flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
