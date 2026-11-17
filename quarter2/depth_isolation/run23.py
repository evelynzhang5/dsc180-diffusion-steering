"""Round 23 driver: does the disparity map change delivered blur when the full blur map is held fixed?

Run from the BokehDiff repo directory (pinned commit), round 17 venv; started like rounds 18-22. PISA attention in
fp32 throughout (as round 22). In BokehDiff's code the disparity channel enters only PISA's occlusion test; the
wrapper can replace that occlusion mask:
  occ "native"  the mask PISA computes from the given disparity map
  occ "frozen"  the mask recorded for the true-geometry run (V1) of the same scene / blur map / call
  occ "zero"    no occlusion
Every input is logged by hash (image, full disparity+blur tensor, occlusion mask per PISA call).
Dropout-free recomputation per cell: attention mass leaving a key's own token, and mass removed by occlusion.
"""
import hashlib
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

OUT = os.environ["R23_OUT"]
SCENES = ["S3", "S6"]
BLURMAPS = [("R2", "P1"), ("R3", "P1")]          # d_f = 0.5; R2 requests 7.8/9.7/11.6 px, R3 13.6/15.5/17.4, anchors 19.4
_orig_val = I.log_validation
_orig_attn = P.customized_scaled_dot_product_attention
STATE = dict(occ="native", store=None, record=None, call=0, cellmap=None, stats=None, hashes=[])
h = lambda t: hashlib.sha1(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def variant(d, d_f, v):
    """depth variants with the blur map unchanged"""
    if v == "V1_true":
        return d
    if v == "V2_flat":
        return np.full_like(d, 0.5)
    c = {"V3_half_c0.5": 0.5, "V4_half_c0.3": 0.3, "V5_half_c0.7": 0.7}.get(v)
    if c is not None:
        return (c + 0.5 * (d - d_f)).astype(np.float32)
    if v == "V6_max0.99":
        return np.minimum(d, 0.99).astype(np.float32)
    raise ValueError(v)


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    k = STATE["call"]
    STATE["call"] += 1
    if STATE["record"] is not None:
        STATE["record"][k] = occ_map.detach().clone()
    if STATE["occ"] == "frozen":
        occ_map = STATE["store"][k]
    elif STATE["occ"] == "zero":
        occ_map = torch.zeros_like(occ_map)
    STATE["hashes"].append(dict(call=k, occ=h(occ_map), coc_tokens=h(disp_coc), occ_mean=float(occ_map.float().mean()),
                                occ_nonfinite=int((~torch.isfinite(occ_map.float())).sum())))
    if STATE["stats"] is not None:
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            q, kk, w, dc, oc = query.float(), key.float(), weight_matrix.float(), disp_coc.float(), occ_map.float()
            gate = torch.sigmoid(hard * (dc[:, [1], None, :] - w))
            qk = q @ kk.transpose(-2, -1) * (1 / np.sqrt(q.size(-1)))
            qk = torch.exp(qk - qk.max(dim=-2, keepdim=True)[0])
            a = qk * gate
            a = a / a.sum(dim=-2, keepdim=True)                       # per key j, over targets i
            m = oc.repeat(1, a.shape[1], 1, 1)
            leave = (1 - torch.diagonal(a, dim1=-2, dim2=-1)).mean(1)[0]
            removed = (a * m).sum(-2).mean(1)[0]
            cm = STATE["cellmap"]
            st = {}
            for cell in range(9):
                sel = cm == cell
                st[cell] = dict(leave=float(leave[sel].mean()), removed_by_occlusion=float(removed[sel].mean()),
                                occ_mask_mean=float(oc[0, 0][:, sel].mean()))
            STATE["stats"].append(dict(call=k, cells=st))
    with torch.autocast("cuda", enabled=False):
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


P.customized_scaled_dot_product_attention = attn


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    conds = []

    def run(scene, tag, disp, coc, occ="native", record=False, **info):
        STATE.update(occ=occ, call=0, record={} if record else None, stats=[], hashes=[])
        s = D.make(scene)
        img = torch.from_numpy(s["image"].astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=img, disp_coc=dc, timeit=True)
        f = f"img/{scene}_{tag}.png"
        image.save(f"{OUT}/{f}")
        a = np.asarray(image).astype(np.float32)
        conds.append(dict(scene=scene, tag=tag, file=f, occ=occ, seconds=time.time() - t0, input_image_hash=h(img),
                          input_disp_coc_hash=h(dc), disp_range=[float(disp.min()), float(disp.max())],
                          pisa_calls=list(STATE["hashes"]), stats=list(STATE["stats"]),
                          finite=bool(np.isfinite(a).all()), black=bool(a.max() < 5), **info))
        return a, (STATE["record"] if record else None)

    for n in SCENES:
        cid = D.make(n)["cid"]
        tok = np.array([[cid[8 + 16 * r, 8 + 16 * c] for c in range(32)] for r in range(32)]).reshape(-1)
        STATE["cellmap"] = torch.from_numpy(tok).to(dev)
        for run_, perm in BLURMAPS:
            d_true, coc = D.maps(cid, run_, perm)
            d_f = D.RUNS[run_][0]
            base = dict(run=run_, perm=perm)
            a1, rec = run(n, f"{run_}{perm}_V1_true_native", d_true, coc, record=True, variant="V1_true", **base)
            store = rec
            a1b, _ = run(n, f"{run_}{perm}_V1_true_native_repeat", d_true, coc, variant="V1_true", kind="repeat", **base)
            conds[-1]["repeat_max_abs_diff"] = float(np.abs(a1 - a1b).max())
            run(n, f"{run_}{perm}_V1_true_zero", d_true, coc, occ="zero", variant="V1_true", **base)
            for v in ["V2_flat", "V3_half_c0.5", "V4_half_c0.3", "V5_half_c0.7", "V6_max0.99"]:
                dv = variant(d_true, d_f, v)
                run(n, f"{run_}{perm}_{v}_native", dv, coc, variant=v, **base)
                STATE["store"] = store
                run(n, f"{run_}{perm}_{v}_frozen", dv, coc, occ="frozen", variant=v, **base)
            print("done", n, run_, perm, flush=True)
    json.dump(dict(conds=conds, peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 1e9), open(f"{OUT}/conds.json", "w"))
    print("done:", len(conds), "outputs", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
