"""Round 26 driver: attention-spread steering in PISA (inference only, training-free).

PISA (fp32) per source token j, target i:  w_ij = exp(q_i.k_j - max_i) * gate_ij ;  a_ij = w_ij / sum_i w_ij.
The share of j's attention that leaves its own token is L_j = O_j / (w_jj + O_j), O_j = sum_{i != j} w_ij.
Steering sets L_j to a chosen target m for every blurred source token, per head, in closed form:
    w_ij <- s_j * w_ij for i != j,   s_j = m * w_jj / ((1 - m) * O_j)
Content similarity still decides WHICH neighbours receive the spread; steering fixes HOW MUCH.
The blur map of steered cells is a fixed reference radius (R_REF, which shapes the neighbourhood); the occlusion
mask, dropout and the rest of the model are unchanged; the focus cell and the anchors are not steered.

Design (round 24/25 patches, whole sources held out):
  dev:   12 dev patches x positions 0, 3 x 11 spread targets m -> one monotone curve m -> delivered radius (frozen)
  test:  raw sweep (same 11 m, positions 0, 3) for consistency at a fixed knob;
         corrected: targets 3, 4, 5, 6, 8, 11.6, 15.5 px via m*(t), positions 0, 2, 4
  baseline (reproduced): round 25's fp32 + original gate + its frozen radius curve, same corrected runs
"""
import json
import math
import os
import sys

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
import patches as PT  # noqa: E402

OUT = os.environ["R26_OUT"]
R25 = os.environ["R25_OUT"]
M_LEVELS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40]
TARGETS = [3, 4, 5, 6, 8, 11.625, 15.5]
R_REF = 12.0
FLOOR = 1.5
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
STATE = dict(mode="base", m_tok=None, stats=None)


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):
        q, k, v, w_, dc, oc = (t.float() for t in (query, key, value, weight_matrix, disp_coc, occ_map))
        if STATE["mode"] == "base":
            return _orig_attn(q, k, v, w_, dc, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, hard=hard, train=train, occ_map=oc).to(value.dtype)
        gate = torch.sigmoid(hard * (dc[:, [1], None, :] - w_))
        qk = q @ k.transpose(-2, -1) * (1 / math.sqrt(v.size(-1)))
        w = torch.exp(qk - torch.max(qk, dim=-2, keepdim=True)[0]) * gate          # (1, h, L targets, S sources)
        m = STATE["m_tok"]                                                         # (S,) target leave share, nan = not steered
        steer = ~torch.isnan(m)
        wjj = torch.diagonal(w, dim1=-2, dim2=-1)                                  # (1, h, S)
        O = w.sum(-2) - wjj
        mm = torch.nan_to_num(m, nan=0.0)[None, None]
        s = torch.where(steer[None, None], mm * wjj / ((1 - mm) * O).clamp_min(1e-30), torch.ones_like(wjj))
        eye = torch.eye(w.shape[-1], device=w.device, dtype=torch.bool)
        w = torch.where(eye, w, w * s[:, :, None, :])
        a = w / w.sum(-2, keepdim=True)
        if STATE["stats"] is not None:
            leave = (1 - torch.diagonal(a, dim1=-2, dim2=-1))[0]                    # (h, S)
            err = (leave - torch.nan_to_num(m, nan=0.0)[None]).abs()[:, steer]
            STATE["stats"].append(dict(median_abs_err=float(err.median()), max_abs_err=float(err.max()),
                                       n_steered=int(steer.sum())))
        mask = oc.repeat(1, a.shape[1], 1, 1)
        ad = torch.dropout(a * (1 - mask), dropout_p, train=True)
        return (ad @ v).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def groups(lib, split):
    ps = [p for p in lib if p["split"] == split]
    return [ps[0::2], ps[1::2]]


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    lib = PT.library()
    conds = []
    gp = lambda r: r / (D.NOM * D.K)

    def run(tag, img_np, cid, disp, coc, mode, m=None, stats=False, **info):
        STATE["mode"] = mode
        STATE["stats"] = [] if stats else None
        if mode == "steer":
            tok = np.array([[cid[8 + 16 * r, 8 + 16 * c] for c in range(32)] for r in range(32)]).reshape(-1)
            mt = np.where(np.isin(tok, PT.FREE), m, np.nan).astype(np.float32)
            STATE["m_tok"] = torch.from_numpy(mt).to(dev)
        img = torch.from_numpy(img_np.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                   aif_image=img, disp_coc=dc, timeit=True)
        f = f"img/{tag}.png"
        image.save(f"{OUT}/{f}")
        a = np.asarray(image).astype(np.float32)
        conds.append(dict(tag=tag, file=f, mode=mode, m=m, seconds=dur, finite=bool(np.isfinite(a).all()),
                          black=bool(a.max() < 5), steer_stats=STATE["stats"], **info))
        return a

    def steer_maps(cid, t_geom):
        """disparity = geometry of the requested blur (as the baseline); blur map of free cells = R_REF"""
        gaps = {k: ("coc", R_REF / (D.NOM * 10), PT.SIGN[k] * gp(t_geom)) for k in PT.FREE}
        return PT.maps(cid, gaps)

    # ---- baseline: reproduce round 25's corrected runs with its frozen curve
    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    base_curve = M.Curve.__new__(M.Curve)
    base_curve.x, base_curve.y = np.array(c25["x"]), np.array(c25["y"])
    repro = []
    for gi, grp in enumerate(groups(lib, "test")):
        for rot in (0, 2, 4):
            img, cid, where = PT.layout(grp, rot)
            for t in TARGETS:
                r, status = base_curve.inverse(t)
                disp, coc = PT.maps(cid, {k: ("coc", r / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE})
                tag = f"cor_g{gi}_r{rot}_base_t{t:.1f}"
                a = run(tag, img, cid, disp, coc, "base", stage="corrected", method="base", group=gi, rot=rot,
                        target=float(t), r_req=float(r), where={str(k): v for k, v in where.items()})
                ref = f"{R25}/img/cor_g{gi}_r{rot}_G0_t{t:.1f}.png"
                repro.append(float(np.abs(a - np.asarray(Image.open(ref)).astype(np.float32)).max()))
    print("baseline reproduces round 25: max |diff|", max(repro), flush=True)

    # ---- dev and raw test sweeps over the spread target m
    for split, stage in (("dev", "dev"), ("test", "raw")):
        for gi, grp in enumerate(groups(lib, split)):
            for rot in (0, 3):
                img, cid, where = PT.layout(grp, rot)
                disp, coc = steer_maps(cid, R_REF)
                for m in M_LEVELS:
                    run(f"{stage}_g{gi}_r{rot}_m{m:.2f}", img, cid, disp, coc, "steer", m=m, stats=(rot == 0),
                        stage=stage, method="steer", group=gi, rot=rot, where={str(k): v for k, v in where.items()})
        print("sweep", split, "done", flush=True)

    # ---- frozen steering curve m -> delivered radius (dev)
    rows = []
    for gi, grp in enumerate(groups(lib, "dev")):
        for rot in (0, 3):
            img, cid, where = PT.layout(grp, rot)
            st = m18.stack(img.astype(np.float32), "disk")
            ints = {k: D.interior(cid, k) for k in PT.FREE}
            Bs = {k: M.stack_at(st, ints[k]) for k in PT.FREE}
            del st
            for c in [c for c in conds if c.get("stage") == "dev" and c["group"] == gi and c["rot"] == rot]:
                y = np.asarray(Image.open(f"{OUT}/{c['file']}")).astype(np.float32)
                for k in PT.FREE:
                    f = M.fit_fast(y[ints[k]], Bs[k])
                    rows.append(dict(m=c["m"], r_hat=f["v"], at_cap=f["at_cap"], patch=where[k]))
    curve = M.Curve(np.array([r["m"] for r in rows]), np.array([48.0 if r["at_cap"] else r["r_hat"] for r in rows]), FLOOR)
    json.dump(dict(steer_curve=curve.as_dict(), base_curve=c25, dev_rows=rows, R_REF=R_REF), open(f"{OUT}/calib.json", "w"), indent=1)
    print("steering curve frozen", [(round(a, 3), round(b, 1)) for a, b in zip(curve.x, curve.y)], flush=True)

    # ---- steered corrected runs
    for gi, grp in enumerate(groups(lib, "test")):
        for rot in (0, 2, 4):
            img, cid, where = PT.layout(grp, rot)
            for t in TARGETS:
                m, status = curve.inverse(t)
                disp, coc = steer_maps(cid, t)
                run(f"cor_g{gi}_r{rot}_steer_t{t:.1f}", img, cid, disp, coc, "steer", m=float(min(m, 0.95)), stats=(rot == 0),
                    stage="corrected", method="steer", group=gi, rot=rot, target=float(t), status=status,
                    where={str(k): v for k, v in where.items()})
        print("steered corrected group", gi, flush=True)
    img, cid, where = PT.layout(groups(lib, "test")[0], 0)
    disp, coc = steer_maps(cid, R_REF)
    a = run("repeat", img, cid, disp, coc, "steer", m=0.10, stage="repeat")
    b = np.asarray(Image.open(f"{OUT}/img/raw_g0_r0_m0.10.png")).astype(np.float32)
    meta = dict(conds=conds, base_repro_max_diff=max(repro), repeat_max_abs_diff=float(np.abs(a - b).max()),
                peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                seconds={mo: float(np.median([c["seconds"] for c in conds if c["mode"] == mo])) for mo in ("base", "steer")})
    json.dump(meta, open(f"{OUT}/conds.json", "w"))
    print("done:", len(conds), "outputs; repeat", meta["repeat_max_abs_diff"], "; s/img", meta["seconds"], flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
