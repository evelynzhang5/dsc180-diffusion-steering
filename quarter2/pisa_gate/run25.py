"""Round 25 driver: replace PISA's optical gate with a cell-transfer gate for sub-token blur (inference only).

PISA (fp32 throughout) computes, per source token j and target token i:
    a_ij  proportional to  exp(q_i.k_j - max_i) * gate_ij,   normalized over targets i, then occlusion mask, dropout.
  G0 (original): gate_ij = sigmoid(ln 1e7 * (coc_j - 51.2 * dist_ij)) - essentially 0 for neighbours below ~16 px.
  G1 (this test): gate_ij = fraction of a uniform 16x16 source cell's light that a disc of radius r_j = 9.6875 * coc_j
                  carries into target cell i (disc convolved with both cells; sums to 1; equals G0's behaviour at
                  r = 0 and approximates an anti-aliased G0 for large r).
Everything else is identical (content similarity, per-source normalization, occlusion, dropout). The attention
function is re-implemented so only the gate differs; a check run with the original sigmoid gate in the
re-implementation must reproduce the untouched original exactly.
Design (patches from round 24; sources held out):
  dev:  12 dev patches, positions 0 and 3, 11 request levels (1-19 px), each gate -> C1 curve per gate (same budget)
  test: 12 test patches; raw: positions 0 and 3, same 11 levels; corrected: positions 0, 2, 4, targets 3,4,5,6,8,11.6,15.5
"""
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.signal import fftconvolve

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402
import PISA_attn_processor as PA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402
import patches as PT  # noqa: E402

OUT = os.environ["R25_OUT"]
LEVELS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 19]           # requested radius, px (K = 40, gap = r / 38.75 <= 0.5)
TARGETS = [3, 4, 5, 6, 8, 11.625, 15.5]
FLOOR = 1.5
R_STEP, R_MAX, NB = 0.25, 48.0, 4
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
STATE = dict(gate="G0", impl="orig", table=None, off=None)


def transfer_table():
    """W[r_index, dy+NB, dx+NB]: cell-to-cell light transfer for a disc of radius r (px), 16 px cells."""
    tri = np.r_[np.arange(1, 17), np.arange(15, 0, -1)].astype(float)
    A = np.outer(tri, tri)                                        # autocorrelation of a 16x16 cell, peak 256
    rs = np.arange(0, R_MAX + 1e-9, R_STEP)
    W = np.zeros((len(rs), 2 * NB + 1, 2 * NB + 1))
    for i, r in enumerate(rs):
        k = fftconvolve(A, m18.disk(r), mode="full")
        cy = k.shape[0] // 2
        for dy in range(-NB, NB + 1):
            for dx in range(-NB, NB + 1):
                y, x = cy + 16 * dy, cy + 16 * dx
                if 0 <= y < k.shape[0] and 0 <= x < k.shape[1]:
                    W[i, dy + NB, dx + NB] = k[y, x] / 256.0
    return torch.tensor(W.reshape(len(rs), -1), dtype=torch.float32)


def offsets(n=32):
    t = torch.arange(n * n)
    ry, rx = t // n, t % n
    dy = ry[:, None] - ry[None, :]                                # target i (rows) minus source j (cols)
    dx = rx[:, None] - rx[None, :]
    valid = (dy.abs() <= NB) & (dx.abs() <= NB)
    idx = ((dy + NB) * (2 * NB + 1) + (dx + NB)).clamp(0, (2 * NB + 1) ** 2 - 1)
    return idx, valid


def g1_gate(coc_keys):
    """coc_keys (S,) -> gate (S_targets, S_sources)"""
    tab = STATE["table"]
    r = (coc_keys * 9.6875).clamp(0, R_MAX) / R_STEP
    lo = r.floor().long().clamp(max=tab.shape[0] - 1)
    hi = (lo + 1).clamp(max=tab.shape[0] - 1)
    f = (r - lo.float())[:, None]
    Wk = tab[lo] * (1 - f) + tab[hi] * f                          # (S, 81) per source
    idx, valid = STATE["off"]
    G = torch.gather(Wk, 1, idx.t().contiguous()).t()             # G[i, j] = Wk[j, idx[i, j]]
    return G * valid


def reimpl(query, key, value, gate, occ_map, dropout_p):
    """PISA's attention with a given gate (same operations and order as the original, fp32)."""
    qk = query @ key.transpose(-2, -1) * (1 / math.sqrt(value.size(-1)))
    mask = occ_map.repeat(1, qk.shape[1], 1, 1)
    qk = torch.exp(qk - torch.max(qk, dim=-2, keepdim=True)[0])
    a = qk * gate
    a = a / torch.sum(a, dim=-2, keepdim=True)
    ad = torch.dropout(a * (1 - mask), dropout_p, train=True)
    return ad.to(value.dtype) @ value


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):
        q, k, v, w, dc, oc = (t.float() for t in (query, key, value, weight_matrix, disp_coc, occ_map))
        if STATE["impl"] == "orig":
            out = _orig_attn(q, k, v, w, dc, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                             scale=scale, hard=hard, train=train, occ_map=oc)
        else:
            if STATE["gate"] == "G0":
                gate = torch.sigmoid(hard * (dc[:, [1], None, :] - w))
            else:
                gate = g1_gate(dc[0, 1])[None, None]
            out = reimpl(q, k, v, gate, oc, dropout_p)
        return out.to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def groups(lib, split):
    ps = [p for p in lib if p["split"] == split]
    return [ps[0::2], ps[1::2]]


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    STATE["table"] = transfer_table().to(dev)
    STATE["off"] = tuple(t.to(dev) for t in offsets())
    os.makedirs(f"{OUT}/img", exist_ok=True)
    lib = PT.library()
    conds = []

    def run(tag, img_np, disp, coc, gate, impl="re", **info):
        STATE["gate"], STATE["impl"] = gate, impl
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
        conds.append(dict(tag=tag, file=f, gate=gate, impl=impl, seconds=dur, finite=bool(np.isfinite(a).all()),
                          black=bool(a.max() < 5), **info))
        return a

    gp = lambda r: r / (D.NOM * D.K)                               # request radius -> gap
    # ---- check: re-implementation with the original gate == untouched original
    img, cid, where = PT.layout(groups(lib, "test")[0], 0)
    disp, coc = PT.maps(cid, {k: PT.SIGN[k] * gp(10) for k in PT.FREE})
    a = run("check_orig", img, disp, coc, "G0", impl="orig", stage="check")
    b = run("check_reimpl_G0", img, disp, coc, "G0", impl="re", stage="check")
    check_diff = float(np.abs(a - b).max())
    print("check: re-implementation (G0) vs original, max |diff|", check_diff, flush=True)

    # ---- dev + raw test sweeps
    for split, stage in (("dev", "dev"), ("test", "raw")):
        for gi, grp in enumerate(groups(lib, split)):
            for rot in (0, 3):
                img, cid, where = PT.layout(grp, rot)
                for gate in ("G0", "G1"):
                    for r in LEVELS:
                        gaps = {k: PT.SIGN[k] * gp(r) for k in PT.FREE}
                        disp, coc = PT.maps(cid, gaps)
                        run(f"{stage}_g{gi}_r{rot}_{gate}_req{r}", img, disp, coc, gate, stage=stage, group=gi, rot=rot,
                            r_req=float(r), target=float(r), where={str(k): v for k, v in where.items()})
        print("sweep", split, "done", flush=True)

    # ---- calibrate on dev (per gate, same budget), frozen before corrected runs
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
                    rows.append(dict(gate=c["gate"], r_req=c["r_req"], r_hat=f["v"], at_cap=f["at_cap"], patch=where[k]))
    curves = {}
    for gate in ("G0", "G1"):
        rv = [r for r in rows if r["gate"] == gate]
        curves[gate] = M.Curve(np.array([r["r_req"] for r in rv]), np.array([48.0 if r["at_cap"] else r["r_hat"] for r in rv]), FLOOR)
    json.dump(dict(C1={g: c.as_dict() for g, c in curves.items()}, dev_rows=rows, floor=FLOOR), open(f"{OUT}/calib.json", "w"), indent=1)
    print("calibration frozen", {g: [(round(a, 1), round(b, 1)) for a, b in zip(c.x, c.y)] for g, c in curves.items()}, flush=True)

    # ---- corrected test runs
    for gi, grp in enumerate(groups(lib, "test")):
        for rot in (0, 2, 4):
            img, cid, where = PT.layout(grp, rot)
            for gate in ("G0", "G1"):
                for t in TARGETS:
                    r, status = curves[gate].inverse(t)
                    gaps = {k: ("coc", r / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE}
                    disp, coc = PT.maps(cid, gaps)
                    run(f"cor_g{gi}_r{rot}_{gate}_t{t:.1f}", img, disp, coc, gate, stage="corrected", group=gi, rot=rot,
                        r_req=float(r), target=float(t), status=status, where={str(k): v for k, v in where.items()})
        print("corrected group", gi, flush=True)
    img, cid, where = PT.layout(groups(lib, "test")[0], 0)
    disp, coc = PT.maps(cid, {k: PT.SIGN[k] * gp(5) for k in PT.FREE})
    a = run("repeat", img, disp, coc, "G1", stage="repeat")
    b = np.asarray(Image.open(f"{OUT}/img/raw_g0_r0_G1_req5.png")).astype(np.float32)
    meta = dict(conds=conds, check_max_abs_diff=check_diff, repeat_max_abs_diff=float(np.abs(a - b).max()),
                peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                seconds={g: float(np.median([c["seconds"] for c in conds if c["gate"] == g])) for g in ("G0", "G1")})
    json.dump(meta, open(f"{OUT}/conds.json", "w"))
    print("done:", len(conds), "outputs; repeat diff", meta["repeat_max_abs_diff"], "; s/img", meta["seconds"], flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
