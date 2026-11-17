"""Round 24 driver: content dependence of BokehDiff's delivered blur, a patch-statistics calibration, and an
intervention on PISA's content-similarity (query-key) weights.

Run from the BokehDiff repo directory (pinned commit), round 17 venv; started like rounds 18-23. PISA attention in
fp32 throughout. Attention variants (only the query-key logits are scaled; the optical gate, normalization,
occlusion mask and dropout are untouched):
  A0  tau = 1    (unchanged)
  A1  tau = 0.5  (content similarity weakened)
  A2  tau = 0    (content similarity removed: uniform within the optical support)
Stages:
  1. development: 12 dev patches (2 groups of 6) x 3 positions x 8 request levels (gap 0.15..0.50), each variant
  2. calibrations from dev measurements, frozen to calib.json before any test run:
       C1 per variant: monotone curve request -> measured radius (same budget for every variant)
       CP for A0:      log r = a + b log r_req + c . z(patch statistics)
  3. test (12 held-out-source patches, all 6 positions, targets 7.8 / 11.6 / 15.5 px):
       raw requests for A0/A1/A2; corrected requests A0-C1, A0-CP, A1-C1, A2-C1
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
import patches as PT  # noqa: E402

OUT = os.environ["R24_OUT"]
TAU = {"A0": 1.0, "A1": 0.5, "A2": 0.0}
DEV_GAPS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
DEV_ROTS = [0, 2, 4]
TEST_GAPS = [0.20, 0.30, 0.40]             # targets 7.75 / 11.6 / 15.5 px
FLOOR = 1.5                                 # curve anchor at request 0 (K = 0 softness, rounds 19/22)
REQ_MAX = D.NOM * D.K * 1.0                 # 38.75 px: requests are clamped here
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
STATE = dict(tau=1.0, stats=None, cellmap=None)


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    q = query.float() * STATE["tau"]
    if STATE["stats"] is not None:
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            gate = torch.sigmoid(hard * (disp_coc.float()[:, [1], None, :] - weight_matrix.float()))
            qk = q @ key.float().transpose(-2, -1) * (1 / np.sqrt(q.size(-1)))
            qk = torch.exp(qk - qk.max(dim=-2, keepdim=True)[0])
            a = qk * gate
            a = a / a.sum(dim=-2, keepdim=True)
            leave = (1 - torch.diagonal(a, dim1=-2, dim2=-1)).mean(1)[0]
            ent = (-(a * torch.log(a.clamp_min(1e-30))).sum(-2)).mean(1)[0]
            cm = STATE["cellmap"]
            STATE["stats"].append({str(c): dict(leave=float(leave[cm == c].mean()), entropy=float(ent[cm == c].mean()))
                                   for c in PT.FREE})
    with torch.autocast("cuda", enabled=False):
        return _orig_attn(q, key.float(), value.float(), weight_matrix.float(), disp_coc.float(), attn_mask=attn_mask,
                          dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard, train=train,
                          occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def groups(lib, split):
    ps = [p for p in lib if p["split"] == split]
    return [ps[0::2], ps[1::2]]                 # one patch per source in each group


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    lib = PT.library()
    conds = []

    def run(tag, img_np, cid, disp, coc, variant, stats=False, **info):
        STATE["tau"] = TAU[variant]
        STATE["stats"] = [] if stats else None
        tok = np.array([[cid[8 + 16 * r, 8 + 16 * c] for c in range(32)] for r in range(32)]).reshape(-1)
        STATE["cellmap"] = torch.from_numpy(tok).to(dev)
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
        conds.append(dict(tag=tag, file=f, variant=variant, seconds=dur, finite=bool(np.isfinite(a).all()),
                          black=bool(a.max() < 5), attn_stats=STATE["stats"], **info))
        return a

    # ---- stage 1: development
    dev_layouts = {}
    for gi, grp in enumerate(groups(lib, "dev")):
        for rot in DEV_ROTS:
            img, cid, where = PT.layout(grp, rot)
            dev_layouts[(gi, rot)] = (img, cid, where)
            for v in TAU:
                for g in DEV_GAPS:
                    gaps = {k: PT.SIGN[k] * g for k in PT.FREE}
                    disp, coc = PT.maps(cid, gaps)
                    run(f"dev_g{gi}_r{rot}_{v}_gap{g:.2f}", img, cid, disp, coc, v, stage="dev", group=gi, rot=rot,
                        where=where, cells={str(k): dict(patch=where[k], gap=gaps[k], r_req=D.NOM * D.K * g) for k in PT.FREE})
    print("stage 1 done", len(conds), flush=True)

    # ---- stage 2: measure dev, fit calibrations
    rows = []
    pstats = {p["id"]: p["stats"] for p in lib}
    for (gi, rot), (img, cid, where) in dev_layouts.items():
        st = m18.stack(img.astype(np.float32), "disk")
        Bs = {k: M.stack_at(st, D.interior(cid, k)) for k in PT.FREE}
        ints = {k: D.interior(cid, k) for k in PT.FREE}
        del st
        for c in [c for c in conds if c["stage"] == "dev" and c["group"] == gi and c["rot"] == rot]:
            y = np.asarray(Image.open(f"{OUT}/{c['file']}")).astype(np.float32)
            for k in PT.FREE:
                f = M.fit_fast(y[ints[k]], Bs[k])
                cc = c["cells"][str(k)]
                rows.append(dict(variant=c["variant"], patch=cc["patch"], cell=k, r_req=cc["r_req"], r_hat=f["v"],
                                 at_cap=f["at_cap"]))
    zkeys = ["log_std", "log_grad", "hf_ratio"]
    devp = sorted({r["patch"] for r in rows})
    zmu = {k: float(np.mean([pstats[p][k] for p in devp])) for k in zkeys}
    zsd = {k: float(np.std([pstats[p][k] for p in devp]) + 1e-9) for k in zkeys}
    z = lambda pid: np.array([(pstats[pid][k] - zmu[k]) / zsd[k] for k in zkeys])
    curves = {}
    for v in TAU:
        rv = [r for r in rows if r["variant"] == v]
        x = np.array([r["r_req"] for r in rv])
        y = np.array([48.0 if r["at_cap"] else r["r_hat"] for r in rv])
        curves[v] = M.Curve(x, y, FLOOR)
    ra = [r for r in rows if r["variant"] == "A0" and not r["at_cap"]]
    X = np.stack([np.ones(len(ra)), np.log([r["r_req"] for r in ra])] + [np.array([z(r["patch"])[i] for r in ra]) for i in range(3)], 1)
    beta, *_ = np.linalg.lstsq(X, np.log([r["r_hat"] for r in ra]), rcond=None)
    calib = dict(C1={v: c.as_dict() for v, c in curves.items()}, CP=dict(beta=beta.tolist(), zkeys=zkeys, zmu=zmu, zsd=zsd),
                 dev_rows=rows, floor=FLOOR)
    json.dump(calib, open(f"{OUT}/calib.json", "w"), indent=1)
    print("calibrations frozen; CP beta", np.round(beta, 3).tolist(), flush=True)

    def req_C1(v, t):
        r, st_ = curves[v].inverse(t)
        return min(r, REQ_MAX), st_

    def req_CP(pid, t):
        b = beta
        r = float(np.exp((np.log(t) - b[0] - b[2:] @ z(pid)) / b[1]))
        return min(max(r, 0.0), REQ_MAX), ("clamped" if r > REQ_MAX else "ok")

    # ---- stage 3: test
    for gi, grp in enumerate(groups(lib, "test")):
        for rot in range(6):
            img, cid, where = PT.layout(grp, rot)
            for g in TEST_GAPS:
                t = D.NOM * D.K * g
                for v in TAU:            # raw requests
                    gaps = {k: PT.SIGN[k] * g for k in PT.FREE}
                    disp, coc = PT.maps(cid, gaps)
                    run(f"test_g{gi}_r{rot}_{v}_raw_t{t:.1f}", img, cid, disp, coc, v, stats=(rot == 0), stage="test",
                        method=f"{v}-raw", group=gi, rot=rot, target=t, where=where,
                        cells={str(k): dict(patch=where[k], gap=gaps[k], r_req=t, status="raw") for k in PT.FREE})
                for meth, v in (("A0-C1", "A0"), ("A0-CP", "A0"), ("A1-C1", "A1"), ("A2-C1", "A2")):
                    gaps, cells = {}, {}
                    for k in PT.FREE:
                        r, st_ = req_CP(where[k], t) if meth == "A0-CP" else req_C1(v, t)
                        gaps[k] = ("coc", r / (D.NOM * 10), PT.SIGN[k] * g)
                        cells[str(k)] = dict(patch=where[k], gap=PT.SIGN[k] * g, r_req=r, status=st_)
                    disp, coc = PT.maps(cid, gaps)
                    run(f"test_g{gi}_r{rot}_{meth}_t{t:.1f}", img, cid, disp, coc, v, stage="test", method=meth,
                        group=gi, rot=rot, target=t, where=where, cells=cells)
        print("stage 3 group", gi, flush=True)
    # repeat
    img, cid, where = PT.layout(groups(lib, "test")[0], 0)
    gaps = {k: PT.SIGN[k] * 0.3 for k in PT.FREE}
    disp, coc = PT.maps(cid, gaps)
    a = run("repeat", img, cid, disp, coc, "A0", stage="repeat")
    b = np.asarray(Image.open(f"{OUT}/img/test_g0_r0_A0_raw_t11.6.png")).astype(np.float32)
    meta = dict(conds=conds, repeat_max_abs_diff=float(np.abs(a - b).max()), peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                mean_seconds=float(np.mean([c["seconds"] for c in conds])))
    json.dump(meta, open(f"{OUT}/conds.json", "w"))
    print("done:", len(conds), "outputs; repeat diff", meta["repeat_max_abs_diff"], flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
