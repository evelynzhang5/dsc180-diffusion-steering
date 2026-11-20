"""Round 35 driver: round 34's optimised activation edit with a preservation term, run to convergence.

Round 34 matched the teacher inside the case cells (up_blocks.1) but its loss constrained only those cells, and the
later layers spread the edit into the focus and the other cells. Here the target is the *regional-paste latent*
    Z* = Z0 + m (Z_T - Z0)       (Z0 = the student's own unedited final latent, m = case-cell area mask)
scored over the WHOLE latent (in-cell term toward the teacher + out-of-cell term toward the unedited student). Control
C2 (regional paste) already shows Z* decodes to a valid, preserved image. The edit stays confined to the case cells at
the location. Adaptive length: at least MIN_STEPS, at most MAX_STEPS; stop once the loss fell < 1 % over the last
10 % of steps (checked every 50). Everything else (checks, controls, request-only comparison, metric) as round 34.

Round 34 notes follow.

Teacher = round 33's: the 4 case cells already blurred to the target (sRGB disk blur pasted in) with a zero blur
request there; everything else as the student (round 25's baseline). Its *final denoised latent* Z_T is the signal.
BokehDiff takes one DDPM step from the clean input latent x:  z = (x - sqrt(1-a_t) eps) / sqrt(a_t), so the teacher's
UNet output is relative to its own (blurred) input and cannot be transplanted (round 33). Instead:

Checks (all before any optimisation; the run aborts on failure)
  G1 baseline reproduces round 25 exactly (both targets); repeat identical
  G2 the hand-computed x0 formula matches the scheduler's step output (max diff < 1e-3)
  G3 pass-through: teacher error (revised metric) <= 0.5 x request-only in >= 6/8 cases, teacher valid
  C1 positive control: replace the whole final latent with Z_T -> the teacher image exactly (max diff 0)
  C2 regional control: paste Z_T into the case cells only (area mask on the latent grid) -> valid, and error
     <= 0.5 x request-only in >= 6/8 cases (the target is attainable region-wise at the latent)
Optimised edit (privileged upper bound for any edit at a location; all 4 case cells jointly, one combined image)
  delta at location L (PISA block output, mid_block, up_blocks.0, up_blocks.1), confined to the case cells:
      h' = h + m * D * rms_m(h),   D optimised by Adam (lr 0.05, 150 steps) to minimise the masked MSE between
      the student's final latent and Z_T in the case cells. Final image decoded from the best D.
Request-only comparison: round 31's request search (192 generations) re-selected with the revised metric.
"""
import copy
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
import metrics34 as Q  # noqa: E402
import patches as PT  # noqa: E402

OUT = os.environ["R35_OUT"]
R25 = os.environ["R25_OUT"]
R31 = os.environ["R31_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
LOCS = ["post_pisa", "up0", "up1"]                  # mid_block converged in round 34 and cannot express the correction
STEPS = 5 if SMOKE else 1000                          # maximum; adaptive stop below
MIN_STEPS, STOP_DROP = (5, 0.0) if SMOKE else (200, 0.01)
LR = 0.05
CELLS = [k for k, _ in Q.CASES]
OTHER = [k for k in PT.FREE if k not in CELLS]
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
S = dict(mode=None, loc=None, D=None, rms=None, cap=None, inner=False, ZT=None, mlat=None, info=None)


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # round 25's fp32 PISA baseline
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn
MASK = {}


def edit(h, name):
    if S["mode"] != "opt" or S["loc"] != name:
        return h
    x = h.float()
    m = F.adaptive_avg_pool2d(MASK["case"], x.shape[-2:])
    if S["rms"] is None:                                  # first (unedited) pass: record the region's feature rms
        S["rms"] = float(torch.sqrt((m * x ** 2).sum() / (m.sum() * x.shape[1])))
        S["hnorm"] = float(torch.sqrt((m * x ** 2).sum()))
    if S["D"] is None:
        return h
    return (x + m * S["D"] * S["rms"]).to(h.dtype)


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
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    unet, sched = pipeline.unet, pipeline.scheduler
    unet.requires_grad_(False)
    print("scheduler", type(sched).__name__, "prediction", sched.config.prediction_type, flush=True)
    assert sched.config.prediction_type == "epsilon"
    mods = {"post_pisa": unet.down_blocks[1].attentions[0], "mid": unet.mid_block, "up0": unet.up_blocks[0],
            "up1": unet.up_blocks[1]}
    for n, mod in mods.items():
        mod.register_forward_hook(post_hook(n))

    def cap(module, args, kwargs):
        if not S["inner"]:
            S["cap"] = (args, kwargs)
    unet.register_forward_pre_hook(cap, with_kwargs=True)
    orig_step = sched.step

    def x0_of(eps, t, x):
        a = sched.alphas_cumprod.to(x.device)[int(t)].float()
        return (x.float() - torch.sqrt(1 - a) * eps.float()) / torch.sqrt(a)

    def step(noise_pred, t, latents, *a, **kw):
        if S["mode"] == "check":
            ref = orig_step(noise_pred, t, latents, *a, **{k: v for k, v in kw.items() if k != "generator"})[0]
            S["info"] = dict(x0_formula_maxdiff=float((x0_of(noise_pred, t, latents) - ref.float()).abs().max()))
        if S["mode"] == "opt":
            noise_pred = optimise(t, latents)
        out = orig_step(noise_pred, t, latents, *a, **kw)
        z = out[0]
        if S["mode"] == "teacher":
            S["ZT"] = z.detach().clone()
        if S["mode"] in ("replace_all", "replace_region"):
            m = torch.ones_like(S["mlat"]) if S["mode"] == "replace_all" else S["mlat"]
            z = (z + m * (S["ZT"] - z)).to(z.dtype)
            out = (z,) + tuple(out[1:])
        return out
    sched.step = step

    def optimise(t, latents):
        with torch.inference_mode(False):                # the pipeline runs under inference_mode
            return _optimise(t, latents)

    def cl(x):                                            # normal (non-inference) copies of captured inputs
        if isinstance(x, torch.Tensor):
            return x.clone()
        if isinstance(x, dict):
            return {k: cl(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(cl(v) for v in x)
        return x

    def _optimise(t, latents):
        args, kwargs = cl(S["cap"][0]), cl(S["cap"][1])
        latents = latents.clone()
        S["inner"] = True
        ml = S["mlat"].clone()
        tgt = S["ZT"].float().clone()
        S["D"], S["rms"] = None, None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            Z0 = x0_of(unet(*args, **kwargs)[0], t, latents).float().clone()   # unedited latent; records rms
        tgt = Z0 + ml * (tgt - Z0)                        # regional-paste latent (C2) as the whole-image target
        out_w = 1 - ml
        Dp = None
        best, losses, l_in, l_out, t1 = (float("inf"), None), [], [], [], time.time()
        torch.cuda.reset_peak_memory_stats()
        for i in range(STEPS + 1):
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.float16):
                if Dp is None:
                    Dp = torch.zeros(S["shape"], device=latents.device, requires_grad=True)
                    opt = torch.optim.Adam([Dp], lr=LR)
                S["D"] = Dp
                eps = unet(*args, **kwargs)[0]
                se = (x0_of(eps, t, latents) - tgt) ** 2
                loss = se.mean()
            lv = float(loss)
            losses.append(lv)
            l_in.append(float((se * ml).sum() / (ml.sum() * tgt.shape[1])))
            l_out.append(float((se * out_w).sum() / (out_w.sum() * tgt.shape[1])))
            if lv < best[0]:
                best = (lv, Dp.detach().clone(), l_in[-1], l_out[-1])
            n = max(1, len(losses) // 10)
            if i == STEPS or (i >= MIN_STEPS and i % 50 == 0
                              and (losses[-n - 1] - losses[-1]) / max(losses[-n - 1], 1e-12) < STOP_DROP):
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        S["D"] = best[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            eps = unet(*args, **kwargs)[0]
        S["inner"] = False
        Dm = (best[1] * F.adaptive_avg_pool2d(MASK["case"], best[1].shape[-2:]))[0]          # C x H x W
        X = Dm.flatten(1).T                                                                 # tokens x C
        X = X[X.abs().sum(1) > 0]
        sv = torch.linalg.svdvals(X.float()) ** 2
        S["info"] = dict(losses=losses, best_loss=best[0], loss0=losses[0], sec=time.time() - t1, steps=len(losses) - 1,
                         loss_in=l_in, loss_out=l_out, best_in=best[2], best_out=best[3],
                         peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                         delta_over_h=float(torch.sqrt((Dm ** 2).sum()) * S["rms"] / S["hnorm"]),
                         energy_top1=float(sv[0] / sv.sum()), energy_top5=float(sv[:5].sum() / sv.sum()))
        return eps

    # shapes of each location (needed for D) from a recording pass
    shapes = {}

    def shp(name):
        def fn(module, args, out):
            x = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else out.sample)
            shapes[name] = tuple(x.shape)
        return fn
    hs = [mod.register_forward_hook(shp(n)) for n, mod in mods.items()]

    img, cid, where = Q.scene()
    im = img.astype(np.float32)
    case = np.isin(cid, CELLS)
    MASK["case"] = torch.from_numpy(case.astype(np.float32))[None, None].to(dev)
    S["mlat"] = F.adaptive_avg_pool2d(MASK["case"], (D.S // 8, D.S // 8))
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

    def gen(t, tag, teacher=False, mode=None, loc=None):
        r0, _ = curve.inverse(t)
        g = {k: ("coc", r0 / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE}
        x = im
        if teacher:
            x = np.where(case[..., None], tgts[t], im)
            for k in CELLS:
                g[k] = ("coc", 0.0, PT.SIGN[k] * gp(t))
        disp, coc = PT.maps(cid, g)
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        x_in = torch.from_numpy(x / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        S["mode"], S["loc"], S["info"] = mode, loc, None
        if loc:
            S["shape"] = shapes[loc]
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x_in, disp_coc=dc, timeit=True)
        S["mode"], S["loc"], S["D"] = None, None, None
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        n_gen[0] += 1
        return a.astype(np.float32)

    valid = lambda d, t: abs(d["r_hat"] - t) <= m18.TOL(t) and d["focus_add"] <= Q.FOCUS_ADD_MAX

    def measure(y, t, k):
        return dict(err=Q.off_mae(y, tgts[t], ints[k]), err_old=Q.aff_mae(y, tgts[t], ints[k]),
                    shape=Q.shape_mae(y, tgts[t], ints[k]), contrast=Q.contrast(y, tgts[t], ints[k]),
                    r_hat=float(M.fit_fast(y[ints[k]], B[k])["v"]), focus_add=Q.raw_mae(y, im, fm) - foc0[t],
                    others_vs_base=Q.raw_mae(y, base[t], om))

    # ---- request-only comparison: round 31's search, re-selected with the revised metric
    r31 = json.load(open(f"{R31}/conds.json"))["rows"]
    req = {}
    for k, p in Q.CASES:
        for t in Q.TARGETS:
            cand = [r for r in r31 if r["cell"] == k and r["target"] == t and r["stage"] in ("input", "base")]
            sc = []
            for r in cand:
                if abs(r["r_hat"] - t) <= m18.TOL(t) and r["focus_add"] <= Q.FOCUS_ADD_MAX:
                    y = np.asarray(Image.open(f"{R31}/img/{r['tag']}.png")).astype(np.float32)
                    sc.append((Q.off_mae(y, tgts[t], ints[k]), r["tag"], r["r_hat"], Q.contrast(y, tgts[t], ints[k]),
                               r.get("others_vs_base", 0.0)))
            b = min(sc) if sc else None
            req[f"{p}@{t:g}"] = dict(err=b[0], tag=b[1], r_hat=b[2], contrast=b[3], others=b[4], n_valid=len(sc)) if b else None
    print("request-only (revised metric):", {k: (round(v["err"], 2), v["tag"]) if v else None for k, v in req.items()}, flush=True)

    # ---- checks
    gate, ok = {}, True
    for t in Q.TARGETS:
        base[t] = gen(t, f"base_t{t:.1f}", mode="check")
        gate[f"x0_formula_{t:g}"] = S["info"]["x0_formula_maxdiff"]
        foc0[t] = Q.raw_mae(base[t], im, fm)
        ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{t:.1f}.png")).astype(np.float32)
        gate[f"repro_r25_{t:g}"] = float(np.abs(base[t] - ref).max())
    for h in hs:
        h.remove()
    print("location shapes", shapes, flush=True)
    gate["repeat"] = float(np.abs(gen(11.625, "rep_t11.6") - base[11.625]).max())
    ok &= all(gate[f"repro_r25_{t:g}"] == 0 and gate[f"x0_formula_{t:g}"] < 1e-3 for t in Q.TARGETS) and gate["repeat"] == 0
    ctrl = {}
    for t in Q.TARGETS:
        teach = gen(t, f"teacher_t{t:.1f}", teacher=True, mode="teacher")
        ZT = S["ZT"]
        full = gen(t, f"ctrl_all_t{t:.1f}", mode="replace_all")
        gate[f"C1_replace_all_{t:g}"] = float(np.abs(full - teach).max())
        S["ZT"] = ZT
        reg = gen(t, f"ctrl_region_t{t:.1f}", mode="replace_region")
        for k, p in Q.CASES:
            key = f"{p}@{t:g}"
            a, b = measure(teach, t, k), measure(reg, t, k)
            rq = req[key]
            ctrl[key] = dict(teacher=a, region=b, request=rq,
                             G3=valid(a, t) and rq is not None and a["err"] <= 0.5 * rq["err"],
                             C2=valid(b, t) and rq is not None and b["err"] <= 0.5 * rq["err"])
        torch.save(ZT.cpu(), f"{OUT}/ZT_t{t:.1f}.pt")
    gate["G3_cases"] = sum(c["G3"] for c in ctrl.values())
    gate["C2_cases"] = sum(c["C2"] for c in ctrl.values())
    ok &= all(gate[f"C1_replace_all_{t:g}"] == 0 for t in Q.TARGETS) and gate["G3_cases"] >= 6 and gate["C2_cases"] >= 6
    print("gate", gate, "ok", ok, flush=True)
    for key, c in ctrl.items():
        print(f"  {key:18s} request {c['request']['err']:.2f} | teacher {c['teacher']['err']:.2f} r{c['teacher']['r_hat']:.1f}"
              f" | region-paste {c['region']['err']:.2f} r{c['region']['r_hat']:.1f}", flush=True)
    meta = dict(gate=gate, gate_ok=bool(ok), controls=ctrl, request=req, locs=LOCS, steps=STEPS, lr=LR, smoke=SMOKE,
                min_steps=MIN_STEPS, stop_drop=STOP_DROP)
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    if not ok:
        print("CHECKS FAILED: no optimisation run", flush=True)
        os._exit(0)

    # ---- optimised edits (all 4 case cells jointly; combined image)
    opt = {}
    for t in (Q.TARGETS[-1:] if SMOKE else Q.TARGETS):
        S["ZT"] = torch.load(f"{OUT}/ZT_t{t:.1f}.pt").to(dev)
        for loc in (LOCS[:1] if SMOKE else LOCS):
            y = gen(t, f"opt_{loc}_t{t:.1f}", mode="opt", loc=loc)
            info = S["info"]
            opt[f"{loc}@{t:g}"] = dict(info=info, cells={p: measure(y, t, k) for k, p in Q.CASES})
            print(f"opt {loc} t={t:g}: loss {info['loss0']:.4f} -> {info['best_loss']:.4f} (in {info['best_in']:.4f}, "
                  f"out {info['best_out']:.5f}) {info['steps']} steps in {info['sec']:.0f}s, "
                  f"peak {info['peak_mem_gb']:.1f} GB, |delta|/|h| {info['delta_over_h']:.2f}, top1 {info['energy_top1']:.2f} | "
                  + " ".join(f"{p} {d['err']:.2f} r{d['r_hat']:.1f}" for p, d in opt[f'{loc}@{t:g}']['cells'].items()), flush=True)
            meta["opt"] = opt
            json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    meta.update(n_generations=n_gen[0], minutes=(time.time() - t0) / 60)
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    print("done:", n_gen[0], "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
