"""Round 36 driver: does an inner product read depth, does an edit control blur, does adapting its strength help?

Mentor's proposal, in testable form:   d_hat_p = w_d . h_p + b   (read depth at location p)
                                       h'_p    = h_p + a_p u      (local edit; a_p = strength, u = direction)
Location: up_blocks.0 output (1280 x 32 x 32), preselected (round 32's blur-amount direction lives there).
Configuration: round 25's baseline (fp32 PISA, frozen radius curve C1 G0, seed 0). Scenes: scenes36 (12 development
scenes from 6 development source images, 12 test scenes from 6 fresh source images; depth, texture, position and focus
setting vary independently; equal requested blur at different depths in every scene).
Request = calibrated (curve inverse of the nominal radius r* = NOM K |gap|) in every generation.

Order (the probe and the strength rule are fitted on development scenes only and frozen before any test generation):
  checks  G1 round 25 baseline reproduced exactly with the hooks attached; G2 a repeat is identical
  dev     every dev scene x a in GRID (u = round 32's up0 size direction, same a in all 6 free cells); a = 0 runs
          also save the up0 activations
  fit     (1) ridge probe d_hat = w.h + b on dev a = 0 interior tokens (label = assigned disparity), lambda by
          leave-one-source-out CV; also probes for |gap| (requested blur) and signed gap, reported only
          (2) strength rule g(signed gap) = piecewise-linear through the median best dev a per signed gap
          (best = lowest revised error among valid; else radius closest to r*); fixed a = grid value with the
          lowest dev median ratio to a = 0
  D0      disparity-only sensitivity: test scenes 0-1 with every gap's sign flipped (same blur map, mirrored depth)
  test    every test scene x a in GRID (calibrated request = a 0, fixed, per-cell oracle); a = 0 saves activations
  step 3  test scenes 0-5 x {w_d (probe weight, unit, scaled to |u|), random (same norm)} x a in STEP3
          (u itself is covered by the test grid)
  step 4  every test scene x {depth-map: a_k = g(gap_k); proxy: a_p = g(d_hat_p - d_f) per token, d_hat read from the
          unedited activation in the same forward pass; shuffled: proxy cell-mean strengths permuted across the
          free cells by a fixed per-scene derangement}
Edits are applied in the free cells only (area-fraction mask); focus and anchors are never edited.
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
import metrics34 as Q  # noqa: E402
import patches as PT  # noqa: E402
import scenes36 as SC  # noqa: E402

OUT = os.environ["R36_OUT"]
R25 = os.environ["R25_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
GRID = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
STEP3 = [-2.0, -1.0, 1.0, 2.0]
LAMBDAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
PRESERVE_MAX, CON_MARGIN = 0.5, 0.05
FREE = PT.FREE
KEEP = [0, 4, 8]                                      # never edited: anchors and focus
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
S = dict(edit=None, capture=False, feat=None)


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # round 25's fp32 PISA baseline
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def strength_map(x, E):
    """per-token strength (1,1,H,W) for edit spec E, given the unedited activation x (1,C,H,W)."""
    if E["kind"] == "map":
        return E["A"]
    dh = (torch.einsum("chw,c->hw", x[0], E["w"]) + E["b"]).cpu().numpy()          # probe read-out per token
    a = np.interp(dh - E["d_f"], E["knots_x"], E["knots_y"])                       # g(d_hat - d_f)
    A = torch.from_numpy(a.astype(np.float32)).to(x.device)[None, None]
    if E["kind"] == "proxy":
        return A * E["free"]
    cm = E["cm"]                                                                    # shuffled: cell means permuted
    mean = {k: float((A * cm[k]).sum() / cm[k].sum()) for k in FREE}
    S["shuffle_means"] = mean
    return sum(mean[E["perm"][k]] * cm[k] for k in FREE)


def hook(module, args, out):
    h = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, tuple) else out.sample)
    x = h.float()
    if S["capture"]:
        S["feat"] = x[0].half().cpu().numpy()
    E = S["edit"]
    if E is not None:
        A = strength_map(x, E)
        S["A_mean"] = {k: float((A * E["cm"][k]).sum() / E["cm"][k].sum()) for k in FREE}
        h2 = (x + A * E["u"].view(1, -1, 1, 1)).to(h.dtype)
    else:
        return out
    if isinstance(out, torch.Tensor):
        return h2
    if isinstance(out, tuple):
        return (h2,) + tuple(out[1:])
    out.sample = h2
    return out


def ridge_fit(X, y, groups):
    """standardised ridge, lambda by leave-one-group-out CV; returns raw-space (w, b), lambda, cv table."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    cv = {}
    for lam in LAMBDAS:
        err = []
        for gname in sorted(set(groups)):
            tr, te = groups != gname, groups == gname
            Zt, yt = Z[tr], y[tr]
            ym = yt.mean()
            K = Zt @ Zt.T
            alpha = np.linalg.solve(K + lam * np.eye(len(K)), yt - ym)
            err.append(((Z[te] @ (Zt.T @ alpha) + ym - y[te]) ** 2).mean())
        cv[lam] = float(np.mean(err))
    lam = min(cv, key=cv.get)
    ym = y.mean()
    alpha = np.linalg.solve(Z @ Z.T + lam * np.eye(len(Z)), y - ym)
    wz = Z.T @ alpha
    w = wz / sd
    return w, float(ym - mu @ w), lam, cv


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    os.makedirs(f"{OUT}/feats", exist_ok=True)
    unet = pipeline.unet
    unet.up_blocks[0].register_forward_hook(hook)
    V = np.load(f"{HERE}/results/directions32.npz")
    unorm = json.load(open(f"{HERE}/results/directions32.json"))["up0_size"]["raw_norm"]
    u_size = torch.from_numpy(V["up0_size"] * unorm).float().to(dev)
    u_rand = torch.from_numpy(V["up0_random"] * unorm).float().to(dev)
    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])
    t0, n_gen, rows = time.time(), [0], []
    H = W = D.S // 16

    def run(x_img, disp, coc, tag, edit=None, capture=False):
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        x_in = torch.from_numpy(x_img.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        S["edit"], S["capture"], S["feat"], S["A_mean"], S["shuffle_means"] = edit, capture, None, None, None
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x_in, disp_coc=dc, timeit=True)
        S["edit"], S["capture"] = None, False
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        n_gen[0] += 1
        return a.astype(np.float32)

    # ---- G1/G2
    gate = {}
    img25, cid25, _ = Q.scene()
    r0, _ = curve.inverse(11.625)
    g25 = {k: ("coc", r0 / (D.NOM * 10), PT.SIGN[k] * 11.625 / (D.NOM * D.K)) for k in FREE}
    y25 = run(img25, *PT.maps(cid25, g25), "g1_r25_t11.6")
    ref = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t11.6.png")).astype(np.float32)
    gate["G1_repro_r25"] = float(np.abs(y25 - ref).max())
    print("G1", gate, flush=True)

    # ---- per-scene context
    ctx = {}

    def prep(sc):
        if sc["name"] in ctx:
            return ctx[sc["name"]]
        im, cid = sc["img"].astype(np.float32), sc["cid"]
        st = m18.stack(im, "disk")
        ints = {k: D.interior(cid, k) for k in FREE + KEEP}
        keep = ints[0] | ints[4] | ints[8]
        rs = {k: SC.rstar(g) for k, g in sc["gaps"].items()}
        tg = {r: Q.target(sc["img"], r) for r in set(rs.values())}
        coc = {k: curve.inverse(rs[k])[0] / (D.NOM * 10) for k in FREE}
        cm = {k: F.adaptive_avg_pool2d(torch.from_numpy((cid == k).astype(np.float32))[None, None].to(dev), (H, W))
              for k in FREE}
        tok = {k: (F.adaptive_avg_pool2d(torch.from_numpy(ints[k].astype(np.float32))[None, None], (H, W))[0, 0]
                   .numpy() > 0.999) for k in FREE}
        c = dict(im=im, ints=ints, keep=keep, B={k: M.stack_at(st, ints[k]) for k in FREE}, rs=rs, tg=tg,
                 maps=SC.maps(sc, coc), coc=coc, cm=cm, free=sum(cm.values()), tok=tok, y0=None)
        ctx[sc["name"]] = c
        return c

    def measure(sc, y, kind, **info):
        c = prep(sc)
        y0 = c["y0"] if c["y0"] is not None else y
        cells = {}
        for k in FREE:
            m, r = c["ints"][k], c["rs"][k]
            cells[k] = dict(err=Q.off_mae(y, c["tg"][r], m), r_hat=float(M.fit_fast(y[m], c["B"][k])["v"]),
                            contrast=Q.contrast(y, c["tg"][r], m), r_star=r, gap=sc["gaps"][k], d=sc["disp"][k],
                            src=sc["where"][k].rsplit("_", 1)[0], patch=sc["where"][k], A=(S["A_mean"] or {}).get(k, 0.0))
        row = dict(scene=sc["name"], split=sc["split"], d_f=sc["d_f"], kind=kind, cells=cells,
                   preserve=Q.raw_mae(y, y0, c["keep"]),
                   focus_add=Q.raw_mae(y, c["im"], c["ints"][4]) - Q.raw_mae(y0, c["im"], c["ints"][4]), **info)
        if S["shuffle_means"]:
            row["shuffle_means"] = S["shuffle_means"]
        rows.append(row)
        return row

    def grid_edit(sc, a, u):
        c = prep(sc)
        return dict(kind="map", A=a * c["free"], u=u, cm=c["cm"])

    def scene_run(sc, tag, edit=None, capture=False, maps=None):
        c = prep(sc)
        return run(sc["img"], *(maps or c["maps"]), tag, edit=edit, capture=capture)

    dev_sc, test_sc = SC.plan("dev"), SC.plan("test")
    if SMOKE:
        dev_sc, test_sc = dev_sc[:2], test_sc[:1]
    grid = [-2.0, 0.0, 2.0] if SMOKE else GRID

    def grid_pass(scenes):
        for sc in scenes:
            c = prep(sc)
            c["y0"] = scene_run(sc, f"{sc['name']}_a0", capture=True)
            np.save(f"{OUT}/feats/{sc['name']}.npy", S["feat"])
            measure(sc, c["y0"], "grid", a=0.0, dir="size")
            for a in grid:
                if a != 0:
                    measure(sc, scene_run(sc, f"{sc['name']}_size_a{a:+g}", edit=grid_edit(sc, a, u_size)),
                            "grid", a=a, dir="size")
            if sc["name"].endswith("00"):
                gate[f"G2_repeat_{sc['split']}"] = float(np.abs(scene_run(sc, f"{sc['name']}_rep") - c["y0"]).max())
            print(f"{sc['name']} grid done ({n_gen[0]} gens, {(time.time() - t0) / 60:.1f} min) "
                  + " ".join(f"{k}:{d['r_star']:.1f}->{d['r_hat']:.1f}/{d['err']:.2f}" for k, d in rows[-1 - (len(grid) - 1)]["cells"].items()),
                  flush=True)

    grid_pass(dev_sc)
    ok = gate["G1_repro_r25"] == 0 and gate.get("G2_repeat_dev", 0) == 0
    print("gate", gate, "ok", ok, flush=True)
    if not ok:
        json.dump(dict(gate=gate, gate_ok=False, rows=rows), open(f"{OUT}/conds.json", "w"), indent=1)
        print("CHECKS FAILED", flush=True)
        os._exit(0)

    # ---- fit (development only), then freeze
    def tokens(scenes):
        X, lab = [], []
        for sc in scenes:
            f = np.load(f"{OUT}/feats/{sc['name']}.npy").astype(np.float32)        # C x H x W
            for k in FREE:
                t = prep(sc)["tok"][k]
                X.append(f[:, t].T)
                lab += [(sc["disp"][k], abs(sc["gaps"][k]), sc["gaps"][k], sc["where"][k].rsplit("_", 1)[0], k,
                         sc["name"], sc["d_f"])] * int(t.sum())
        return np.concatenate(X), lab

    Xd, ld = tokens(dev_sc)
    grp = np.array([l[3] for l in ld])
    probes = {}
    for j, nm in enumerate(["disp", "abs_gap", "gap"]):
        w, b, lam, cv = ridge_fit(Xd, np.array([l[j] for l in ld], float), grp)
        probes[nm] = dict(w=w, b=b, lam=lam, cv=cv)
        print(f"probe {nm}: lambda {lam:g} cv mse {cv[lam]:.4f} (label var {np.var([l[j] for l in ld]):.4f})", flush=True)
    wd = probes["disp"]["w"]
    u_wd = torch.from_numpy(wd / np.linalg.norm(wd) * unorm).float().to(dev)
    w_t = torch.from_numpy(wd).float().to(dev)

    best = {}
    dev_rows = [r for r in rows if r["split"] == "dev" and r["kind"] == "grid"]

    def pick(rs_, k):
        c = [(r["a"], r["cells"][k], r["preserve"]) for r in rs_]
        c0 = next(d for a, d, _ in c if a == 0)
        ok_ = [(d["err"], a) for a, d, p in c if abs(d["r_hat"] - d["r_star"]) <= m18.TOL(d["r_star"])
               and p <= PRESERVE_MAX and d["contrast"] >= c0["contrast"] - CON_MARGIN]
        return min(ok_)[1] if ok_ else min((abs(d["r_hat"] - d["r_star"]), a) for a, d, _ in c)[1]

    for sc in dev_sc:
        rs_ = [r for r in dev_rows if r["scene"] == sc["name"]]
        for k in FREE:
            best.setdefault(sc["gaps"][k], []).append(pick(rs_, k))
    knots_x = sorted(best)
    knots_y = [float(np.median(best[g])) for g in knots_x]

    def ratio(r, r0_, k):
        d, d0 = r["cells"][k], r0_["cells"][k]
        v = lambda dd, p: abs(dd["r_hat"] - dd["r_star"]) <= m18.TOL(dd["r_star"]) and p <= PRESERVE_MAX
        if not (v(d, r["preserve"]) and d["contrast"] >= d0["contrast"] - CON_MARGIN):
            return float("inf")
        return d["err"] / d0["err"] if v(d0, 0.0) else 0.0

    fixed_score = {}
    for a in grid:
        rr = []
        for sc in dev_sc:
            r0_ = next(r for r in dev_rows if r["scene"] == sc["name"] and r["a"] == 0)
            ra = next(r for r in dev_rows if r["scene"] == sc["name"] and r["a"] == a)
            rr += [ratio(ra, r0_, k) for k in FREE]
        fixed_score[a] = float(np.median(rr))
    a_fix = min(fixed_score, key=lambda a: (fixed_score[a], abs(a)))
    rule = dict(knots_x=knots_x, knots_y=knots_y, best_dev=best and {str(g): v for g, v in best.items()},
                a_fixed=a_fix, fixed_dev_median=fixed_score)
    np.savez(f"{OUT}/probe36.npz", **{f"{n}_w": p["w"] for n, p in probes.items()},
             **{f"{n}_b": np.array(p["b"]) for n, p in probes.items()})
    json.dump(dict(rule=rule, probes={n: dict(b=p["b"], lam=p["lam"], cv={str(k): v for k, v in p["cv"].items()})
                                      for n, p in probes.items()}),
              open(f"{OUT}/frozen36.json", "w"), indent=1)
    print("FROZEN rule", rule, flush=True)

    # ---- D0: disparity-only sensitivity
    for sc in test_sc[:2]:
        c = prep(sc)
        flip = dict(sc, gaps={k: -g for k, g in sc["gaps"].items()})
        dmap, _ = SC.maps(flip, c["coc"])
        c["y0"] = None
        y0 = scene_run(sc, f"{sc['name']}_a0_d0ref", capture=True)
        f0 = S["feat"].astype(np.float32)
        yf = scene_run(sc, f"{sc['name']}_dispflip", capture=True, maps=(dmap, c["maps"][1]))
        ff = S["feat"].astype(np.float32)
        d0 = {}
        for k in FREE:
            m, t = c["ints"][k], c["tok"][k]
            d0[k] = dict(pix=Q.raw_mae(yf, y0, m), r_hat0=float(M.fit_fast(y0[m], c["B"][k])["v"]),
                         r_hat_flip=float(M.fit_fast(yf[m], c["B"][k])["v"]),
                         feat_rel=float(np.linalg.norm(ff[:, t] - f0[:, t]) / np.linalg.norm(f0[:, t])),
                         probe_shift=float(((ff[:, t] - f0[:, t]).T @ wd).mean()), gap=sc["gaps"][k])
        gate[f"D0_{sc['name']}"] = d0
        print("D0", sc["name"], {k: (round(v["pix"], 2), round(v["feat_rel"], 3), round(v["probe_shift"], 3)) for k, v in d0.items()},
              flush=True)
        c["y0"] = None

    # ---- test grid (calibrated request, fixed, oracle) + activations
    grid_pass(test_sc)
    Xt, lt = tokens(test_sc)
    np.savez(f"{OUT}/probe36_test.npz", **{f"pred_{n}": Xt @ p["w"] + p["b"] for n, p in probes.items()},
             disp=np.array([l[0] for l in lt]), abs_gap=np.array([l[1] for l in lt]), gap=np.array([l[2] for l in lt]),
             src=np.array([l[3] for l in lt]), cell=np.array([l[4] for l in lt]), scene=np.array([l[5] for l in lt]),
             d_f=np.array([l[6] for l in lt]),
             dev_pred_disp=Xd @ probes["disp"]["w"] + probes["disp"]["b"], dev_disp=np.array([l[0] for l in ld]))

    # ---- step 3: causal edit test (w_d and random; u = size direction is in the test grid)
    for sc in test_sc[:1 if SMOKE else 6]:
        for nm, u in (("wd", u_wd), ("random", u_rand)):
            for a in ([-2.0, 2.0] if SMOKE else STEP3):
                measure(sc, scene_run(sc, f"{sc['name']}_{nm}_a{a:+g}", edit=grid_edit(sc, a, u)), "step3", a=a, dir=nm)
        print(f"{sc['name']} step3 done ({n_gen[0]} gens)", flush=True)

    # ---- step 4: adaptive strengths (rule frozen above)
    for i, sc in enumerate(test_sc):
        c = prep(sc)
        rng = np.random.default_rng(1000 + i)
        while True:
            p = rng.permutation(FREE)
            if all(p[j] != FREE[j] for j in range(6)):
                break
        perm = {FREE[j]: int(p[j]) for j in range(6)}
        Ad = sum(float(np.interp(g, knots_x, knots_y)) * c["cm"][k] for k, g in sc["gaps"].items())
        base = dict(u=u_size, cm=c["cm"], free=c["free"], w=w_t, b=probes["disp"]["b"], d_f=sc["d_f"],
                    knots_x=knots_x, knots_y=knots_y, perm=perm)
        for nm, E in (("depth", dict(base, kind="map", A=Ad)), ("proxy", dict(base, kind="proxy")),
                      ("shuffled", dict(base, kind="shuffle"))):
            measure(sc, scene_run(sc, f"{sc['name']}_{nm}", edit=E), "method", method=nm, perm=perm)
        print(f"{sc['name']} methods done ({n_gen[0]} gens) "
              + " | ".join(f"{r['method']} " + " ".join(f"{d['A']:+.1f}" for d in r["cells"].values()) for r in rows[-3:]),
              flush=True)

    meta = dict(gate=gate, gate_ok=True, rows=rows, grid=grid, step3=STEP3, smoke=SMOKE, unorm=unorm,
                n_generations=n_gen[0], minutes=(time.time() - t0) / 60)
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1, default=float)
    print("done:", n_gen[0], "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
