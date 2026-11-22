"""Round 39 step 1 driver (GPU): does BokehDiff make a repeatable boundary error that request adjustment cannot fix?

Plain BokehDiff (no steering). Scenes, settings and levels from scenes39; requests from the frozen linear-light
calibration curve (results/calib_linear.json, curve_lin); measurement and validity from measure39.

Stage V (checks; the run stops unless all pass):
  V1  GPU ray tracer == numpy ray tracer (n_side 8, one scene, setting C): max |diff| <= 1e-3 (0-255)
  V2  repeatability: the same generation twice: max |diff| == 0
  V3  convergence: 1024 vs 4096 aperture samples, 6 scenes (each shape x bg kind) x A/B/C at L1:
      mean |diff| in every edge band <= 0.1
  V4  available information: reference with the background reconstructed from visible pixels vs the true one,
      same 6 x 3: mean |diff| in every edge band <= 0.1
Stage G (per scene: one zero-request generation k0, the focus reference; then per setting x level, 24 x 3 x 2):
  reference  ray tracer (1024 samples) at the target radii; saved to ref/ as float32 .npy
  base       calibrated request (global multiplier 1.0)
  global     multiplier on every blurred pixel's request: 0.6 .. 1.4 step 0.1
  edge       at the best valid global multiplier (lowest E), the request within 16 px of the edge (|dist| <= 16)
             multiplied by 0, 0.5, 0.75, 1.25, 1.5, 2.0
Each generation is measured inline; rows go to conds.json. Images in img/.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import lens39 as LN  # noqa: E402
import measure19 as M  # noqa: E402
import measure39 as Q  # noqa: E402
import scenes39 as S39  # noqa: E402

OUT = os.environ["R39_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
MG = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
ME = [0.0, 0.5, 0.75, 1.25, 1.5, 2.0]
EDGE_W = 16
NS, NS_HI = 32, 64
VAL_SCENES = [0, 1, 8, 9, 16, 17]
_orig_val = I.log_validation


def targets(setting, r):
    return {"A": dict(bg=r), "B": dict(fg=r), "C": dict(fg=r / 2, bg=r / 2)}[setting]


def reference(b, setting, tg, n_side, B=None):
    fg = dict(F=b["F"], alpha=b["alpha"], d=1.0)
    bg = dict(F=b["B"] if B is None else B, alpha=np.ones_like(b["alpha"]), d=0.0)
    rad = {}
    if "fg" in tg:
        rad[0] = tg["fg"]
    if "bg" in tg:
        rad[1] = tg["bg"]
    return LN.raytrace_torch([fg, bg], S39.SETTINGS[setting], n_side, "cuda", radius=rad)


def edge_means(a, c, bands):
    d = np.abs(a - c).mean(2)
    return {n: float(d[bands[n]].mean()) for n in S39.EDGE_BANDS}


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    for d in ("img", "ref", "input"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    cl = json.load(open(f"{HERE}/results/calib_linear.json"))["curve_lin"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(cl["x"]), np.array(cl["y"])
    coc_of = lambda r: curve.inverse(r)[0] / (D.NOM * 10)
    t0, n_gen = time.time(), [0]

    def run(img, disp, coc, tag):
        dc = torch.from_numpy(np.stack([disp, coc]).astype(np.float32)[None]).to(dev, dt)
        x_in = torch.from_numpy(img.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.autocast("cuda", dtype=torch.float16):
            image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                 aif_image=x_in, disp_coc=dc, timeit=True)
        a = np.asarray(image)
        Image.fromarray(a).save(f"{OUT}/img/{tag}.png")
        n_gen[0] += 1
        return a.astype(np.float64)

    def coc_map(b, tg, mg=1.0, me=1.0):
        c = np.where(b["fgmask"], coc_of(tg["fg"]) if "fg" in tg else 0.0, coc_of(tg["bg"]) if "bg" in tg else 0.0)
        c = c * mg
        c[np.abs(b["dist"]) <= EDGE_W] *= me
        return c.astype(np.float32)

    plan = S39.plan()
    scenes = plan[:1] if SMOKE else plan
    settings = ["A", "B"] if SMOKE else list(S39.SETTINGS)
    levels = ["L1"] if SMOKE else list(S39.LEVELS)
    mg_grid, me_grid = ([0.8, 1.0], [0.5]) if SMOKE else (MG, ME)
    val_ids = [0] if SMOKE else VAL_SCENES

    # ---- stage V
    checks = {}
    b0 = S39.build(plan[0])
    b0["bands"] = S39.band_masks(b0["dist"])
    fg0, bg0 = dict(F=b0["F"], alpha=b0["alpha"], d=1.0), dict(F=b0["B"], alpha=np.ones((S39.S, S39.S)), d=0.0)
    rn = LN.raytrace([fg0, bg0], 0.5, 8)
    rt = LN.raytrace_torch([fg0, bg0], 0.5, 8, "cuda")
    checks["V1_torch_vs_numpy"] = float(np.abs(rn - rt).max())
    tg0 = targets("A", 15.5)
    ya = run(b0["img"], b0["disp"], coc_map(b0, tg0), "v2_a")
    yb = run(b0["img"], b0["disp"], coc_map(b0, tg0), "v2_b")
    checks["V2_repeat"] = float(np.abs(ya - yb).max())
    conv, avail = {}, {}
    for i in val_ids:
        b = S39.build(plan[i])
        b["bands"] = S39.band_masks(b["dist"])
        Brec = S39.reconstruct_bg(b["img"], b["alpha"], b["bg_kind"])
        for st in S39.SETTINGS:
            tg = targets(st, S39.LEVELS["L1"])
            r32 = reference(b, st, tg, NS)
            conv[f"{b['name']}/{st}"] = edge_means(r32, reference(b, st, tg, NS_HI), b["bands"])
            avail[f"{b['name']}/{st}"] = edge_means(r32, reference(b, st, tg, NS, B=Brec), b["bands"])
        print("validation", b["name"], flush=True)
    checks["V3_convergence_max"] = max(max(v.values()) for v in conv.values())
    checks["V4_available_max"] = max(max(v.values()) for v in avail.values())
    ok = (checks["V1_torch_vs_numpy"] <= 1e-3 and checks["V2_repeat"] == 0 and checks["V3_convergence_max"] <= 0.1
          and checks["V4_available_max"] <= 0.1)
    print("checks", checks, "OK" if ok else "FAILED", flush=True)
    meta = dict(checks=checks, checks_ok=ok, convergence=conv, available=avail, smoke=SMOKE, rows=[],
                mg=mg_grid, me=me_grid, edge_w=EDGE_W, n_side=NS)
    if not ok:
        json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1, default=float)
        sys.stdout.flush()
        os._exit(0)

    # ---- stage G
    rows = meta["rows"]
    for sc in scenes:
        b = S39.build(sc)
        b["bands"] = S39.band_masks(b["dist"])
        Image.fromarray(b["img"]).save(f"{OUT}/input/{sc['name']}.png")
        Bfg = Q.lin_stack_fg(b)
        k0 = run(b["img"], b["disp"], np.zeros((S39.S, S39.S), np.float32), f"{sc['name']}_k0")
        for st in settings:
            for lv in levels:
                tg = targets(st, S39.LEVELS[lv])
                ref = reference(b, st, tg, NS)
                np.save(f"{OUT}/ref/{sc['name']}_{st}_{lv}.npy", ref.astype(np.float32))
                key = f"{sc['name']}_{st}_{lv}"

                def gen(kind, mg, me):
                    y = run(b["img"], b["disp"], coc_map(b, tg, mg, me), f"{key}_{kind}_g{mg:.2f}_e{me:.2f}")
                    m = Q.measure(y, ref, b, st, tg, Bfg, k0)
                    rows.append(dict(scene=sc["name"], shape=sc["shape"], tex=sc["tex"], bg=sc["bg_kind"], setting=st,
                                     level=lv, targets=tg, kind=kind, mg=mg, me=me, **m))
                    return m

                gl = {mg: gen("global", mg, 1.0) for mg in mg_grid}
                valid = [mg for mg in mg_grid if gl[mg]["valid"]]
                best = min(valid or mg_grid, key=lambda mg: gl[mg]["E"])
                for me in me_grid:
                    gen("edge", best, me)
                print(f"{key}: base E {gl[1.0]['E']:.2f} ({gl[1.0]['E_band']}, valid {gl[1.0]['valid']}) "
                      f"best global x{best} E {gl[best]['E']:.2f} | edge "
                      + " ".join(f"{r['me']}:{r['E']:.2f}{'' if r['valid'] else '*'}" for r in rows[-len(me_grid):])
                      + f" | {n_gen[0]} gens {(time.time() - t0) / 60:.0f} min", flush=True)
        json.dump(meta, open(f"{OUT}/conds_partial.json", "w"), default=float)
    meta.update(n_generations=n_gen[0], minutes=(time.time() - t0) / 60)
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1, default=float)
    print("done:", n_gen[0], "generations", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
