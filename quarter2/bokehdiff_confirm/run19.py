"""Round 19 inference driver. Run from the BokehDiff repo directory (pinned commit), inside the round 17 venv:
    python run19.py <inference_hf.py arguments; --test_data_dir points at round 18's scene A (only used to start main)>

As in round 18, BokehDiff's main() builds the pipeline (released LoRA + VAE, PISA attention) and its first
log_validation call is intercepted. Then:
  stage 1  uncorrected runs, all 6 scenes: K=0 reference; R0-R6 in both orientations (P1, P2); R3 P1 with seed 1;
           one in-process repeat (S1 R3 P1)
  stage 2  radius of every non-focus cell in the 2 development scenes (S1, S2) -> calibrations C0/C1/C2, frozen to
           calib.json before any corrected run
  stage 3  test scenes S3-S6, R1-R6 x P1/P2, rerun with corrected blur requests for C0, C1, C2
The disparity channel always keeps the true geometry; only the blur map is changed by a correction.
fp16 handling as in rounds 17-18; global RNG reseeded before every call (PISA dropout and occlusion sampling).
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import measure19 as M  # noqa: E402

OUT = os.environ["R19_OUT"]
_orig = I.log_validation


def cell_table(run, perm):
    d_f, cl = D.cells(run, perm)
    for c in cl:
        c["r_nom"] = D.NOM * D.K * abs(c["gap"])
    return d_f, cl


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    os.makedirs(f"{OUT}/maps", exist_ok=True)
    conds = []

    def run(scene, tag, disp, coc, sd=0, **info):
        s = D.make(scene)
        img = torch.from_numpy(s["image"].astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig(pipeline, sd, train_T, accelerator, validation_prompt=validation_prompt,
                               aif_image=img, disp_coc=dc, timeit=True)
        f = f"img/{scene}_{tag}.png"
        image.save(f"{OUT}/{f}")
        np.savez_compressed(f"{OUT}/maps/{scene}_{tag}.npz", disp=disp.astype(np.float16), coc=coc.astype(np.float16))
        a = np.asarray(image).astype(np.float32)
        conds.append(dict(scene=scene, tag=tag, file=f, seed=sd, seconds=dur, finite=bool(np.isfinite(a).all()),
                          black=bool(a.max() < 5), **info))
        return a

    # ---- stage 1
    for n in D.SCENES:
        cid = D.make(n)["cid"]
        disp, _ = D.maps(cid, "R0", "P1")
        run(n, "K0", disp, np.zeros_like(disp), kind="K0", run="K0", perm="P1", method="none")
        for r in D.RUNS:
            for p in ("P1", "P2"):
                disp, coc = D.maps(cid, r, p)
                d_f, cl = cell_table(r, p)
                run(n, f"{r}_{p}", disp, coc, kind="grid", run=r, perm=p, method="none", d_f=d_f, cells=cl)
        disp, coc = D.maps(cid, "R3", "P1")
        d_f, cl = cell_table("R3", "P1")
        run(n, "R3_P1_seed1", disp, coc, sd=1, kind="seed1", run="R3", perm="P1", method="none", d_f=d_f, cells=cl)
        print("stage 1", n, flush=True)
    cid = D.make("S1")["cid"]
    disp, coc = D.maps(cid, "R3", "P1")
    a = run("S1", "R3_P1_repeat", disp, coc, kind="repeat", run="R3", perm="P1", method="none")
    b = np.asarray(Image.open(f"{OUT}/img/S1_R3_P1.png")).astype(np.float32)
    repeat_diff = float(np.abs(a - b).max())
    print("repeat max |diff|", repeat_diff, flush=True)

    # ---- stage 2: dev measurements -> frozen calibrations
    dev_rows, floor_vals = [], []
    for n in D.DEV:
        s = D.make(n)
        st = m18.stack(s["image"].astype(np.float32), "disk")
        Bs = {k: M.stack_at(st, D.interior(s["cid"], k)) for k in range(9)}
        ints = {k: D.interior(s["cid"], k) for k in range(9)}
        y0 = np.asarray(Image.open(f"{OUT}/img/{n}_K0.png")).astype(np.float32)
        floor_vals += [M.fit_fast(y0[ints[k]], Bs[k])["v"] for k in range(9)]
        for c in [c for c in conds if c["scene"] == n and c["kind"] == "grid" and c["run"] in D.GRID_RUNS]:
            y = np.asarray(Image.open(f"{OUT}/{c['file']}")).astype(np.float32)
            for cl in c["cells"]:
                if cl["side"] == "focus":
                    continue
                f = M.fit_fast(y[ints[cl["cell"]]], Bs[cl["cell"]])
                dev_rows.append(dict(scene=n, run=c["run"], perm=c["perm"], cell=cl["cell"], side=cl["side"],
                                     r_nom=cl["r_nom"], r_hat=f["v"], at_cap=f["at_cap"]))
    floor = float(np.median(floor_vals))
    c0, c1, c2 = M.fit_calibrations(dev_rows, floor)
    calib = dict(floor=floor, C0=c0, C1=c1.as_dict(), C2={k: v.as_dict() for k, v in c2.items()}, dev_rows=dev_rows)
    json.dump(calib, open(f"{OUT}/calib.json", "w"), indent=1)
    print("calibration frozen: floor %.2f, C0 gain %.3f, C1 points %d" % (floor, c0, len(c1.x)), flush=True)

    # ---- stage 3: corrected reruns on test scenes
    for n in D.TEST:
        cid = D.make(n)["cid"]
        for r in D.GRID_RUNS:
            for p in ("P1", "P2"):
                d_f, cl = cell_table(r, p)
                for meth in ("C0", "C1", "C2"):
                    req = M.corrected_requests(cl, meth, c0, c1, c2)
                    over = {k: v[0] / (D.NOM * 10) for k, v in req.items()}
                    disp, coc = D.maps(cid, r, p, coc_override=over)
                    cl2 = [dict(c, r_req=req[c["cell"]][0] if c["cell"] in req else 0.0,
                                req_status=req[c["cell"]][1] if c["cell"] in req else "focus") for c in cl]
                    run(n, f"{r}_{p}_{meth}", disp, coc, kind="corrected", run=r, perm=p, method=meth, d_f=d_f, cells=cl2)
        print("stage 3", n, flush=True)

    meta = dict(conds=conds, repeat_max_abs_diff=repeat_diff, peak_mem_alloc_gb=torch.cuda.max_memory_allocated() / 1e9,
                mean_seconds=float(np.mean([c["seconds"] for c in conds])), n_outputs=len(conds), K=D.K)
    json.dump(meta, open(f"{OUT}/conds.json", "w"), indent=1)
    print("done:", len(conds), "outputs, peak alloc GB %.2f" % meta["peak_mem_alloc_gb"], flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
