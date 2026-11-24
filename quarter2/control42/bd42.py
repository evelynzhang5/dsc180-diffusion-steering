"""Round 42 (GPU): depth-aware control of BokehDiff's blur. Same driver as round 40 bd40.py (processing size,
preprocessing, training depth convention, prompt, seed 0, fp16); only the K handed to it changes.

Per held-out test scene x target (4, 8, 12 px), arms (plan and rule frozen in results/plan42.json, rule42.json):
  fixed    round 40's one K per target. Outputs REUSED from round 40 (s3/bd_test); determinism check: the first 3 test
           scenes are re-rendered at the fixed K and compared pixel-wise with round 40.
  depth    K from the frozen depth rule (inputs only).
  correct  render at the depth K, measure the delivered blur on the FEEDBACK tiles (even-indexed behind tiles; all
           tiles if fewer than 2 even ones), K2 = K1 + (t - r1) / (slope * gap), clipped to [4, 64], render again.
           If r1 is unmeasurable, no correction is made (K2 = K1, flagged).
  oracle   reference-assisted search (offline upper bound, not a usable controller): render the round 40 K grid, measure
           delivered blur on ALL tiles, take the first crossing of the target (K linear), render there. If the grid never
           crosses, the grid K closest to the target is kept.
Timing: render seconds per call (synchronised, model loaded); measurement seconds for correct/oracle.
usage (BokehDiff repo as cwd): python bd42.py --test_data_dir <one input png> --output_dir <unused> ...
env: R40_DATA, R40_S3 (round 40 stage 3 out), R42_OUT, SMOKE
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compare40"))
import blur40 as BL  # noqa: E402
import common40 as C  # noqa: E402

DATA, S3, OUT = os.environ["R40_DATA"], os.environ["R40_S3"], os.environ["R42_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
RULE = json.load(open(os.path.join(HERE, "results", "rule42.json")))
PLAN = json.load(open(os.path.join(HERE, "results", "plan42.json")))
_orig_val = I.log_validation


def proc_size(w, h):                                      # = bd40.proc_size
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def prepared_disp(i, W2, H2):                             # = bd40.prepared_disp, training convention
    d = np.load(f"{DATA}/depth/{i}_pred.npy").astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min())
    dm = cv2.dilate(cv2.erode(d, np.ones((3, 3))), np.ones((3, 3)))
    d = np.clip((d - dm.min()) / (dm.max() - dm.min()), 0, 1)
    m = np.array(Image.open(f"{DATA}/mask/{i}.png").resize((d.shape[1], d.shape[0]), resample=Image.LANCZOS)
                 .convert("L")) / 255.0
    f = float(np.median(d[m > 0.5]))
    d, f = 1 - d, 1 - f
    d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR))
    return d.astype(np.float32), f


def crossing(Ks, rs, t):
    for k in range(len(Ks) - 1):
        lo, hi = rs[k], rs[k + 1]
        if np.isfinite(lo) and np.isfinite(hi) and (lo - t) * (hi - t) <= 0 and lo != hi:
            return Ks[k] + (t - lo) / (hi - lo) * (Ks[k + 1] - Ks[k]), True
    fin = [(abs(r - t), K) for K, r in zip(Ks, rs) if np.isfinite(r)]
    return (min(fin)[1] if fin else Ks[len(Ks) // 2]), False


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    scenes, regions, review = C.load(DATA)
    form, prm = RULE["form"], RULE["params"]
    slope = prm["a"] if form == "P" else prm["b"]
    test = PLAN["test_scenes"]
    targets = [f"{t:g}" for t in C.TARGETS]
    grid = C.K_GRID
    if SMOKE:
        test, targets, grid = test[:1], ["8"], [8, 20, 48]
    for sub in ["fixed_check", "depth", "correct", "oracle_grid", "oracle"]:
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    rows = []

    def render(i, K, tag):
        x = Image.open(f"{DATA}/input/{i}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        d, f = prepared_disp(i, W2, H2)
        dc = torch.from_numpy(np.stack([d, np.abs(d - f) * K / 10])[None]).to(dev, dt)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                   aif_image=x_in, disp_coc=dc, timeit=True)
        torch.cuda.synchronize()
        image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
        print(tag, f"K {K:.2f}", f"{dur:.2f}s", flush=True)
        return f"{tag}.png", float(dur)

    for n, i in enumerate(test):
        x = C.img(f"{DATA}/input/{i}.png")
        tiles = regions[i]["tiles"]
        fb_tiles = tiles[0::2] if len(tiles[0::2]) >= C.MIN_TILES else tiles
        t0 = time.time()
        st_fb, st_all = BL.TileStack(x, fb_tiles), BL.TileStack(x, tiles)
        stack_s = time.time() - t0
        g = PLAN["plan"][i]["gap"]
        if n < 3:                                                    # determinism check vs round 40
            for t in targets:
                K = PLAN["plan"][i]["targets"][t]["fixed_K"]
                f, _ = render(i, K, f"fixed_check/{i}_t{t}")
                a, b = C.img(f"{OUT}/{f}"), C.img(f"{S3}/bd_test/{i}_t{t}.png")
                rows.append(dict(id=i, target=float(t), arm="fixed_check", K=K, file=f,
                                 max_abs_diff_vs_r40=float(np.abs(a - b).max()),
                                 mean_abs_diff_vs_r40=float(np.abs(a - b).mean())))
        grid_r = []                                                  # oracle grid (shared by the 3 targets)
        grid_s = 0.0
        for K in grid:
            f, dur = render(i, K, f"oracle_grid/{i}_K{K:g}")
            t1 = time.time()
            grid_r.append(st_all.measure(C.img(f"{OUT}/{f}"))["r"])
            grid_s += dur + time.time() - t1
        for t in targets:
            p = PLAN["plan"][i]["targets"][t]
            K1 = p["depth_K"]
            f1, d1 = render(i, K1, f"depth/{i}_t{t}")
            rows.append(dict(id=i, target=float(t), arm="depth", K=K1, file=f1, render_s=d1, total_s=d1, n_renders=1))
            t1 = time.time()
            r1 = st_fb.measure(C.img(f"{OUT}/{f1}"))["r"]
            meas_s = time.time() - t1 + stack_s
            if np.isfinite(r1):
                K2 = float(np.clip(K1 + (float(t) - r1) / (slope * g), RULE["k_min"], RULE["k_max"]))
                f2, d2 = render(i, K2, f"correct/{i}_t{t}")
                rows.append(dict(id=i, target=float(t), arm="correct", K=K2, K1=K1, r1_feedback=r1, file=f2,
                                 render_s=d1 + d2, total_s=d1 + d2 + meas_s, n_renders=2, corrected=True,
                                 feedback_tiles=len(fb_tiles)))
            else:
                rows.append(dict(id=i, target=float(t), arm="correct", K=K1, K1=K1, r1_feedback=r1, file=f1,
                                 render_s=d1, total_s=d1 + meas_s, n_renders=1, corrected=False,
                                 feedback_tiles=len(fb_tiles)))
            Ko, crossed = crossing(grid, grid_r, float(t))
            if crossed:
                fo, do = render(i, Ko, f"oracle/{i}_t{t}")
            else:
                fo, do = f"oracle_grid/{i}_K{Ko:g}.png", 0.0
            rows.append(dict(id=i, target=float(t), arm="oracle", K=float(Ko), crossed=crossed, file=fo,
                             grid_r=grid_r, render_s=do, total_s=grid_s + do, n_renders=len(grid) + int(crossed)))
        json.dump(dict(smoke=SMOKE, grid=grid, rows=rows), open(f"{OUT}/bd42.json", "w"), indent=1)
    json.dump(dict(smoke=SMOKE, grid=grid, rows=rows), open(f"{OUT}/bd42.json", "w"), indent=1)
    print("bd42 done", len(rows), "rows", flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
