"""Round 47 (GPU): BokehDiff renders for the separation pipeline. FROZEN before training.
Same driver as round 46 (fp32 PISA with the in-call assertion, seed 0, maps in the processing convention).
  SEP     T1 and T2: BokehDiff(B_hat, the background's own map d_B, round 46 focus and K), per variant (full, oracle).
  T2 base T2 composites with the original map (A: pane at the focus) and the see-through map (S: d_B), for the output blend.
  CHECK   BokehDiff(true B) must equal round 46's reference render (pipeline check, max diff <= 1/255).
  T3      rb_48: BokehDiff(B_hat full) with round 43's see-through map, focus and the 12 px K (round 43 bd43.json).
usage (BokehDiff repo as cwd): python bd47.py --test_data_dir <png with depth> --output_dir <unused> ...
env: R46_DATA, R46_RUN, R47_T2, R47_SEP (infer47 out), R43_RUN, R47_OUT, SMOKE
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402
import PISA_attn_processor as P  # noqa: E402

D46, R46, T2, SEP, R43, OUT = (os.environ[k] for k in ["R46_DATA", "R46_RUN", "R47_T2", "R47_SEP", "R43_RUN", "R47_OUT"])
SMOKE = os.environ.get("SMOKE", "0") == "1"
META = json.load(open(f"{D46}/compose46.json"))
_orig_val = I.log_validation
_orig_attn = P.customized_scaled_dot_product_attention
CALLS = Counter()


def _f(t):
    return t.float() if torch.is_tensor(t) else t


def attn_fp32(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):
        out = _orig_attn(_f(query), _f(key), _f(value), _f(weight_matrix), _f(disp_coc), attn_mask=attn_mask,
                         dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard, train=train,
                         occ_map=_f(occ_map))
    assert out.dtype == torch.float32, out.dtype
    CALLS[(str(query.dtype), str(out.dtype))] += 1
    return out.to(value.dtype)


P.customized_scaled_dot_product_attention = attn_fp32


def proc_size(w, h):
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    rows, checks = [], {}

    def render(path, dmap, f, K, tag):
        x = Image.open(path).convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        d = dmap if dmap.shape == (H2, W2) else np.asarray(
            Image.fromarray(dmap.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR))
        dc = torch.from_numpy(np.stack([d, np.abs(d - f) * K / 10])[None].astype(np.float32)).to(dev, dt)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                   aif_image=x_in, disp_coc=dc, timeit=True)
        os.makedirs(os.path.dirname(f"{OUT}/{tag}"), exist_ok=True)
        image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
        print(tag, f"K {K:.2f} {dur:.2f}s", flush=True)
        return f"{tag}.png", float(time.time() - t0)

    f = META["focus"]
    pane = np.load(f"{D46}/maps/pane.npy")
    m47 = json.load(open(f"{T2}/compose47.json"))
    inf = json.load(open(f"{SEP}/infer47.json"))["rows"]
    bgs = META["bgs"][:1] if SMOKE else META["bgs"]
    for bg in bgs:                                                     # pipeline check
        K, dB = META["backgrounds"][bg]["K"], np.load(f"{D46}/maps/{bg}_dB.npy")
        fn, _ = render(f"{D46}/input/{bg}__bgonly.png", dB, f, K, f"check/ref_{bg}")
        a = np.asarray(Image.open(f"{OUT}/{fn}")).astype(int)
        b = np.asarray(Image.open(f"{R46}/ref/{bg}.png")).astype(int)
        checks[f"pipeline_{bg}"] = int(np.abs(a - b).max())
    for r in inf:                                                      # SEP renders (T1, T2)
        if r["set"] not in ("T1", "T2"):
            continue
        bg = r["name"].split("__")[0]
        if bg not in bgs:
            continue
        K, dB = META["backgrounds"][bg]["K"], np.load(f"{D46}/maps/{bg}_dB.npy")
        fn, s = render(f"{SEP}/{r['tag']}_B.png", dB, f, K, f"sep/{r['tag']}")
        rows.append(dict(kind="sep", set=r["set"], name=r["name"], variant=r["variant"], file=fn, seconds=s))
    for c, m in m47["composites"].items():                            # T2 baselines
        if m["bg"] not in bgs:
            continue
        K, dB = META["backgrounds"][m["bg"]]["K"], np.load(f"{D46}/maps/{m['bg']}_dB.npy")
        for slot, mp in [("A", np.where(pane > 0.5, f, dB)), ("S", dB)]:
            fn, s = render(f"{T2}/input/{c}.png", mp, f, K, f"t2base/{slot}/{c}")
            rows.append(dict(kind=slot, set="T2", name=c, file=fn, seconds=s))
    r43 = [r for r in json.load(open(f"{R43}/bd43.json"))["rows"] if r["id"] == "rb_48"
           and r["cond"] == "seethrough" and r["k_tag"] == "12"][0]   # T3
    d43 = np.load(f"{R43}/{r43['depth']}").astype(np.float32)
    fn, s = render(f"{SEP}/T3/full/rb_48_B.png", d43, r43["focus"], r43["K"], "sep/T3/full/rb_48")
    rows.append(dict(kind="sep", set="T3", name="rb_48", variant="full", file=fn, seconds=s, K=r43["K"]))
    json.dump(dict(smoke=SMOKE, rows=rows, checks=checks, calls={"|".join(k): v for k, v in CALLS.items()}),
              open(f"{OUT}/bd47.json", "w"), indent=1)
    print("bd47 done", len(rows), "renders; checks", checks, "; PISA calls", dict(CALLS), flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
