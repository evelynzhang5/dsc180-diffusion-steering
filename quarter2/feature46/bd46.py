"""Round 46 (GPU): can internal feature mixing beat output blending on transparent content? (two-run methods)

Same BokehDiff driver as rounds 40-45 (processing size, prompt, seed 0, fp16 pipeline, round 45's fp32 PISA wrapper
with the in-call float32 assertion). Depth maps are built directly in the processing convention from compose46.py
(no subject mask, no re-normalisation): for background bg, d_B (its own map), pane depth = focus f = 0.95, the same K.
Renders per background: REF = BokehDiff(B alone, d_B) (the controlled compositing reference uses it).
Renders per composite (input I = alpha F + (1 - alpha) B):
  A        original map: d_B with the pane set to f (the pane is solid and in focus);
  S        see-through map: d_B everywhere (transmitted content far; also the "blur everything" negative control);
  M2       surface-aware single map: d_B with alpha >= 0.5 pixels set to f;
  C_pisa   PRIMARY feature blend: output of BokehDiff's only depth-conditioned block (unet.down_blocks[1].attentions[0],
           the PISA transformer) replaced, tile by tile, by alpha * h_A + (1 - alpha) * h_S (alpha area-averaged to the
           block's resolution). Everything computed before this block is depth-free and identical in A and S, and no
           later layer receives depth, so this one tensor is the complete state that differs between the runs; the
           whole continuation (rest of the UNet incl. skips, scheduler step, VAE) is then re-run.
  C_latent exploratory: the latent entering the VAE decoder blended (alpha area-averaged to latent resolution).
  C_dec    exploratory: the VAE decoder's mid-block output blended.
  The pixel output blend O = alpha * A + (1 - alpha) * S (linear light) is computed on CPU.
alpha reaches the tile hook as a third disp_coc channel (PISA reads only channels 0 and 1; checked below).
Controls (first composite, and the flat composite):
  zero intervention  A rendered with a 2-channel disp_coc and no hooks must equal A (<= 1/255);
  endpoints          each C_* with alpha := 1 must equal A and with alpha := 0 must equal S, through the full continuation;
  flat colour        flat grey background with the tinted surface: A, S, C_* (brightness checked on CPU).
usage (BokehDiff repo as cwd): python bd46.py --test_data_dir <one input png> --output_dir <unused> ...
env: R46_DATA, R46_OUT, SMOKE
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())
import inference_hf as I  # noqa: E402
import PISA_attn_processor as P  # noqa: E402

DATA, OUT = os.environ["R46_DATA"], os.environ["R46_OUT"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
META = json.load(open(f"{DATA}/compose46.json"))
_orig_val = I.log_validation
_orig_attn = P.customized_scaled_dot_product_attention
CALLS = Counter()


def _f(t):
    return t.float() if torch.is_tensor(t) else t


def attn_fp32(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):                     # = bd45.attn_fp32
        out = _orig_attn(_f(query), _f(key), _f(value), _f(weight_matrix), _f(disp_coc), attn_mask=attn_mask,
                         dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard, train=train,
                         occ_map=_f(occ_map))
    assert out.dtype == torch.float32, out.dtype
    CALLS[(str(query.dtype), str(out.dtype))] += 1
    return out.to(value.dtype)


P.customized_scaled_dot_product_attention = attn_fp32


class Hooks:
    """record: store each call's tensor in order; replace: alpha-blend the stored A and S tensors of the same call."""

    def __init__(self):
        self.mode = {"pisa": "off", "latent": "off", "dec": "off"}
        self.store = {"A": {"pisa": [], "latent": [], "dec": []}, "S": {"pisa": [], "latent": [], "dec": []}}
        self.slot, self.k, self.alpha_full, self.alpha_const = None, 0, None, None

    def reset(self, slot=None):
        self.slot, self.k = slot, 0
        if slot:
            for v in self.store[slot].values():
                v.clear()

    def _alpha(self, a_pix, size):
        if self.alpha_const is not None:
            return torch.full((1, 1, *size), float(self.alpha_const), device=a_pix.device)
        return F.interpolate(a_pix.float(), size=size, mode="area")

    def _blend(self, loc, x, a_pix):
        if self.mode[loc] == "record":
            self.store[self.slot][loc].append(x.detach().clone())
            return x
        if self.mode[loc] == "replace":
            k = self.k
            self.k += 1
            hA, hS = self.store["A"][loc][k].float(), self.store["S"][loc][k].float()
            a = self._alpha(a_pix, hA.shape[-2:])
            return (a * hA + (1 - a) * hS).to(x.dtype)
        return x


H = Hooks()


def pisa_hook(module, args, kwargs, output):
    if H.mode["pisa"] == "off":
        return output
    x = output[0] if isinstance(output, tuple) else output.sample
    a_pix = kwargs["cross_attention_kwargs"]["disp_coc"][:, 2:3]
    y = H._blend("pisa", x, a_pix)
    return (y,) + tuple(output[1:]) if isinstance(output, tuple) else type(output)(sample=y)


def latent_hook(module, args):
    if H.mode["latent"] == "off":
        return None
    z = args[0]
    y = H._blend("latent", z, H.alpha_full)
    return (y,) + tuple(args[1:])


def dec_hook(module, args, output):
    if H.mode["dec"] == "off":
        return output
    return H._blend("dec", output, H.alpha_full)


def proc_size(w, h):                                                # = bd40.proc_size
    s = 1024 / min(w, h) if min(w, h) < 1024 else 1.0
    return int(round(w * s / 64)) * 64, int(round(h * s / 64)) * 64


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    dev, dt = accelerator.device, pipeline.vae.dtype
    unet = pipeline.unet.module if hasattr(pipeline.unet, "module") else pipeline.unet
    vae = pipeline.vae.module if hasattr(pipeline.vae, "module") else pipeline.vae
    unet.down_blocks[1].attentions[0].register_forward_hook(pisa_hook, with_kwargs=True)
    vae.decoder.register_forward_pre_hook(latent_hook)
    vae.decoder.mid_block.register_forward_hook(dec_hook)
    pane = np.load(f"{DATA}/maps/pane.npy")
    f = META["focus"]
    rows = []

    def render(inp, dmap, K, alpha, tag, two_channel=False):
        x = Image.open(f"{DATA}/input/{inp}.png").convert("RGB")
        W2, H2 = proc_size(*x.size)
        xs = np.asarray(x.resize((W2, H2), resample=Image.LANCZOS)).astype(np.float32)
        rs = lambda m: np.asarray(Image.fromarray(m.astype(np.float32)).resize((W2, H2), resample=Image.BILINEAR))
        d, a = rs(dmap), rs(alpha)
        ch = [d, np.abs(d - f) * K / 10] + ([] if two_channel else [a])
        dc = torch.from_numpy(np.stack(ch)[None]).to(dev, dt)
        H.alpha_full = torch.from_numpy(a[None, None]).to(dev)
        x_in = torch.from_numpy(xs / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            image, dur = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                   aif_image=x_in, disp_coc=dc, timeit=True)
        torch.cuda.synchronize()
        os.makedirs(os.path.dirname(f"{OUT}/{tag}"), exist_ok=True)
        image.resize(x.size, resample=Image.LANCZOS).save(f"{OUT}/{tag}.png")
        print(tag, f"K {K:.2f}", f"{dur:.2f}s", flush=True)
        return f"{tag}.png", float(time.time() - t0)

    def diff(f1, f2):
        a = np.asarray(Image.open(f"{OUT}/{f1}")).astype(np.int16)
        b = np.asarray(Image.open(f"{OUT}/{f2}")).astype(np.int16)
        return int(np.abs(a - b).max())

    bgs = META["bgs"] + ["flat"]
    if SMOKE:
        bgs = bgs[:1]
    checks = {}
    first = True
    for bg in bgs:
        K = META["backgrounds"][bg]["K"]
        dB = np.load(f"{DATA}/maps/{bg}_dB.npy")
        comps = [c for c, m in META["composites"].items() if m["bg"] == bg]
        if SMOKE:
            comps = comps[:1]
        if bg != "flat":
            fr, s = render(f"{bg}__bgonly", dB, K, np.zeros_like(dB), f"ref/{bg}")
            rows.append(dict(kind="ref", bg=bg, file=fr, K=K, seconds=s))
        for c in comps:
            alpha = np.load(f"{DATA}/alpha/{c}.npy")
            maps = dict(A=np.where(pane > 0.5, f, dB), S=dB, M2=np.where(alpha >= 0.5, f, dB))
            for slot in ["A", "S"]:
                H.mode = {k: "record" for k in H.mode}
                H.reset(slot)
                fn, s = render(c, maps[slot], K, alpha, f"{slot}/{c}")
                rows.append(dict(kind=slot, comp=c, bg=bg, file=fn, K=K, seconds=s))
            H.mode = {k: "off" for k in H.mode}
            fn, s = render(c, maps["M2"], K, alpha, f"M2/{c}")
            rows.append(dict(kind="M2", comp=c, bg=bg, file=fn, K=K, seconds=s))
            for loc in ["pisa", "latent", "dec"]:
                variants = [(None, f"C_{loc}")]
                if first or bg == "flat":
                    variants += [(1.0, f"check/C_{loc}_a1"), (0.0, f"check/C_{loc}_a0")]
                for const, tagdir in variants:
                    H.mode = {k: ("replace" if k == loc else "off") for k in H.mode}
                    H.reset()
                    H.alpha_const = const
                    fn, s = render(c, maps["A"], K, alpha, f"{tagdir}/{c}")
                    H.alpha_const = None
                    if const is None:
                        rows.append(dict(kind=f"C_{loc}", comp=c, bg=bg, file=fn, K=K, seconds=s))
                    else:
                        ref = f"A/{c}.png" if const == 1.0 else f"S/{c}.png"
                        checks[f"{c}|{loc}|alpha{const:g}"] = diff(fn, ref)
            H.mode = {k: "off" for k in H.mode}
            if first:
                fn, _ = render(c, maps["A"], K, alpha, f"check/A_2ch/{c}", two_channel=True)
                checks[f"{c}|zero_intervention_2ch"] = diff(fn, f"A/{c}.png")
                first = False
            json.dump(dict(smoke=SMOKE, rows=rows, checks=checks, calls={"|".join(k): v for k, v in CALLS.items()}),
                      open(f"{OUT}/bd46.json", "w"), indent=1)
    json.dump(dict(smoke=SMOKE, rows=rows, checks=checks, calls={"|".join(k): v for k, v in CALLS.items()}),
              open(f"{OUT}/bd46.json", "w"), indent=1)
    print("bd46 done", len(rows), "renders; checks", checks, "; PISA calls", dict(CALLS), flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
