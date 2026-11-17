"""Round 28 driver: where is BokehDiff's content-dependent kernel shape decided, and is it physical (linear-light) bokeh?

Run from the BokehDiff repo directory (pinned commit), round 17 venv; started like rounds 18-27.
Configuration = round 25's baseline (fp32 PISA, original gate, frozen radius curve), corrected runs on the 12
held-out patches x positions 0/2/4 x targets 3-15.5 px. The pipeline's VAE decode is wrapped to capture the final
latents; each output is decoded twice:
  ft    BokehDiff's fine-tuned VAE (the normal output; must equal round 25's image)
  orig  the original RealVisXL V5.0 VAE (loaded separately, fp32)
Decoder-only pathway: known blurs applied in pixel space (sRGB-space disk and linear-light disk) are encoded with
BokehDiff's VAE encoder and decoded with both decoders.
"""
import json
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

OUT = os.environ["R28_OUT"]
R25 = os.environ["R25_OUT"]
TARGETS = [3, 4, 5, 6, 8, 11.625, 15.5]
PATH_TARGETS = [5, 8, 11.625]
GAMMA = 2.2
_orig_val = I.log_validation
_orig_attn = PA.customized_scaled_dot_product_attention
CAP = {}


def attn(query, key, value, weight_matrix=None, disp_coc=None, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, hard=1.0, train=True, occ_map=None):
    with torch.autocast("cuda", enabled=False):          # identical to round 25's baseline path
        return _orig_attn(query.float(), key.float(), value.float(), weight_matrix.float(), disp_coc.float(),
                          attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, hard=hard,
                          train=train, occ_map=occ_map.float()).to(value.dtype)


PA.customized_scaled_dot_product_attention = attn


def lin_blur(img_u8, r):
    x = (img_u8.astype(np.float64) / 255.0) ** GAMMA
    b = m18.blur(x, m18.disk(r))
    return np.clip(b, 0, 1) ** (1 / GAMMA) * 255.0


def groups(lib):
    ps = [p for p in lib if p["split"] == "test"]
    return [ps[0::2], ps[1::2]]


def hijack(pipeline, seed, train_T, accelerator, validation_prompt="an excellent photo with a large aperture",
           aif_image=None, disp_coc=None, timeit=False):
    from diffusers import AutoencoderKL
    dev, dt = accelerator.device, pipeline.vae.dtype
    os.makedirs(f"{OUT}/img", exist_ok=True)
    vae_ft = pipeline.vae
    vae_orig = AutoencoderKL.from_pretrained("SG161222/RealVisXL_V5.0", subfolder="vae").to(dev, torch.float32).eval()
    # which parts of the VAE did BokehDiff change?
    sd_o = vae_orig.state_dict()
    diff = {"encoder": 0, "decoder": 0, "other": 0}
    tot = {"encoder": 0, "decoder": 0, "other": 0}
    for k, v in vae_ft.state_dict().items():
        part = "encoder" if k.startswith(("encoder", "quant_conv")) else "decoder" if k.startswith(("decoder", "post_quant_conv")) else "other"
        tot[part] += 1
        if k in sd_o and not torch.equal(v.cpu(), sd_o[k].to(v.dtype).cpu()):
            diff[part] += 1
    print("VAE tensors changed by BokehDiff:", diff, "of", tot, flush=True)

    orig_decode = vae_ft.decode

    def capture_decode(z, return_dict=True, **kw):
        CAP["z"] = z.detach().clone()
        return orig_decode(z, return_dict=return_dict, **kw)

    vae_ft.decode = capture_decode

    def decode_with(vae, z):
        """decode raw (unscaled) latents in fp32; restores the VAE's dtype afterwards"""
        d0 = next(vae.parameters()).dtype
        vae.to(torch.float32)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            x = (orig_decode if vae is vae_ft else vae.decode)(z.to(torch.float32), return_dict=False)[0]
        vae.to(d0)
        x = (x / 2 + 0.5).clamp(0, 1)
        return np.uint8(255 * x[0].permute(1, 2, 0).cpu().numpy())

    def encode_ft(img_u8):
        x = torch.from_numpy(img_u8.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            vae_ft.to(torch.float32)
            z = vae_ft.encode(x).latent_dist.mode()          # raw latents = what the pipeline passes to decode
        vae_ft.to(dt)
        return z

    lib = PT.library()
    c25 = json.load(open(f"{R25}/calib.json"))["C1"]["G0"]
    curve = M.Curve.__new__(M.Curve)
    curve.x, curve.y = np.array(c25["x"]), np.array(c25["y"])
    gp = lambda r: r / (D.NOM * D.K)
    conds, repro = [], []
    for gi, grp in enumerate(groups(lib)):
        for rot in (0, 2, 4):
            img, cid, where = PT.layout(grp, rot)
            for t in TARGETS:
                r, _ = curve.inverse(t)
                disp, coc = PT.maps(cid, {k: ("coc", r / (D.NOM * 10), PT.SIGN[k] * gp(t)) for k in PT.FREE})
                x = torch.from_numpy(img.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1)[None].to(dev, dt)
                dc = torch.from_numpy(np.stack([disp, coc])[None]).to(dev, dt)
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                with torch.autocast("cuda", dtype=torch.float16):
                    image, _ = _orig_val(pipeline, 0, train_T, accelerator, validation_prompt=validation_prompt,
                                         aif_image=x.to(pipeline.vae.dtype), disp_coc=dc, timeit=True)
                a = np.asarray(image)
                tag = f"g{gi}_r{rot}_t{t:.1f}"
                Image.fromarray(a).save(f"{OUT}/img/{tag}_ft.png")
                b = decode_with(vae_orig, CAP["z"])
                Image.fromarray(b).save(f"{OUT}/img/{tag}_orig.png")
                ref = np.asarray(Image.open(f"{R25}/img/cor_g{gi}_r{rot}_G0_t{t:.1f}.png")).astype(np.float32)
                repro.append(float(np.abs(a.astype(np.float32) - ref).max()))
                conds.append(dict(tag=tag, group=gi, rot=rot, target=t, where={str(k): v for k, v in where.items()}))
            # decoder-only pathway on known pixel-space blurs
            for t in PATH_TARGETS:
                for kind in ("srgb", "lin"):
                    blurred = (np.clip(m18.blur(img.astype(np.float64), m18.disk(t)), 0, 255) if kind == "srgb" else lin_blur(img, t)).round().astype(np.uint8)
                    z = encode_ft(blurred)
                    for name, vae in (("ft", vae_ft), ("orig", vae_orig)):
                        Image.fromarray(decode_with(vae, z)).save(f"{OUT}/img/path_g{gi}_r{rot}_{kind}{t:.1f}_{name}.png")
        print("group", gi, "done", flush=True)
    meta = dict(conds=conds, repro_max_diff=max(repro), vae_changed=diff, vae_total=tot)
    json.dump(meta, open(f"{OUT}/conds.json", "w"))
    print("done:", len(conds), "generations; reproduces round 25: max |diff|", max(repro), flush=True)
    sys.stdout.flush()
    os._exit(0)


I.log_validation = hijack
if __name__ == "__main__":
    I.main()
