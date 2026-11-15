"""Proposal smoke tests 2 and 3 (pipeline checks only; not results). Run on a GPU pod after smoke test 1:

    python quarter2/sd_dof_masked/smoke_paired.py paired   # smoke test 2
    python quarter2/sd_dof_masked/smoke_paired.py lora     # smoke test 3 (needs test 2's rendered pairs)

paired: 8 sharp SD 1.5 images (seed 42) -> Depth Anything V2 inverse depth -> rendered depth-dependent background
        blur at 3 levels (max Gaussian sigma 2, 4, 8 px; the nearest 40% of pixels, the meters' foreground, stays
        exactly sharp) -> activations of each original and edited image at round 4's layer and noise level, with the
        same noise and the caption as conditioning -> a paired direction (mean of per-pair changes at the strongest
        level), compared with round 9's directions, and run once through the steered sampler.
lora:   a rank-4 LoRA on the UNet attention, trained 50 steps on the strongest-level edited images with their
        captions (the plain diffusion loss); saved, reloaded into a fresh pipeline, sampled at adapter weights
        0, 0.5 and 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sd_dof_masked as M  # noqa: E402  (also puts round 4's sd_dof on the path)

D = M.D
OUT = M.ROOT / "outputs" / "q2_sd_dof_masked_smoke"
SEED = 42
LEVELS = [2.0, 4.0, 8.0]
CAPS = [c for _, c in M.CAPTIONS] + ["A cat sitting on a windowsill next to a potted plant",
                                     "A man riding a bicycle on a country road"]
LORA_STEPS, LORA_RANK, LORA_LR, LORA_BS = 50, 4, 5e-4, 2
log = M.log


def write(path, L, checks):
    """Write the checks table plus extra lines; return whether every check passed."""
    T = ["| check | result | detail |", "|---|---|---|"]
    T += [f"| {n} | {'PASS' if ok is True else 'FAIL' if ok is False else 'info'} | {d} |" for n, ok, d in checks]
    text = "\n".join([L[0], ""] + T + [""] + L[1:]) + "\n"
    path.write_text(text)
    print(text, flush=True)
    return all(ok is not False for _, ok, _ in checks)


def sheet(rows, path, W=160):
    s = Image.new("RGB", (W * len(rows[0]), W * len(rows)), "white")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            s.paste(im.resize((W, W), Image.LANCZOS), (c * W, r * W))
    s.save(path)
    return len(rows) * len(rows[0])


@torch.no_grad()
def inv_depth(meter, img):
    inp = meter.dproc(images=img, return_tensors="pt").to("cuda")
    pred = meter.depth(**inp).predicted_depth
    pred = torch.nn.functional.interpolate(pred[:, None], size=img.size[::-1], mode="bicubic", align_corners=False)
    return pred[0, 0].float().cpu().numpy()


def render(rgb, invd, level, n=9):
    """Per-pixel Gaussian blur, sigma = level x (normalised distance behind the foreground); foreground untouched."""
    from scipy.ndimage import gaussian_filter
    q, lo = np.quantile(invd, 0.6), invd.min()
    sig = level * np.clip((q - invd) / max(q - lo, 1e-6), 0, 1)
    sigmas = np.linspace(0, level, n)
    stack = np.stack([rgb] + [np.stack([gaussian_filter(rgb[..., c], s) for c in range(3)], -1)
                              for s in sigmas[1:]])
    pos = sig / level * (n - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
    fr = (pos - i0)[..., None]
    yy, xx = np.indices(sig.shape)
    out = (1 - fr) * stack[i0, yy, xx] + fr * stack[i0 + 1, yy, xx]
    return Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8)), invd >= q


@torch.no_grad()
def layer_acts(sd, img, cap, noise_seed):
    """Activation (1280, 16, 16) of an image's latent forward-noised to round 4's sigma, conditioned on its caption."""
    p = sd.pipe
    t = int((sd.sig_all - D.SIG_COLLECT).abs().argmin())
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].float() / 127.5 - 1
    z0 = p.vae.encode(x.to("cuda", sd.dt)).latent_dist.mean * p.vae.config.scaling_factor
    noise = torch.randn(z0.shape, generator=torch.Generator().manual_seed(noise_seed)).to("cuda", sd.dt)
    tt = torch.full((1,), t, device="cuda", dtype=torch.long)
    p.unet(p.scheduler.add_noise(z0, noise, tt), tt, encoder_hidden_states=sd.embed([cap]))
    return sd.hook.last[0].float().cpu()


def paired():
    out = OUT / "paired"
    (out / "images").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    sd, meter = D.SD("cuda"), D.DofMeter("cuda")
    orig = sd.generate(CAPS, [SEED] * len(CAPS), "none", bs=4)
    edits, fgs, pix = {L: [] for L in LEVELS}, [], {L: [] for L in LEVELS}
    for i, im in enumerate(orig):
        im.save(out / "images" / f"{i:02d}_orig.png")
        invd = inv_depth(meter, im)
        rgb = np.asarray(im).astype(float)
        for L in LEVELS:
            e, fg = render(rgb, invd, L)
            e.save(out / "images" / f"{i:02d}_L{L:g}.png")
            edits[L].append(e)
            d = np.abs(np.asarray(e).astype(float) - rgb).mean(2) / 255
            pix[L].append((float(d[fg].mean()), float(d[~fg].mean())))
        fgs.append(fg)
        np.save(out / "images" / f"{i:02d}_fg.npy", np.packbits(fg))
    (out / "captions.json").write_text(json.dumps(CAPS, indent=1))
    log(f"rendered {len(orig)} x {len(LEVELS)} pairs")

    # paired activations: same noise and caption for the original and each edit
    A0 = [layer_acts(sd, im, c, SEED + i) for i, (im, c) in enumerate(zip(orig, CAPS))]
    A = {L: [layer_acts(sd, im, c, SEED + i) for i, (im, c) in enumerate(zip(edits[L], CAPS))] for L in LEVELS}
    same_noise = float((layer_acts(sd, orig[0], CAPS[0], SEED) - A0[0]).abs().max() / A0[0].abs().max())
    delta = {L: [a - a0 for a, a0 in zip(A[L], A0)] for L in LEVELS}
    norms = {L: float(np.mean([d.norm() for d in delta[L]])) for L in LEVELS}
    Lmax = max(LEVELS)
    # where in the 16 x 16 grid the paired change lands: energy at all-background vs all-foreground positions
    spatial = []
    for d, fg in zip(delta[Lmax], fgs):
        e, g = (d ** 2).sum(0).numpy(), M.bg_grid(fg)
        if (g == 1).any() and (g == 0).any():
            spatial.append(float(e[g == 1].mean() / e[g == 0].mean()))
    flat = torch.stack([d.flatten() for d in delta[Lmax]])
    unit = flat / flat.norm(dim=1, keepdim=True)
    cos_pairs = (unit @ unit.T)[~torch.eye(len(unit), dtype=bool)]
    paired_dir = flat.mean(0)
    dirs = torch.load(M.DIRS)
    cos = lambda a, b: float(a.float() @ b.float() / (a.float().norm() * b.float().norm()))
    cos_ref = {k: cos(paired_dir, dirs[k].flatten()) for k in ("rfm_eig1", "meandiff") if k in dirs}
    torch.save({"paired_dir": paired_dir, "level": Lmax}, out / "paired_direction.pt")

    # run the paired direction once through the steered sampler (round 9's s 4)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    steered = sd.generate(CAPS[:2], [SEED] * 2, "rfm", 4.0, paired_dir, bs=2)
    gen_s = (time.time() - t0) / 2
    pr = []
    for i, im in enumerate(steered):
        im.save(out / "images" / f"{i:02d}_paired_s4.png")
        g0 = np.asarray(orig[i].convert("L")).astype(float)
        g1 = np.asarray(im.convert("L")).astype(float)
        pr.append(M.pair(g0, g1, np.asarray(orig[i]).astype(float), np.asarray(im).astype(float), fgs[i]))
    n1 = sheet([[orig[i]] + [edits[L][i] for L in LEVELS] for i in (0, 1, 3, 4)], out / "pairs_examples.png")
    n2 = sheet([[orig[i], steered[i]] for i in range(2)], out / "paired_direction_examples.png")

    fgmax = max(f for L in LEVELS for f, _ in pix[L])
    bgm = [float(np.mean([b for _, b in pix[L]])) for L in LEVELS]
    checks = [
        ("rendered pairs: foreground pixels unchanged (max mean change < 0.005)", fgmax < 0.005, f"max {fgmax:.4f}"),
        ("rendered pairs: background change > 0 and grows with level", bgm[0] > 0 and bgm == sorted(bgm),
         " / ".join(f"{b:.4f}" for b in bgm)),
        ("paired activations: same shape for original and edit",
         all(a.shape == a0.shape == (1280, 16, 16) for L in LEVELS for a, a0 in zip(A[L], A0)), "1280 x 16 x 16"),
        ("paired activations: repeating the original with the same noise gives the same activations", same_noise < 1e-3,
         f"max relative change {same_noise:.2e}"),
        ("activation change grows with level", list(norms.values()) == sorted(norms.values()),
         " / ".join(f"{v:.1f}" for v in norms.values())),
        ("paired direction: cosine with round 9's directions", None,
         ", ".join(f"{k} {v:+.3f}" for k, v in cos_ref.items())),
        ("per-pair changes: mean cosine between pairs (1 = one shared direction)", None,
         f"{float(cos_pairs.mean()):+.3f} (min {float(cos_pairs.min()):+.3f}, max {float(cos_pairs.max()):+.3f})"),
        ("per-pair changes: energy at background / foreground positions (median)", None,
         f"{np.median(spatial):.2f} over {len(spatial)} images" if spatial else "no image has both"),
        ("paired direction runs through the sampler; metrics finite",
         all(np.isfinite(v) for p in pr for v in p.values()),
         "; ".join(f"bg {p['bg']:+.3f} fg {p['fg']:+.3f}" for p in pr) + f"; {gen_s:.1f} s per image, peak "
         f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB"),
        ("contact sheets <= 25 images each", n1 <= 25 and n2 <= 25, f"{n1} and {n2}"),
    ]
    L = [f"# Smoke test 2: paired data ({len(CAPS)} captions x seed {SEED}, levels {LEVELS}; pipeline check)",
         "Rendered pairs: columns original, sigma 2, 4, 8 (`pairs_examples.png`); paired direction at s 4 next to the "
         "original (`paired_direction_examples.png`). The paired direction is the mean of per-pair changes, i.e. the "
         "difference of the two group means: it only differs from a mean-difference direction through the data.",
         f"\nTotal time {time.time() - t_start:.0f} s."]
    (out / "paired.json").write_text(json.dumps(dict(pix=pix, norms=norms, cos_ref=cos_ref, spatial=spatial,
                                                     cos_pairs_mean=float(cos_pairs.mean()), steered=pr), indent=1))
    if not write(out / "paired_checks.md", L, checks):
        sys.exit("smoke test 2: some checks failed")


def lora():
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    src, out = OUT / "paired", OUT / "lora"
    out.mkdir(parents=True, exist_ok=True)
    caps = json.loads((src / "captions.json").read_text())
    Lmax = max(LEVELS)
    t_start = time.time()
    sd = D.SD("cuda")
    p, dt = sd.pipe, sd.dt
    plain = sd.generate(caps[:2], [SEED] * 2, "none", bs=2)       # before any adapter exists
    with torch.no_grad():
        z0 = torch.cat([p.vae.encode((torch.from_numpy(np.asarray(Image.open(src / "images" / f"{i:02d}_L{Lmax:g}.png")))
                                      .permute(2, 0, 1)[None].float() / 127.5 - 1).to("cuda", dt)).latent_dist.mean
                        for i in range(len(caps))]) * p.vae.config.scaling_factor
        emb = torch.cat([sd.embed([c]) for c in caps])
    unet = p.unet
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [q for q in unet.parameters() if q.requires_grad]
    for q in params:
        q.data = q.data.float()
    unet.enable_gradient_checkpointing()
    unet.train()
    opt = torch.optim.AdamW(params, lr=LORA_LR)
    amp = dt == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ac = p.scheduler.alphas_cumprod

    # fixed evaluation loss: every image at 4 timesteps with fixed noise, before and after training
    g = torch.Generator().manual_seed(0)
    ev_noise = torch.randn((len(caps) * 4,) + tuple(z0.shape[1:]), generator=g).to("cuda", dt)
    ev_t = torch.tensor([100, 300, 500, 700] * len(caps), device="cuda")
    ev_idx = torch.arange(len(caps)).repeat_interleave(4).to("cuda")

    def loss_of(idx, t, noise):
        zt = p.scheduler.add_noise(z0[idx], noise, t)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = unet(zt, t, encoder_hidden_states=emb[idx]).sample
        return torch.nn.functional.mse_loss(pred.float(), noise.float())

    @torch.no_grad()
    def eval_loss():
        return float(np.mean([loss_of(ev_idx[j:j + 4], ev_t[j:j + 4], ev_noise[j:j + 4]).item()
                              for j in range(0, len(ev_t), 4)]))

    before = eval_loss()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    gt = torch.Generator().manual_seed(1)
    for step in range(LORA_STEPS):
        idx = torch.randint(0, len(caps), (LORA_BS,), generator=gt).to("cuda")
        t = torch.randint(0, len(ac), (LORA_BS,), generator=gt).to("cuda")
        noise = torch.randn((LORA_BS,) + tuple(z0.shape[1:]), generator=gt).to("cuda", dt)
        loss = loss_of(idx, t, noise)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss))
    train_s, train_gb = time.time() - t0, torch.cuda.max_memory_allocated() / 2 ** 30
    unet.eval()
    after = eval_loss()
    log(f"LoRA: {LORA_STEPS} steps in {train_s:.0f} s; eval loss {before:.4f} -> {after:.4f}")
    StableDiffusionPipeline.save_lora_weights(save_directory=str(out / "adapter"), safe_serialization=True,
                                              unet_lora_layers=convert_state_dict_to_diffusers(
                                                  get_peft_model_state_dict(unet)))
    n_lora = sum(q.numel() for q in params)
    del sd, p, unet, opt, params
    torch.cuda.empty_cache()

    # reload into a fresh pipeline and sample at three adapter weights
    sd2 = D.SD("cuda")
    sd2.pipe.load_lora_weights(str(out / "adapter"), weight_name="pytorch_lora_weights.safetensors",
                               adapter_name="dof")
    ims = {}
    for w in (0.0, 0.5, 1.0):
        sd2.pipe.set_adapters(["dof"], adapter_weights=[w])
        ims[w] = sd2.generate(caps[:2], [SEED] * 2, "none", bs=2)
    arr = lambda im: np.asarray(im).astype(float)
    d0 = max(float(np.abs(arr(a) - arr(b)).mean() / 255) for a, b in zip(ims[0.0], plain))
    d1 = min(float(np.abs(arr(a) - arr(b)).mean() / 255) for a, b in zip(ims[1.0], ims[0.0]))
    fgs = [np.unpackbits(np.load(src / "images" / f"{i:02d}_fg.npy"))[:512 * 512].reshape(512, 512).astype(bool)
           for i in range(2)]
    pr = {w: [M.pair(np.asarray(a.convert("L")).astype(float), np.asarray(b.convert("L")).astype(float), arr(a), arr(b),
                     fg) for a, b, fg in zip(plain, ims[w], fgs)] for w in (0.5, 1.0)}
    n = sheet([[plain[i], ims[0.0][i], ims[0.5][i], ims[1.0][i]] for i in range(2)], out / "lora_examples.png")
    checks = [
        ("training runs; loss finite", all(np.isfinite(losses)),
         f"{LORA_STEPS} steps, batch {LORA_BS}, {n_lora:,} LoRA parameters, {train_s:.0f} s, peak {train_gb:.2f} GB"),
        ("fixed evaluation loss goes down", after < before, f"{before:.4f} -> {after:.4f}"),
        ("adapter saved and reloaded into a fresh pipeline", (out / "adapter" / "pytorch_lora_weights.safetensors")
         .exists(), "pytorch_lora_weights.safetensors"),
        ("adapter weight 0 reproduces the plain model (mean pixel change < 0.004)", d0 < 0.004, f"max {d0:.4f}"),
        ("adapter weight 1 changes the image (mean pixel change > 0.01)", d1 > 0.01, f"min {d1:.4f}"),
        ("paired changes vs plain at weights 0.5 / 1 (bg, fg)", None,
         " / ".join(f"bg {np.mean([q['bg'] for q in pr[w]]):+.3f} fg {np.mean([q['fg'] for q in pr[w]]):+.3f}"
                    for w in (0.5, 1.0))),
        ("contact sheet <= 25 images", n <= 25, f"{n} images"),
    ]
    L = [f"# Smoke test 3: LoRA on the rendered pairs ({len(caps)} images at sigma {Lmax:g}; pipeline check)",
         "Columns in `lora_examples.png`: plain model, adapter weight 0, 0.5, 1 (same caption and seed). 50 steps "
         "on 8 images only checks the training, saving and loading path; it is not a trained baseline.",
         f"\nTotal time {time.time() - t_start:.0f} s."]
    (out / "lora.json").write_text(json.dumps(dict(losses=losses, before=before, after=after, d0=d0, d1=d1,
                                                   pairs={str(k): v for k, v in pr.items()}), indent=1))
    if not write(out / "lora_checks.md", L, checks):
        sys.exit("smoke test 3: some checks failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["paired", "lora"])
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    {"paired": paired, "lora": lora}[a.stage]()


if __name__ == "__main__":
    main()
