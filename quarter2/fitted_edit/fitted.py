"""Q2 round 16: can a small spatial activation edit blur the background without changing the subject, even when it is
fitted separately for each image? A privileged feasibility test (see README.md). Run on a GPU pod:

    python quarter2/fitted_edit/fitted.py gen     # scenes, fixed subject masks, rendered targets, basis, probe
    python quarter2/fitted_edit/fitted.py fit     # tuned masked RFM, fitted global and spatial edits, controls, report

Fixed for every method: SD 1.5, caption, seed, text CFG 7.5, 50-step DDIM, the unedited image's subject mask
(checked by eye after `gen`, then never recomputed). Targets: the unedited image with its background Gaussian-blurred
(round 12's post-hoc render, subject pasted back with a feathered alpha) at the background log-sharpness changes
-0.5 and -2.0. The targets are diagnostic endpoints, not evidence that steering beats rendering.

Fitted edits (privileged: the optimizer sees the target image):
  layer   up_blocks[2].resnets[-1] of the U-Net decoder (640 channels at 32 x 32; round 4 edited a 1280 x 16 x 16
          encoder-side block)
  edit    h(x) <- h(x) + |h(x)| sum_r a_r(x, t) b_r, on both CFG rows; |h(x)| is the channel norm at position x
  basis   b_1..b_8: a shared channel basis (top 8 principal directions of the layer's feature change between the
          unedited images and their targets, forward-noised with the same noise; pooled over the 8 scenes)
  global  a_r(x, t) = a_r(seg(t)) m(x): 8 coefficients per time segment, masked to the background (fraction of non-
          subject pixels in each cell), like the tuned masked RFM
  spatial a_r(x, t) = A_r(seg(t)) on a 16 x 16 grid, bilinearly upsampled to 32 x 32; free to vary across the image
  time    steps 17-49 of 50 (sigma 2.65 -> 0.04; the edit starts from the cached unedited latent at step 17), two
          segments 17-32 and 33-49 with their own coefficients
  loss    on the decoded final image x (exact gradient through every sampling step and the VAE decoder):
          MSE_bg(x, T)/MSE_bg(ref, T) + ((dlogsharp_bg(x) - target)/0.15)^2 + MSE_subject(x, T)/3^2
          + MSE_band(x, T)/max(MSE_band(ref, T), 3^2)          (pixels 0-255)
  budget  Adam, the same number of iterations and learning rate for every fit; the lowest-loss iterate is kept
Baseline: round 13's masked RFM (rfm_eig1 at down_blocks[2], third CFG branch, s 4), strength tuned per scene and
target in each of round 13's windows (full, early, middle, late); the best window that reaches the target is used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof_timing"))
import timing as TM  # noqa: E402  (round 13: Sampler, windows, layout; brings rounds 12, 11, 10 and 4)

G, D, M, P, PL = TM.G, TM.D, TM.M, TM.P, TM.PL
OUT = ROOT / "outputs" / "q2_fitted_edit"
SEED = 42
# (kind, caption, CLIPSeg noun, threshold); fixed before generation
SCENES = [
    ("person", "A woman holding a red umbrella on a city street", "woman", 0.4),
    ("animal", "A dog sitting in front of a wooden fence in a garden", "dog", 0.4),
    ("animal, cluttered", "A deer standing in a forest, the whole forest in sharp focus", "deer", 0.4),
    ("small subject", "A small bird standing far away on a long empty beach", "small bird", 0.3),
    ("off-centre", "A horse standing at the far left edge of a wide green pasture, rolling hills behind it", "horse",
     0.4),
    ("object", "A red ceramic coffee mug on a wooden table in a busy cafe", "coffee mug", 0.4),
    ("hair boundary", "A woman with long curly hair standing in a field of tall grass", "woman", 0.4),
    ("thin boundary", "A man riding a bicycle on a country road", "man riding a bicycle", 0.4),
]
OFFC_SEEDS = [42, 43, 44, 45, 46]   # off-centre scene: first seed whose subject centre is >= 20% of the width off centre
# set only after the manual mask check (README): scene 0's "woman" mask left out the umbrella she holds
MASK_FIX = {0: ("woman+red umbrella", 0.4)}   # "a+b": union of the CLIPSeg masks for a and b
TARGETS = [-0.5, -2.0]
TOL, SUBJ_MAX, BG_AGREE = 0.15, 3.0, 0.5
K0, SPLIT, NSTEP = 17, 33, 50
R, GRID, C_LAYER, H_LAYER = 8, 16, 640, 32
BASIS_STEPS = [20, 28, 36, 44]
N_ITER, LR = 24, 0.08              # set from the probe, revised after the first optimization control failed (README)
LR_RES_GRID = [0.01, 0.02, 0.04]   # spatial per-cell residual step size: chosen on control scene 0 only
CTRL_SCENES, CTRL_STD = [0, 3], 0.1
log, arr = G.log, G.arr


# ---------------------------------------------------------------- the edit and a differentiable sampler
class EditHook:
    def __init__(self, module):
        self.f, self.k, self.last = None, None, None
        module.register_forward_hook(self.fn)

    def fn(self, mod, inp, out):
        self.last = out
        if self.f is None:
            return None
        return out + self.f(out, self.k)


def make_edit(B, coef, kind, bgm):
    """B (R, C) unit rows; coef (2, R) for global or (2, R, GRID, GRID) for spatial; bgm (1, 1, 32, 32)."""
    def f(h, k):
        c = coef[0 if k < SPLIT else 1]
        if kind == "global":
            a = c.view(1, R, 1, 1) * bgm
        else:
            a = F.interpolate(c[None], size=h.shape[-2:], mode="bilinear", align_corners=False)
        e = torch.einsum("brhw,rc->bchw", a, B)
        return h.norm(dim=1, keepdim=True) * e
    return f


class Fitter:
    def __init__(self, sd):
        self.sd, self.p = sd, sd.pipe
        self.ts = list(self.p.scheduler.timesteps)
        self.uncond = sd.embed([""])
        self.hook = EditHook(self.p.unet.up_blocks[2].resnets[-1])
        for m in (self.p.unet, self.p.vae, self.p.text_encoder):
            m.requires_grad_(False)
        self.p.unet.enable_gradient_checkpointing()
        self.p.vae.enable_gradient_checkpointing()

    def train(self, on):
        self.p.unet.train(on)                        # dropout is 0 in SD 1.5; train mode only switches checkpointing
        self.p.vae.train(on)

    @torch.no_grad()
    def unsteered(self, cap, seed):
        p, cond = self.p, self.sd.embed([cap])
        z = torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(seed))[None]
        z = (z * p.scheduler.init_noise_sigma).to(self.sd.dev, self.sd.dt)
        zs = []
        for k in range(NSTEP):
            zs.append(z.clone())
            z = self.step(z, k, cond)
        return zs, cond, self.decode_img(z)

    def step(self, z, k, cond):
        t = self.ts[k]
        self.hook.k = k
        out = self.p.unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([self.uncond, cond])).sample
        eu, ec = out[:1], out[1:]
        return self.p.scheduler.step(eu + D.CFG * (ec - eu), t, z).prev_sample

    def decode(self, z):
        x = self.p.vae.decode(z / self.p.vae.config.scaling_factor).sample.float()
        return (x.clamp(-1, 1) + 1) * 127.5                       # (1, 3, 512, 512), 0-255

    @torch.no_grad()
    def decode_img(self, z):
        x = self.decode(z).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(x)

    @torch.no_grad()
    def run(self, zs, cond, f, keep=False):
        self.hook.f = f
        z, traj = zs[K0], []
        for k in range(K0, NSTEP):
            traj.append(z)
            z = self.step(z, k, cond)
        self.hook.f = None
        return (z, traj) if keep else z

    def loss_grad(self, zs, cond, f, loss):
        """Exact gradient of loss(decoded final image) w.r.t. the edit's parameters: re-run each step with grad."""
        self.train(True)
        zT, traj = self.run(zs, cond, f, keep=True)
        zT = zT.detach().requires_grad_(True)
        L, parts = loss(self.decode(zT))
        L.backward()
        g = zT.grad
        self.hook.f = f
        for k in range(NSTEP - 1, K0 - 1, -1):
            zi = traj[k - K0].detach().requires_grad_(True)
            zo = self.step(zi, k, cond)
            zo.backward(g)
            g = zi.grad
        self.hook.f = None
        self.train(False)
        return float(L), parts

    def fit(self, zs, cond, B, kind, bgm, loss, n_iter, lr, lr_res=None):
        """global: coef (2, R), step lr. spatial: coef = shared (2, R) + per-cell residual (2, R, GRID, GRID); the
        shared part gets step lr, the residual lr_res (same set of edits, better-conditioned first steps)."""
        sh = torch.zeros(2, R, device="cuda", requires_grad=True)
        groups = [dict(params=[sh], lr=lr)]
        if kind == "spatial":
            res = torch.zeros(2, R, GRID, GRID, device="cuda", requires_grad=True)
            groups.append(dict(params=[res], lr=lr_res))
        opt = torch.optim.Adam(groups)
        best, hist = (float("inf"), None, -1), []
        for it in range(n_iter):
            opt.zero_grad()
            coef = sh if kind == "global" else sh[..., None, None] + res
            L, parts = self.loss_grad(zs, cond, make_edit(B, coef, kind, bgm), loss)
            hist.append(dict(it=it, loss=L, **parts, gnorm=float(sum(g["params"][0].grad.norm() ** 2
                                                                    for g in groups) ** 0.5)))
            if L < best[0]:
                best = (L, coef.detach().clone(), it)
            opt.step()
        return best, hist


# ---------------------------------------------------------------- loss on the decoded image
def lapvar_t(x, m):
    g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(g[:, None], (1, 1, 1, 1), mode="replicate"), k)[0, 0]
    return torch.log(lap[m].var() + 1e-6)


class Loss:
    def __init__(self, ref, T, R_, target):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().cuda()
        self.ref, self.T = t(ref).permute(2, 0, 1)[None], t(T).permute(2, 0, 1)[None]
        self.m = {k: torch.from_numpy(R_[k]).cuda() for k in ("background", "subject", "band")}
        self.bg0 = float(self.mse(self.ref, "background"))
        self.band0 = max(float(self.mse(self.ref, "band")), SUBJ_MAX ** 2)
        self.lv0 = float(lapvar_t(self.ref, self.m["background"]))
        self.target = target

    def mse(self, x, k):
        return ((x - self.T) ** 2).mean(1)[0][self.m[k]].mean()

    def __call__(self, x):
        parts = dict(bg=self.mse(x, "background") / max(self.bg0, 1e-6),
                     sharp=((lapvar_t(x, self.m["background"]) - self.lv0 - self.target) / TOL) ** 2,
                     subj=self.mse(x, "subject") / SUBJ_MAX ** 2,
                     band=self.mse(x, "band") / self.band0)
        return sum(parts.values()), {k: float(v) for k, v in parts.items()}


# ---------------------------------------------------------------- scenes, masks, targets
def scene_dir():
    d = OUT / "scenes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def seg_union(seg, img, noun, thr):
    ms = [TM.seg_at(seg, img, n, thr)[0] for n in noun.split("+")]
    return np.logical_or.reduce(ms), None


def layout(seg, noun, thr, ref_mask, img):
    m, _ = seg_union(seg, img, noun, thr)
    if not m.any():
        return 0.0, float("nan")
    iou = float((m & ref_mask).sum() / max((m | ref_mask).sum(), 1))
    return iou, float(np.linalg.norm(np.array(np.nonzero(m)).mean(1) - np.array(np.nonzero(ref_mask)).mean(1)))


def find_mask(seg, img, noun, thr):
    m, _ = seg_union(seg, img, noun, thr)
    tries = [dict(thr=thr, area=float(m.mean()))]
    for t2 in (0.3, 0.2, 0.1):
        if 0.0005 <= m.mean() <= 0.8 or t2 >= thr:
            continue
        m, _ = seg_union(seg, img, noun, t2)
        tries.append(dict(thr=t2, area=float(m.mean())))
        thr = t2
    ok = 0.0005 <= m.mean() <= 0.8
    return (m if ok else None), thr, tries


def render_target(ref_rgb, subj, R_, T):
    """Bisection on the Gaussian radius so that the background log-sharpness change equals T."""
    lv0 = G.lapvar(ref_rgb, R_["background"])
    ch = lambda r: G.lapvar(arr(G.to_img(G.posthoc(ref_rgb, subj, r))), R_["background"]) - lv0
    lo, hi = 0.02, 8.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ch(mid) > T else (lo, mid)
    r = 0.5 * (lo + hi)
    return r, arr(G.to_img(G.posthoc(ref_rgb, subj, r))), ch(r)


def load_scene(i):
    d = OUT / "scenes"
    meta = json.loads((d / "scenes.json").read_text())[i]
    subj = M.mask(d / f"{i}_subject.npy")
    return meta, subj


def gen():
    out = scene_dir()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    D.Hook = PL.ComboHook
    sd, seg = D.SD("cuda"), G.Seg("cuda")
    fz = Fitter(sd)
    metas, overlays = [], []
    for i, (kind, cap, noun, thr) in enumerate(SCENES):
        noun, thr = MASK_FIX.get(i, (noun, thr))
        seeds = OFFC_SEEDS if kind == "off-centre" else [SEED]
        for seed in seeds:
            zs, cond, ref = fz.unsteered(cap, seed)
            subj, thr_used, tries = find_mask(seg, ref, noun, thr)
            off = None
            if subj is not None:
                off = float(abs(np.nonzero(subj)[1].mean() - 256) / 512)
            if kind != "off-centre" or (off is not None and off >= 0.2):
                break
        meta = dict(scene=i, kind=kind, caption=cap, noun=noun, seed=seed, thr=thr_used, tries=tries,
                    mask_ok=subj is not None, offcentre=off)
        if subj is None:
            metas.append(meta)
            log(f"scene {i}: no subject mask")
            continue
        ref.save(out / f"{i}_ref.png")
        np.save(out / f"{i}_subject.npy", np.packbits(subj))
        R_ = G.regions(subj)
        rgb = arr(ref)
        meta.update(area=float(subj.mean()), interior_px=int(R_["interior"].sum()), targets={})
        for T in TARGETS:
            r, Timg, ach = render_target(rgb, subj, R_, T)
            G.to_img(Timg).save(out / f"{i}_target_T{abs(T):g}.png")
            meta["targets"][str(T)] = dict(radius=r, achieved=ach,
                                           bg_mae_ref=float(np.abs(rgb - Timg)[R_["background"]].mean()),
                                           band_mae_ref=float(np.abs(rgb - Timg)[R_["band"]].mean()))
        metas.append(meta)
        overlays.append(G.overlay(ref, subj))
        log(f"scene {i} ({kind}): seed {seed}, mask {noun}@{thr_used} area {subj.mean():.4f} off-centre {off}; "
            + ", ".join(f"T{k}: r {v['radius']:.3f} bg {v['achieved']:+.3f}" for k, v in meta["targets"].items()))
    (out / "scenes.json").write_text(json.dumps(metas, indent=1))
    P.sheet([overlays[:4], overlays[4:8]], OUT / "masks_check.png", W=256)

    basis(fz, metas)
    probe(fz, metas)
    log(f"gen done in {time.time() - t0:.0f} s")


@torch.no_grad()
def basis(fz, metas):
    """Shared channel basis: principal directions of the layer's feature change, unedited -> target."""
    p, d = fz.p, OUT / "scenes"
    cov, n = torch.zeros(C_LAYER, C_LAYER, device="cuda", dtype=torch.float64), 0
    g = torch.Generator().manual_seed(0)
    for meta in metas:
        if not meta["mask_ok"]:
            continue
        i, cond = meta["scene"], fz.sd.embed([meta["caption"]])
        enc = lambda im: p.vae.encode((torch.from_numpy(arr(im)).permute(2, 0, 1)[None].float() / 127.5 - 1)
                                      .cuda()).latent_dist.mean * p.vae.config.scaling_factor
        z_ref = enc(Image.open(d / f"{i}_ref.png"))
        for T in TARGETS:
            z_t = enc(Image.open(d / f"{i}_target_T{abs(T):g}.png"))
            for k in BASIS_STEPS:
                t = fz.ts[k]
                noise = torch.randn(z_ref.shape, generator=g).cuda()
                fs = []
                for z0 in (z_ref, z_t):
                    zt = p.scheduler.add_noise(z0, noise, t.view(1))
                    p.unet(zt, t, encoder_hidden_states=cond)
                    fs.append(fz.hook.last.float())
                v = (fs[1] - fs[0])[0].flatten(1).T.double()           # (1024, 640)
                cov += v.T @ v
                n += v.shape[0]
    ev, evec = torch.linalg.eigh(cov / n)
    B = evec[:, -R:].flip(1).T.float().contiguous()                     # (R, 640), unit rows
    frac = (ev[-R:].flip(0) / ev.sum()).tolist()
    torch.save(dict(B=B.cpu(), frac=frac), OUT / "basis.pt")
    log(f"basis: top-{R} variance fractions {[round(x, 3) for x in frac]} (sum {sum(frac):.3f})")


def bg_cells(subj):
    k = subj.shape[0] // H_LAYER
    g = (~subj).reshape(H_LAYER, k, H_LAYER, k).mean((1, 3)).astype(np.float32)
    return torch.from_numpy(g).view(1, 1, H_LAYER, H_LAYER).cuda()


def setup(fz, i):
    meta, subj = load_scene(i)
    zs, cond, ref = fz.unsteered(meta["caption"], meta["seed"])
    saved = arr(Image.open(OUT / "scenes" / f"{i}_ref.png"))
    R_ = G.regions(subj)
    tg = {T: arr(Image.open(OUT / "scenes" / f"{i}_target_T{abs(T):g}.png")) for T in TARGETS}
    return meta, subj, R_, zs, cond, ref, float(np.abs(arr(ref) - saved).max()), tg


def probe(fz, metas):
    """Speed, memory, determinism, gradient check and edit scale on scene 0; sets nothing, only informs the budget."""
    B = torch.load(OUT / "basis.pt")["B"].cuda()
    meta, subj, R_, zs, cond, ref, redo, tg = setup(fz, 0)
    bgm, T = bg_cells(subj), TARGETS[0]
    loss = Loss(arr(ref), tg[T], R_, T)
    res = dict(regenerate_max_diff=redo)
    z0 = fz.run(zs, cond, None)
    zz = fz.run(zs, cond, make_edit(B, torch.zeros(2, R, device="cuda"), "global", bgm))
    res["zero_edit_max_diff"] = float(np.abs(arr(fz.decode_img(z0)) - arr(fz.decode_img(zz))).max())
    res["cached_vs_full_max_diff"] = float(np.abs(arr(fz.decode_img(z0)) - arr(ref)).max())

    torch.cuda.synchronize()
    t = time.time()
    fz.run(zs, cond, None)
    torch.cuda.synchronize()
    res["sec_forward_window"] = time.time() - t
    for kind in ("global", "spatial"):
        shape = (2, R) if kind == "global" else (2, R, GRID, GRID)
        coef = (0.01 * torch.randn(shape, generator=torch.Generator().manual_seed(1))).cuda().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        L, _ = fz.loss_grad(zs, cond, make_edit(B, coef, kind, bgm), loss)
        torch.cuda.synchronize()
        res[f"{kind}_sec_iter"] = time.time() - t
        res[f"{kind}_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        gr = coef.grad.detach().clone()
        v = torch.randn(shape, generator=torch.Generator().manual_seed(2)).cuda()
        v = v / v.norm()
        eps = 2e-3
        with torch.no_grad():
            Lp = float(loss(fz.decode(fz.run(zs, cond, make_edit(B, coef.detach() + eps * v, kind, bgm))))[0])
            Lm = float(loss(fz.decode(fz.run(zs, cond, make_edit(B, coef.detach() - eps * v, kind, bgm))))[0])
        fd, ad = (Lp - Lm) / (2 * eps), float((gr * v).sum())
        res[f"{kind}_gradcheck"] = dict(L=L, finite_diff=fd, autograd=ad, rel_err=abs(fd - ad) / max(abs(fd), 1e-12))
        log(f"probe {kind}: {res[f'{kind}_sec_iter']:.1f} s/iter, peak {res[f'{kind}_peak_gb']:.2f} GB, "
            f"grad fd {fd:.4g} vs autograd {ad:.4g}")
    # edit scale: background change and subject change for a global edit along each of the first 2 basis vectors
    sc = []
    for r_ in range(2):
        for a in (-0.4, -0.2, -0.1, 0.1, 0.2, 0.4):
            c = torch.zeros(2, R, device="cuda")
            c[:, r_] = a
            im = arr(fz.decode_img(fz.run(zs, cond, make_edit(B, c, "global", bgm))))
            sc.append(dict(basis=r_, a=a, bg=G.lapvar(im, R_["background"]) - G.lapvar(arr(ref), R_["background"]),
                           int_mae=float(np.abs(im - arr(ref))[R_["interior"]].mean())))
    res["scale"] = sc
    log(f"probe: {json.dumps(res)}")
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- scoring and baselines
def score(ref, Timg, R_, img):
    a = arr(img)
    lv = lambda x: G.lapvar(x, R_["background"])
    mae = lambda x, y, k: float(np.abs(x - y)[R_[k]].mean())
    return dict(bg=lv(a) - lv(ref), int_mae=mae(a, ref, "interior"), subj_mae=mae(a, ref, "subject"),
                band_mae=mae(a, Timg, "band"), bg_mae_T=mae(a, Timg, "background"),
                bg_mae_ref=mae(a, ref, "background"), bg_mae_refT=mae(ref, Timg, "background"))


def rfm_baseline(sd, smp, d, meta, subj, R_, ref, tg):
    """Round 13's tuning (secant on c, <= 4 generations) in each window; returns every window's best try."""
    old, TM.SEED = TM.SEED, meta["seed"]          # round 13's sampler reads its module seed
    try:
        zs, cond, ref13 = smp.unsteered(meta["caption"])
    finally:
        TM.SEED = old
    bgw = torch.from_numpy(M.bg_grid(subj)).view(1, 1, 16, 16).cuda()
    win, out = TM.windows(sd), {}
    for T in TARGETS:
        for wn, steps in win.items():
            tried, c = [], 1.0 if T > -1 else 2.0
            for _ in range(4):
                im = smp.steered(zs, cond, steps, d, bgw, c)
                m = score(arr(ref), tg[T], R_, im)
                tried.append((c, m, im))
                if abs(m["bg"] - T) <= 0.05:
                    break
                pts = sorted([(0.0, 0.0)] + [(cc, mm["bg"]) for cc, mm, _ in tried])
                cn = M.interp([q[0] for q in pts], [q[1] for q in pts], T)
                if cn is None:
                    cc, mm, _ = max(tried, key=lambda x: x[0])
                    cn = cc * min(T / mm["bg"], 4.0) if mm["bg"] < -1e-3 else 4 * cc
                cn = float(np.clip(cn, 0.05, TM.C_MAX))
                if any(abs(cn - cc) < 1e-6 for cc, _, _ in tried):
                    break
                c = cn
            c, m, im = min(tried, key=lambda x: abs(x[1]["bg"] - T))
            m.update(c=c, window=wn)
            out[(wn, T)] = (m, im)
    return out, float(np.abs(arr(ref13) - arr(ref)).max())


def crit(m, T):
    return dict(reach=abs(m["bg"] - T) <= TOL, subject=m["int_mae"] <= SUBJ_MAX,
                agree=m["bg_mae_T"] <= BG_AGREE * m["bg_mae_refT"])


def fit_all():
    out = OUT / "fit"
    (out / "images").mkdir(parents=True, exist_ok=True)
    assert N_ITER and LR and CTRL_STD, "set N_ITER, LR and CTRL_STD from the probe (README) before fitting"
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    D.Hook = PL.ComboHook
    sd, seg = D.SD("cuda"), G.Seg("cuda")
    fz, smp = Fitter(sd), TM.Sampler(sd)
    B = torch.load(OUT / "basis.pt")["B"].cuda()
    d = (lambda v: (v.float() / v.float().norm()).view(1, 1280, 16, 16).cuda())(torch.load(M.DIRS)["rfm_eig1"])
    metas = json.loads((OUT / "scenes" / "scenes.json").read_text())
    rows, ctrl = [], []
    prev = json.loads((out / "fit.json").read_text())["rows"] if (out / "fit.json").exists() else []
    done_ids = {r["scene"] for r in prev}
    rows += prev

    # optimization control: can the same budget recover a known, achievable spatial edit? The residual step size is
    # chosen on control scene 0 only (lowest recovered MSE), then must pass on control scene 3 too.
    if not (out / "control.json").exists():
        for i, lr_res in [(CTRL_SCENES[0], x) for x in LR_RES_GRID] + [(CTRL_SCENES[1], None)]:
            if lr_res is None:
                lr_res = min(ctrl, key=lambda c: c["rel_mse"])["lr_res"]
            meta, subj, R_, zs, cond, ref, redo, tg = setup(fz, i)
            bgm = bg_cells(subj)
            gt = (CTRL_STD * torch.randn(2, R, GRID, GRID, generator=torch.Generator().manual_seed(7))).cuda()
            Y = arr(fz.decode_img(fz.run(zs, cond, make_edit(B, gt, "spatial", bgm))))
            Yt = torch.from_numpy(Y).float().cuda().permute(2, 0, 1)[None]
            e0 = float(((fz.decode(fz.run(zs, cond, None)) - Yt) ** 2).mean())
            lossc = lambda x: (lambda v: (v, dict(mse=float(v))))(((x - Yt) ** 2).mean() / e0)
            (L, coef, it), hist = fz.fit(zs, cond, B, "spatial", bgm, lossc, N_ITER, LR, lr_res)
            im = arr(fz.decode_img(fz.run(zs, cond, make_edit(B, coef, "spatial", bgm))))
            rel = float(((im - Y) ** 2).mean()) / max(float(((arr(ref) - Y) ** 2).mean()), 1e-9)
            ctrl.append(dict(scene=i, lr_res=lr_res, rel_mse=rel, best_it=it, int_mae_gt_vs_ref=float(np.abs(Y - arr(ref))[
                R_["interior"]].mean()), hist=[h["loss"] for h in hist]))
            log(f"control scene {i} lr_res {lr_res}: recovered MSE {rel:.3f} of the zero-edit MSE (best iter {it})")
        (out / "control.json").write_text(json.dumps(ctrl, indent=1))
    ctrl_all = json.loads((out / "control.json").read_text())
    LR_RES = ctrl_all[-1]["lr_res"]
    ctrl = [c for c in ctrl_all if c["lr_res"] == LR_RES]
    log(f"chosen residual step {LR_RES}; controls {[(c['scene'], round(c['rel_mse'], 3)) for c in ctrl]}")

    for meta in metas:
        i = meta["scene"]
        if not meta["mask_ok"] or i in done_ids:
            continue
        ts = time.time()
        meta, subj, R_, zs, cond, ref, redo, tg = setup(fz, i)
        rgb, bgm = arr(ref), bg_cells(subj)
        rf, redo13 = rfm_baseline(sd, smp, d, meta, subj, R_, ref, tg)
        res, imgs = {}, {}
        for T in TARGETS:
            res[f"none|{T}"] = score(rgb, tg[T], R_, ref)
            res[f"target|{T}"] = score(rgb, tg[T], R_, G.to_img(tg[T]))
            cands = [(m, im) for (wn, TT), (m, im) in rf.items() if TT == T]
            ok = [x for x in cands if crit(x[0], T)["reach"]]
            m, im = min(ok, key=lambda x: x[0]["int_mae"]) if ok else min(cands, key=lambda x: abs(x[0]["bg"] - T))
            res[f"rfm|{T}"] = dict(m, all={x[0]["window"]: x[0] for x in cands})
            imgs[f"rfm|{T}"] = im
            loss = Loss(rgb, tg[T], R_, T)
            for kind in ("global", "spatial"):
                (L, coef, it), hist = fz.fit(zs, cond, B, kind, bgm, loss, N_ITER, LR, LR_RES)
                im = fz.decode_img(fz.run(zs, cond, make_edit(B, coef, kind, bgm)))
                m = score(rgb, tg[T], R_, im)
                m.update(loss=L, best_it=it, coef_absmax=float(coef.abs().max()),
                         coef_rms=float(coef.pow(2).mean().sqrt()), hist=hist)
                m["iou"], m["shift"] = layout(seg, meta["noun"], meta["thr"], subj, im)
                res[f"{kind}|{T}"], imgs[f"{kind}|{T}"] = m, im
                im.save(out / "images" / f"{i}_{kind}_T{abs(T):g}.png")
                torch.save(coef.cpu(), out / "images" / f"{i}_{kind}_T{abs(T):g}.pt")
            imgs[f"rfm|{T}"].save(out / "images" / f"{i}_rfm_T{abs(T):g}.png")
            res[f"rfm|{T}"]["iou"], res[f"rfm|{T}"]["shift"] = layout(seg, meta["noun"], meta["thr"], subj,
                                                                       imgs[f"rfm|{T}"])
        rows.append(dict(scene=i, kind=meta["kind"], noun=meta["noun"], regen_max_diff=redo,
                         rfm_sampler_max_diff=redo13, res=res, seconds=time.time() - ts))
        log(f"scene {i}: " + "; ".join(f"{k} bg {v['bg']:+.2f} int {v['int_mae']:.2f} band {v['band_mae']:.2f}"
                                        for k, v in res.items() if not k.startswith(("none", "target"))))
        (out / "fit.json").write_text(json.dumps(dict(rows=rows), indent=1, default=float))
    report(rows, ctrl, time.time() - t_start)


# ---------------------------------------------------------------- report and frozen decision
def judge(r, T):
    """Per scene and target: criteria for global and spatial (README)."""
    res = r["res"]
    out = {}
    base = {k: res[f"{k}|{T}"] for k in ("rfm", "global")}
    for kind in ("rfm", "global", "spatial"):
        out[kind] = crit(res[f"{kind}|{T}"], T)
    feas = {k: m for k, m in base.items() if out[k]["reach"]}
    sp = res[f"spatial|{T}"]
    if feas:
        best = min(feas.values(), key=lambda m: m["int_mae"])
        out["spatial"]["advantage"] = sp["int_mae"] <= 0.5 * best["int_mae"]
    else:
        best = min(base.values(), key=lambda m: abs(m["bg"] - T))
        out["spatial"]["advantage"] = True
    out["spatial"]["boundary"] = sp["band_mae"] <= best["band_mae"]
    out["baseline_feasible"] = sorted(feas)
    return out


def report(rows, ctrl, secs):
    out = OUT / "fit"
    meths = ["none", "target", "rfm", "global", "spatial"]
    n = len(rows)
    L = [f"# Round 16: per-image fitted activation edits ({n} scenes, privileged feasibility test)", ""]
    summ, J = {}, {}
    for T in TARGETS:
        J[T] = {r["scene"]: judge(r, T) for r in rows}
        L += [f"## Target {T}", "", "| method | reached | interior MAE | subject MAE | band MAE vs target | "
              "bg MAE vs target / no-edit's | layout IoU |", "|---|---|---|---|---|---|---|"]
        for mt in meths:
            ms = [r["res"][f"{mt}|{T}"] for r in rows]
            reach = sum(abs(m["bg"] - T) <= TOL for m in ms)
            mean = lambda k: float(np.mean([m[k] for m in ms]))
            ratio = float(np.mean([m["bg_mae_T"] / max(m["bg_mae_refT"], 1e-6) for m in ms]))
            iou = float(np.nanmean([m.get("iou", 1.0) for m in ms])) if mt in ("rfm", "global", "spatial") else 1.0
            summ[f"{mt}|{T}"] = dict(reached=reach, int_mae=mean("int_mae"), subj_mae=mean("subj_mae"),
                                     band_mae=mean("band_mae"), bg_ratio=ratio, iou=iou)
            L.append(f"| {mt} | {reach}/{n} | {mean('int_mae'):.2f} | {mean('subj_mae'):.2f} | {mean('band_mae'):.2f} "
                     f"| {ratio:.2f} | {iou:.2f} |")
        L += ["", "Per scene (bg change / interior MAE; criteria: R reach, S subject <= 3, A background agreement, "
              "V spatial >= 50% lower subject error than the best feasible baseline, B boundary no worse):", "",
              "| scene | rfm (window) | global | spatial | spatial criteria | passes |", "|---|---|---|---|---|---|"]
        for r in rows:
            j, res = J[T][r["scene"]], r["res"]
            f = lambda k: f"{res[f'{k}|{T}']['bg']:+.2f} / {res[f'{k}|{T}']['int_mae']:.2f}"
            cs = j["spatial"]
            flags = "".join(c if cs[k] else "-" for c, k in zip("RSAVB", ("reach", "subject", "agree", "advantage",
                                                                         "boundary")))
            L.append(f"| {r['scene']} {r['kind']} | {f('rfm')} ({res[f'rfm|{T}']['window']}) | {f('global')} | "
                     f"{f('spatial')} | {flags} | {'yes' if all(cs.values()) else 'no'} |")
        L.append("")

    # frozen decision (README)
    sp_pass = {T: sum(all(J[T][s]["spatial"].values()) for s in J[T]) for T in TARGETS}
    gl_pass = {T: sum(all(J[T][s]["global"].values()) for s in J[T]) for T in TARGETS}
    sp_core = {T: sum(all(J[T][s]["spatial"][k] for k in ("reach", "subject", "agree")) for s in J[T]) for T in TARGETS}
    need = n - 1
    ctrl_ok = all(c["rel_mse"] <= 0.1 for c in ctrl)
    if all(sp_pass[T] >= need for T in TARGETS):
        dec = "GO: the spatial edit meets the joint requirements; next, test whether a shared controller learns them"
    elif all(gl_pass[T] >= need for T in TARGETS):
        dec = "SIMPLER: the fitted global edit already passes; drop the spatial-controller claim"
    elif all(sp_core[T] >= need for T in TARGETS):
        dec = "NOT GO: the spatial edit is feasible but does not beat the baselines by the frozen margin"
    else:
        dec = "NO-GO for this architecture"
    modes = {}
    for T in TARGETS:
        for s, j in J[T].items():
            c = j["spatial"]
            mode = ("pass" if all(c.values()) else "blur only by changing the subject" if c["reach"] and not c["subject"]
                    else "subject kept only by missing the blur" if c["subject"] and not c["reach"]
                    else "misses blur and changes subject" if not c["reach"] else "background not like the target"
                    if not c["agree"] else "no advantage / boundary")
            modes.setdefault(T, {}).setdefault(mode, []).append(s)
    L += ["## Decision", "", f"- Spatial edit passes all five criteria: " + ", ".join(
        f"{sp_pass[T]}/{n} at {T}" for T in TARGETS) + f" (GO needs >= {need}/{n} at both).",
          f"- Spatial edit passes reach + subject + agreement: " + ", ".join(f"{sp_core[T]}/{n} at {T}" for T in TARGETS),
          f"- Global edit passes reach + subject + agreement: " + ", ".join(f"{gl_pass[T]}/{n} at {T}" for T in TARGETS),
          f"- Failure modes of the spatial edit: {json.dumps(modes)}",
          f"- Optimization control (recover a known spatial edit, MSE <= 10% of zero-edit): "
          + ", ".join(f"scene {c['scene']} {c['rel_mse']:.3f}" for c in ctrl) + f" -> {'PASS' if ctrl_ok else 'FAIL'}",
          f"- **{dec}**" + ("" if ctrl_ok else " (inconclusive: the optimizer failed its control)"), "",
          f"Total time {secs:.0f} s."]
    checks = [
        ("regenerated unedited images match the saved ones (<= 1 level)", all(r["regen_max_diff"] <= 1 for r in rows),
         ", ".join(f"{r['regen_max_diff']:.0f}" for r in rows)),
        ("round 13 sampler gives the same unedited image", all(r["rfm_sampler_max_diff"] <= 1 for r in rows),
         ", ".join(f"{r['rfm_sampler_max_diff']:.0f}" for r in rows)),
        ("no-edit endpoint: bg change 0, subject change 0", all(abs(r["res"][f"none|{T}"]["bg"]) < 1e-9 and
                                                              r["res"][f"none|{T}"]["int_mae"] == 0 for r in rows
                                                              for T in TARGETS), ""),
        ("target endpoint: reaches the target, subject interior unchanged",
         all(abs(r["res"][f"target|{T}"]["bg"] - T) <= 0.02 and r["res"][f"target|{T}"]["int_mae"] <= 0.5
             for r in rows for T in TARGETS), ""),
        ("optimization control", ctrl_ok, ", ".join(f"{c['rel_mse']:.3f}" for c in ctrl)),
        ("all measures finite", all(np.isfinite(v[k]) for r in rows for v in r["res"].values()
                                    for k in ("bg", "int_mae", "band_mae")), ""),
    ]
    sheets(rows)
    (out / "fit_summary.json").write_text(json.dumps(dict(summary=summ, judge={str(T): J[T] for T in TARGETS},
                                                          decision=dec, control_ok=ctrl_ok), indent=1, default=str))
    P.write(out / "fit_checks.md", L, checks)


def sheets(rows):
    out = OUT / "fit"
    sc = OUT / "scenes"
    for T in TARGETS:
        R5 = []
        for r in rows[:5]:
            i = r["scene"]
            R5.append([Image.open(sc / f"{i}_target_T{abs(T):g}.png")] +
                      [Image.open(out / "images" / f"{i}_{k}_T{abs(T):g}.png") for k in ("rfm", "global", "spatial")])
        P.sheet(R5, out / f"examples_T{abs(T):g}.png", W=200)
    crops = []
    for r in rows:
        i = r["scene"]
        subj = M.mask(sc / f"{i}_subject.npy")
        x0, y0, x1, y1 = G.bbox(subj, pad=24)
        h = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, y0 = int(np.clip(cx - h // 2, 0, 512 - h)), int(np.clip(cy - h // 2, 0, 512 - h))
        box = (x0, y0, x0 + h, y0 + h)
        crops.append([Image.open(sc / f"{i}_target_T2.png").crop(box),
                      Image.open(out / "images" / f"{i}_spatial_T2.png").crop(box)])
    P.sheet([sum(crops[:4], []), sum(crops[4:], [])], out / "boundary_crops_T2.png", W=160)


if __name__ == "__main__":
    {"gen": gen, "fit": fit_all}[sys.argv[1]]()
