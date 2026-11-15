"""Q2 round 11: smoke pilots for three candidate directions (SD 1.5 depth of field). Pipeline checks plus a first
look at the signal; not results. Run on a GPU pod, after round 10's smoke tests (uses their outputs):

    python quarter2/direction_pilots/pilots.py response    # idea 1: measure how edits spread; pick a combination
    python quarter2/direction_pilots/pilots.py geometry    # idea 2: protection masks under depth errors
    python quarter2/direction_pilots/pilots.py residual    # idea 3: carry VAE-lost detail through the optical edit

response  4 scenes (round 10 captions with a clear subject), seed 42, text CFG 7.5, third branch at s 4. Edit
          components at round 4's layer, each h <- h + 0.7 |h| c_k m_k d_k: RFM x background mask, RFM x
          foreground mask, paired direction (round 10 smoke 2) x background, 2nd RFM eigenvector x background.
          Measured per scene: each component at c = 1, RFM-background at c = 2 and RFM-background + paired at c = 1
          (linearity checks). Linear model: background sharpness change a.c; signed pixel change inside the subject
          sum_k c_k Delta_k. Pick c minimising the predicted subject change subject to a.c = target, generate it, and
          compare with masked RFM alone at the c the same model predicts reaches the target.
geometry  round 10 smoke 2's 8 images; ground truth = depth-dependent blur rendered with the estimated depth. Depth
          errors: correlated noise, smoothed depth, a shifted foreground threshold. Methods given the wrong depth:
          plain render, fixed 4 px feathered subject protection, feather width growing with the blur, and an
          average over 8 perturbed depths. Errors against the truth in subject interior, background interior and
          the boundary band, plus background blur reached.
residual  the same images: ground truth T = rendered blur; simulated perfect latent edit = VAE round trip of T.
          r = I - VAE(I). Compare VAE(T) alone, + r, + r carried through the same blur, a subject copy (true mask),
          and subject copy + carried residual.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof_masked"))
import sd_dof_masked as M  # noqa: E402  (round 10: pair, bg_grid, Dino; puts round 4's sd_dof on the path)
import smoke_paired as P   # noqa: E402  (round 10 smoke 2: render, inv_depth, sheet, write)

D = M.D
SMOKE = ROOT / "outputs" / "q2_sd_dof_masked_smoke"
OUT = ROOT / "outputs" / "q2_direction_pilots"
SEED, S = 42, 4.0
SCENES = [0, 1, 3, 5]          # round 10 captions: dog, umbrella, two children, deer
TARGET = -0.5                  # background log-sharpness change asked of every method in `response`
LEVELS = [2.0, 4.0, 8.0]
log = M.log


def arr(im):
    return np.asarray(im).astype(np.float64)


def gray(a):
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8)).convert("L")).astype(np.float64)


# ---------------------------------------------------------------- idea 1: response of the final image to edits
class ComboHook(D.Hook):
    """Round 4's hook with a sum of spatially masked components: h <- h + w |h| sum_k c_k m_k d_k."""

    def __init__(self, module):
        super().__init__(module)
        self.comps = None

    def fn(self, mod, inp, out):
        if not self.comps or self.d is None or self.rows is None:
            return super().fn(mod, inp, out)
        self.last = out
        lo, hi = self.rows
        h = out[lo:hi].float()
        norm = h.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        e = sum(c * m * d for d, m, c in self.comps)
        new = out.clone()
        new[lo:hi] = (h + self.w * norm * e).to(out.dtype)
        return new


def response():
    out = OUT / "response"
    (out / "images").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    D.Hook = ComboHook
    sd, meter, dino = D.SD("cuda"), D.DofMeter("cuda"), M.Dino("cuda")
    dirs = torch.load(M.DIRS)
    unit = lambda v: (v.float() / v.float().norm()).view(1, 1280, 16, 16).cuda()
    d_rfm, d_eig2 = unit(dirs["rfm_eig1"]), unit(dirs["rfm_eig2"])
    d_pair = unit(torch.load(SMOKE / "paired" / "paired_direction.pt")["paired_dir"])
    names = ["rfm_bg", "rfm_fg", "paired_bg", "eig2_bg"]
    caps = [c for _, c in M.CAPTIONS]
    rows, sheet_rows = [], []

    for i in SCENES:
        cap = caps[i]
        none = sd.generate([cap], [SEED], "none", bs=1)[0]
        fg = meter.fg_mask(none)
        bg = torch.from_numpy(M.bg_grid(fg)).view(1, 1, 16, 16).cuda()
        comps = [(d_rfm, bg), (d_rfm, 1 - bg), (d_pair, bg), (d_eig2, bg)]
        e0, rgb0, g0 = dino(none), arr(none), gray(arr(none))
        none.save(out / "images" / f"{i:02d}_none.png")

        def run(tag, c):
            sd.hook.comps = [(d, m, float(ck)) for (d, m), ck in zip(comps, c) if ck != 0]
            im = sd.generate([cap], [SEED], "rfm", S, d_rfm, bs=1)[0]
            sd.hook.comps = None
            im.save(out / "images" / f"{i:02d}_{tag}.png")
            a = arr(im)
            p = M.pair(g0, gray(a), rgb0, a, fg)
            dl = (a - rgb0)[fg]                                   # signed pixel change inside the subject
            p.update(dino=float(e0 @ dino(im)), fg_rms=float(np.sqrt((dl ** 2).mean()) / 255), c=[float(x) for x in c])
            return im, a - rgb0, p

        single = [run(n, np.eye(4)[k]) for k, n in enumerate(names)]
        a_vec = np.array([p["bg"] for _, _, p in single])
        Dl = np.stack([d[fg].ravel() for _, d, _ in single])     # (4, n_fg * 3)
        G = Dl @ Dl.T / Dl.shape[1]
        # linearity checks
        _, d2, p2 = run("rfm_bg_c2", [2, 0, 0, 0])
        _, d13, p13 = run("rfm_bg+paired_bg", [1, 0, 1, 0])
        rel = lambda x, y: float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-9))
        lin = dict(scale_bg=(p2["bg"], 2 * a_vec[0]), scale_pix_relerr=rel(d2[fg], 2 * single[0][1][fg]),
                   sum_bg=(p13["bg"], a_vec[0] + a_vec[2]),
                   sum_pix_relerr=rel(d13[fg], single[0][1][fg] + single[2][1][fg]))
        # baseline: masked RFM alone at the c the linear model says reaches the target
        cb = TARGET / a_vec[0] if a_vec[0] < 0 else None
        base = None
        if cb is not None:
            cb = float(np.clip(cb, 0, 4))
            im_b, _, base = run("masked_matched", [cb, 0, 0, 0])
            base["pred_fg_rms"] = float(abs(cb) * np.sqrt(G[0, 0]) / 255)
        # optimised combination: min c'(G + mu I)c  s.t.  a.c = TARGET
        A = G + 1e-3 * np.trace(G) / 4 * np.eye(4)
        Ai_a = np.linalg.solve(A, a_vec)
        c = TARGET * Ai_a / (a_vec @ Ai_a)
        clipped = bool(np.any(np.abs(c) > 4))
        c = np.clip(c, -4, 4)
        im_o, _, opt = run("optimised", c)
        opt.update(pred_fg_rms=float(np.sqrt(max(c @ G @ c, 0)) / 255), pred_bg=float(a_vec @ c), clipped=clipped)
        rows.append(dict(scene=i, caption=cap, a=a_vec.tolist(), G_diag=np.diag(G).tolist(), single=[p for *_, p in
                         single], lin=lin, base=base, opt=opt))
        sheet_rows.append([none, im_b if base else none, im_o, Image.open(out / "images" / f"{i:02d}_rfm_bg.png")])
        log(f"scene {i}: base {base and round(base['fg_rms'], 4)} opt {round(opt['fg_rms'], 4)}")

    n_sheet = P.sheet(sheet_rows, out / "response_examples.png")
    (out / "response.json").write_text(json.dumps(rows, indent=1))
    ok_rows = [r for r in rows if r["base"]]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    checks = [
        ("every run produced finite measures",
         all(np.isfinite(v) for r in rows for p in r["single"] + [r["opt"]] + ([r["base"]] if r["base"] else [])
             for k, v in p.items() if isinstance(v, float)), f"{len(rows)} scenes"),
        ("masked RFM blurs the background at c = 1 (a < 0) in every scene", len(ok_rows) == len(rows),
         ", ".join(f"{r['a'][0]:+.3f}" for r in rows)),
        ("linearity in scale: bg at c = 2 vs 2 x bg at c = 1", None,
         "; ".join(f"{r['lin']['scale_bg'][0]:+.2f} vs {r['lin']['scale_bg'][1]:+.2f}" for r in rows)),
        ("linearity in scale: subject pixel change, relative error", None,
         ", ".join(f"{r['lin']['scale_pix_relerr']:.2f}" for r in rows)),
        ("additivity: bg of RFM-bg + paired-bg vs sum of the two", None,
         "; ".join(f"{r['lin']['sum_bg'][0]:+.2f} vs {r['lin']['sum_bg'][1]:+.2f}" for r in rows)),
        ("additivity: subject pixel change, relative error", None,
         ", ".join(f"{r['lin']['sum_pix_relerr']:.2f}" for r in rows)),
        ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}"),
    ]
    L = [f"# Pilot 1: response of the final image to edit components ({len(rows)} scenes x seed {SEED}, s {S:g}; "
         f"target background change {TARGET})",
         "Linear model from single components; a relative error near 0 means the response is linear (the model is "
         "trustworthy), near 1 or above means it is not. `response_examples.png` columns: none, masked RFM matched, "
         "optimised combination, masked RFM at c = 1.", "",
         "| scene | method | c (rfm_bg, rfm_fg, paired_bg, eig2_bg) | bg (target " + f"{TARGET}) | predicted bg | "
         "fg sharpness | subject pixel RMS | predicted | pixel FG / BG | DINOv2 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for tag, p in (("masked RFM", r["base"]), ("optimised", r["opt"])):
            if p is None:
                L.append(f"| {r['scene']} | {tag} | (masked RFM does not blur here) | | | | | | | |")
                continue
            pb = p.get("pred_bg", TARGET)
            L.append(f"| {r['scene']} | {tag} | {', '.join(f'{x:+.2f}' for x in p['c'])} | {p['bg']:+.3f} | {pb:+.3f} "
                     f"| {p['fg']:+.3f} | {p['fg_rms']:.4f} | {p['pred_fg_rms']:.4f} | {p['pix_fg']:.3f} / "
                     f"{p['pix_bg']:.3f} | {p['dino']:.3f} |")
    if ok_rows:
        L += ["", f"Means over the {len(ok_rows)} scenes with a baseline: background masked {mean([r['base']['bg'] for r in ok_rows]):+.3f}"
              f" vs optimised {mean([r['opt']['bg'] for r in ok_rows]):+.3f}; subject pixel RMS masked "
              f"{mean([r['base']['fg_rms'] for r in ok_rows]):.4f} vs optimised {mean([r['opt']['fg_rms'] for r in ok_rows]):.4f}; "
              f"DINOv2 masked {mean([r['base']['dino'] for r in ok_rows]):.3f} vs optimised {mean([r['opt']['dino'] for r in ok_rows]):.3f}."]
    L.append(f"\nTotal time {time.time() - t_start:.0f} s.")
    if not P.write(out / "response_checks.md", L, checks):
        sys.exit("pilot 1: some checks failed")


# ---------------------------------------------------------------- shared: renderer with explicit threshold, float
def render2(rgb, invd, level, q, n=9):
    """Round 10's depth blur on float arrays without clipping; foreground = invd >= q stays exact."""
    from scipy.ndimage import gaussian_filter
    sig = level * np.clip((q - invd) / max(q - invd.min(), 1e-6), 0, 1)
    sigmas = np.linspace(0, level, n)
    stack = np.stack([rgb] + [np.stack([gaussian_filter(rgb[..., c], s) for c in range(rgb.shape[2])], -1)
                              for s in sigmas[1:]])
    pos = sig / level * (n - 1)
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
    fr = (pos - i0)[..., None]
    yy, xx = np.indices(sig.shape)
    return (1 - fr) * stack[i0, yy, xx] + fr * stack[i0 + 1, yy, xx]


def regions(fg, level):
    from scipy.ndimage import distance_transform_edt
    w = 2 * level
    fg_in = distance_transform_edt(fg) > w
    bg_in = distance_transform_edt(~fg) > w
    return dict(fg_interior=fg_in, bg_interior=bg_in, boundary=~(fg_in | bg_in))


def lapvar(a, m):
    from scipy.ndimage import laplace
    return float(np.log(laplace(gray(a))[m].var() + 1e-6))


def load_images():
    caps = json.loads((SMOKE / "paired" / "captions.json").read_text())
    return [Image.open(SMOKE / "paired" / "images" / f"{i:02d}_orig.png").convert("RGB") for i in range(len(caps))]


# ---------------------------------------------------------------- idea 2: protection under depth errors
def geometry():
    from scipy.ndimage import gaussian_filter
    out = OUT / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    meter = D.DofMeter("cuda")
    imgs = load_images()
    rng = np.random.default_rng(0)

    def field(shape):                                        # spatially correlated noise, unit std
        f = gaussian_filter(rng.normal(0, 1, shape), 4)
        return f / f.std()
    errs = ["noise", "smoothed", "threshold"]
    meths = ["plain", "feather4", "feather_L", "ensemble8"]
    res = {(e, L, m): [] for e in errs for L in LEVELS for m in meths}
    sheet_rows = []
    for k, im in enumerate(imgs):
        rgb, d = arr(im), P.inv_depth(meter, im)
        q = np.quantile(d, 0.6)
        fg_true = d >= q
        sd_ = d.std()
        for e in errs:
            if e == "noise":
                de, qe = d + 0.15 * sd_ * field(d.shape), None
            elif e == "smoothed":
                de, qe = gaussian_filter(d, 6), None
            else:
                de, qe = d, np.quantile(d, 0.7)                  # foreground shrinks: subject edges get blurred
            qe = np.quantile(de, 0.6) if qe is None else qe
            fe = de >= qe
            for L in LEVELS:
                T = render2(rgb, d, L, q)
                plain = render2(rgb, de, L, qe)
                outs = {"plain": plain}
                for name, w in (("feather4", 4.0), ("feather_L", L)):
                    m = np.clip(gaussian_filter(fe.astype(float), w), 0, 1)[..., None]
                    outs[name] = m * rgb + (1 - m) * plain
                ens = []
                for j in range(8):
                    dj = de + 0.05 * sd_ * field(d.shape)
                    ens.append(render2(rgb, dj, L, np.quantile(dj, rng.uniform(0.55, 0.65))))
                outs["ensemble8"] = np.mean(ens, 0)
                reg = regions(fg_true, L)
                tb = lapvar(T, reg["bg_interior"])
                for name, o in outs.items():
                    r = {k2: float(np.abs(o - T)[m].mean() / 255) for k2, m in reg.items()}
                    r["bg_blur_gap"] = lapvar(o, reg["bg_interior"]) - tb   # > 0: background not blurred enough
                    res[(e, L, name)].append(r)
                if k < 2 and e == "threshold" and L == max(LEVELS):
                    sheet_rows.append([Image.fromarray(np.clip(x, 0, 255).round().astype(np.uint8))
                                       for x in [T] + [outs[m] for m in meths]])
    n_sheet = P.sheet(sheet_rows, out / "geometry_examples.png")
    agg = {f"{e}|{L:g}|{m}": {k2: float(np.mean([r[k2] for r in v])) for k2 in v[0]} for (e, L, m), v in res.items()}
    (out / "geometry.json").write_text(json.dumps(agg, indent=1))
    L_ = [f"# Pilot 2: subject protection under depth errors ({len(imgs)} images; ground truth = our renderer with the "
          "estimated depth; pipeline check)",
          "Mean absolute error against the truth (fraction of 255) by region; boundary = within 2 x sigma of the true "
          "subject edge. bg blur gap: log Laplacian variance of the background interior minus the truth's (> 0: not "
          "blurred enough). `geometry_examples.png` columns: truth, plain, feather 4, feather L, ensemble (threshold "
          "error, sigma 8).", "",
          "Depth errors: noise = correlated noise of 0.15 depth SD; smoothed = depth blurred (sigma 6 px); threshold = "
          "subject cut at the nearest 30% instead of 40%. Ensemble: 8 renders with extra 0.05 SD noise and the cut "
          "drawn from the nearest 35-45%.", "",
          "| depth error | sigma | method | subject interior | boundary | background interior | bg blur gap |",
          "|---|---|---|---|---|---|---|"]
    for e in errs:
        for L in LEVELS:
            for m in meths:
                a = agg[f"{e}|{L:g}|{m}"]
                L_.append(f"| {e} | {L:g} | {m} | {a['fg_interior']:.4f} | {a['boundary']:.4f} | "
                          f"{a['bg_interior']:.4f} | {a['bg_blur_gap']:+.3f} |")
    L_.append(f"\nTotal time {time.time() - t_start:.0f} s.")
    checks = [("all errors finite", all(np.isfinite(v) for a in agg.values() for v in a.values()), f"{len(agg)} cells"),
              ("no depth error gives zero error for the plain render (errors are real)",
               all(agg[f"{e}|8|plain"]["boundary"] > 0 for e in errs), ""),
              ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}")]
    if not P.write(out / "geometry_checks.md", L_, checks):
        sys.exit("pilot 2: some checks failed")


# ---------------------------------------------------------------- idea 3: carry VAE-lost detail through the edit
def residual():
    from scipy.ndimage import gaussian_filter
    out = OUT / "residual"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    sd, meter = D.SD("cuda"), D.DofMeter("cuda")
    vae = sd.pipe.vae

    @torch.no_grad()
    def rt(a):
        x = torch.from_numpy(np.clip(a, 0, 255) / 127.5 - 1).permute(2, 0, 1)[None].to("cuda", sd.dt)
        z = vae.encode(x).latent_dist.mean
        y = vae.decode(z).sample.float().clamp(-1, 1)
        return ((y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.float64)

    def psnr(a, b, m):
        mse = ((np.clip(a, 0, 255) - b)[m] ** 2).mean()
        return float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))

    imgs = load_images()
    meths = ["vae", "direct", "carried", "subject_copy", "subject_copy+carried"]
    res = {(L, m): [] for L in LEVELS for m in meths}
    rt_psnr, sheet_rows = [], []
    for k, im in enumerate(imgs):
        rgb, d = arr(im), P.inv_depth(meter, im)
        q = np.quantile(d, 0.6)
        fg = d >= q
        r = rgb - rt(rgb)
        rt_psnr.append(psnr(rgb - r, rgb, np.ones_like(fg)))
        m = np.clip(gaussian_filter(fg.astype(float), 2), 0, 1)[..., None]
        for L in LEVELS:
            T = render2(rgb, d, L, q)
            E = rt(T)
            rc = render2(r, d, L, q)
            outs = {"vae": E, "direct": E + r, "carried": E + rc, "subject_copy": m * rgb + (1 - m) * E,
                    "subject_copy+carried": m * rgb + (1 - m) * (E + rc)}
            reg = regions(fg, L)
            reg["all"] = np.ones_like(fg)
            for name, o in outs.items():
                res[(L, name)].append({k2: psnr(o, T, mm) for k2, mm in reg.items()})
            if k < 2 and L == max(LEVELS):
                sheet_rows.append([Image.fromarray(np.clip(x, 0, 255).round().astype(np.uint8))
                                   for x in [T] + [outs[n] for n in meths]])
    n_sheet = P.sheet(sheet_rows, out / "residual_examples.png")
    agg = {f"{L:g}|{m}": {k2: float(np.mean([r[k2] for r in v])) for k2 in v[0]} for (L, m), v in res.items()}
    (out / "residual.json").write_text(json.dumps(dict(agg=agg, vae_roundtrip_psnr=rt_psnr), indent=1))
    L_ = [f"# Pilot 3: carrying VAE-lost detail through the optical edit ({len(imgs)} images; pipeline check)",
          "PSNR (dB) against the rendered truth. The truth keeps the subject exactly, so `subject_copy` (with the true "
          "mask) is perfect inside the subject by construction; the background and boundary columns are the test. "
          f"VAE round trip of the originals: {np.mean(rt_psnr):.1f} dB. `residual_examples.png` columns: truth, "
          + ", ".join(meths) + " (sigma 8).", "",
          "| sigma | method | all | subject interior | boundary | background interior |", "|---|---|---|---|---|---|"]
    for L in LEVELS:
        for m_ in meths:
            a = agg[f"{L:g}|{m_}"]
            L_.append(f"| {L:g} | {m_} | {a['all']:.2f} | {a['fg_interior']:.2f} | {a['boundary']:.2f} | "
                      f"{a['bg_interior']:.2f} |")
    L_.append(f"\nTotal time {time.time() - t_start:.0f} s.")
    checks = [("all PSNRs finite", all(np.isfinite(v) for a in agg.values() for v in a.values()), ""),
              ("VAE round trip loses detail (PSNR < 40 dB) but is sane (> 20 dB)",
               20 < np.mean(rt_psnr) < 40, f"{np.mean(rt_psnr):.1f} dB"),
              ("direct residual restores the unedited subject (subject interior PSNR up vs vae alone)",
               agg[f"{max(LEVELS):g}|direct"]["fg_interior"] > agg[f"{max(LEVELS):g}|vae"]["fg_interior"], ""),
              ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}")]
    if not P.write(out / "residual_checks.md", L_, checks):
        sys.exit("pilot 3: some checks failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["response", "geometry", "residual"])
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    {"response": response, "geometry": geometry, "residual": residual}[a.stage]()


if __name__ == "__main__":
    main()
