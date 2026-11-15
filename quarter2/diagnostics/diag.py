"""Q2 round 12: evaluation fixes, then idea 1's feasibility test (SD 1.5 depth of field). Diagnostics, not results.
Run on a GPU pod after rounds 10 and 11 (uses their outputs):

    python quarter2/diagnostics/diag.py evalfix      # fixed subject masks, matched-blur comparison, copy-back baseline
    python quarter2/diagnostics/diag.py vaecheck     # idea 3: is VAE loss even the size of the steering damage?
    python quarter2/diagnostics/diag.py geometry2    # idea 2: clean-geometry and soft-mask baselines added
    python quarter2/diagnostics/diag.py feasibility  # idea 1: do useful measured-response corrections exist?

Evaluation (every stage):
  subject mask  CLIPSeg (CIDAS/clipseg-rd64-refined) with the caption's subject noun, on the UNEDITED reference only;
                checked against the depth mask and on an overlay sheet. Regions: subject; subject interior (> 6 px
                inside); background (> 12 px outside); boundary band (the rest).
  background    change in log Laplacian variance of the background region (paired with the reference).
  subject       pixel RMS vs the reference in the subject and its interior; SSIM averaged over the subject; DINOv2
                cosine of the subject crop; whole-image DINOv2.
  boundary      mean absolute error in the band against an approximate rendered target: the reference blurred with
                depth (round 10's renderer) with the subject pasted back, at the blur level whose background change is
                closest to the method's. Approximate, not a physical render.
  achievement   |background change - target| <= 0.15.
Methods are compared at the same achieved background change, never at the same steering strength.
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
sys.path.insert(0, str(ROOT / "quarter2" / "direction_pilots"))
import pilots as PL  # noqa: E402  (round 11: ComboHook, render2; puts rounds 10 and 4 on the path)

M, P, D = PL.M, PL.P, PL.D
R10 = PL.SMOKE / "images"                   # round 10 smoke 1 images: none, rfm_{global,masked}_s{2,4}
OUT = ROOT / "outputs" / "q2_diagnostics"
SEG = "CIDAS/clipseg-rd64-refined"
SEG_THR = 0.4
SEED, S = 42, 4.0
CAPS = P.CAPS                               # round 10's 8 captions (first 6 = smoke 1's)
SUBJECTS = {0: "dog", 1: "woman", 3: "child", 4: "bird", 5: "deer", 6: "cat", 7: "man riding a bicycle"}
TOL = 0.15
T_LEVELS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
log, arr, gray = M.log, PL.arr, PL.gray


# ---------------------------------------------------------------- masks and measures
class Seg:
    def __init__(self, device):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        self.proc = CLIPSegProcessor.from_pretrained(SEG)
        self.model = CLIPSegForImageSegmentation.from_pretrained(SEG).to(device).eval()
        self.dev = device

    @torch.no_grad()
    def __call__(self, img, noun):
        from scipy.ndimage import binary_fill_holes, label
        x = self.proc(text=[f"a photo of a {noun}"], images=[img], padding="max_length", return_tensors="pt")
        lg = self.model(**x.to(self.dev)).logits
        lg = lg.reshape(-1, lg.shape[-2], lg.shape[-1])[:1, None].float()
        p = torch.nn.functional.interpolate(lg, size=img.size[::-1], mode="bilinear", align_corners=False)
        prob = torch.sigmoid(p)[0, 0].cpu().numpy()
        m = binary_fill_holes(prob > SEG_THR)
        lab, n = label(m)
        if n > 1:                                       # drop specks under 10% of the largest piece
            sizes = np.bincount(lab.ravel())[1:]
            m = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.1 * sizes.max()))
        return m, prob


def regions(subj):
    from scipy.ndimage import distance_transform_edt as edt
    din, dout = edt(subj), edt(~subj)
    interior = din > 6
    if interior.sum() < 200:                            # tiny subject: use all of it
        interior = subj.copy()
    bg = dout > 12
    return dict(subject=subj, interior=interior, background=bg, band=~(interior | bg))


def lapvar(a, m):
    from scipy.ndimage import laplace
    return float(np.log(laplace(gray(a))[m].var() + 1e-6))


def ssim_map(a, b):
    from scipy.ndimage import gaussian_filter as g
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = g(a, 1.5), g(b, 1.5)
    saa, sbb, sab = g(a * a, 1.5) - ma ** 2, g(b * b, 1.5) - mb ** 2, g(a * b, 1.5) - ma * mb
    return ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2))


def bbox(m, pad=8):
    ys, xs = np.nonzero(m)
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1]),
            min(ys.max() + pad + 1, m.shape[0]))


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8))


def blur(rgb, r):
    from scipy.ndimage import gaussian_filter
    if r <= 0:
        return rgb.copy()
    return np.stack([gaussian_filter(rgb[..., c], r) for c in range(3)], -1)


def alpha(subj, feather=3.0):
    """1 on the subject (edges included), fading to 0 over ~2 feather widths outside it."""
    from scipy.ndimage import gaussian_filter
    return np.clip(2 * gaussian_filter(subj.astype(float), feather), 0, 1)[..., None]


def copyback(ref, a, subj):
    al = alpha(subj)
    return al * ref + (1 - al) * a


def posthoc(ref, subj, r):
    al = alpha(subj)
    return al * ref + (1 - al) * blur(ref, r)


def overlay(img, m):
    a = arr(img)
    a[m] = 0.55 * a[m] + 0.45 * np.array([255, 0, 0])
    return to_img(a)


class Scorer:
    def __init__(self, dino):
        self.dino = dino

    def ref(self, img, subj, invd=None):
        rgb = arr(img)
        R = regions(subj)
        box = bbox(subj)
        C = dict(rgb=rgb, g=gray(rgb), R=R, box=box, e=self.dino(img), ec=self.dino(img.crop(box)),
                 lv={k: lapvar(rgb, R[k]) for k in ("background", "interior")}, targets=None)
        if invd is not None:                            # approximate rendered targets for the boundary band
            q = np.quantile(invd[subj], 0.1)
            C["targets"] = []
            for L in T_LEVELS:
                T = PL.render2(rgb, invd, L, q)
                T[subj] = rgb[subj]
                C["targets"].append((L, T, lapvar(T, R["background"]) - C["lv"]["background"]))
        return C

    def __call__(self, C, img):
        a = arr(img) if isinstance(img, Image.Image) else np.asarray(img, dtype=np.float64)
        im = to_img(a)
        a = arr(im)                                     # score what would be saved (uint8)
        R, d = C["R"], a - C["rgb"]
        rms = lambda m: float(np.sqrt((d[m] ** 2).mean()) / 255)
        s = ssim_map(C["g"], gray(a))
        out = dict(bg=lapvar(a, R["background"]) - C["lv"]["background"],
                   fg=lapvar(a, R["interior"]) - C["lv"]["interior"],
                   subj_rms=rms(R["subject"]), int_rms=rms(R["interior"]), band_rms=rms(R["band"]),
                   subj_ssim=float(s[R["subject"]].mean()), subj_dino=float(C["ec"] @ self.dino(im.crop(C["box"]))),
                   dino=float(C["e"] @ self.dino(im)))
        if C["targets"]:
            L, T, _ = min(C["targets"], key=lambda t: abs(t[2] - out["bg"]))
            out.update(bnd_err=float(np.abs(a - T)[R["band"]].mean() / 255), bnd_level=L)
        return out


KEYS = ["bg", "fg", "subj_rms", "int_rms", "subj_ssim", "subj_dino", "bnd_err", "dino"]


def fmt(m, keys=KEYS):
    return " | ".join("" if m.get(k) is None else f"{m[k]:+.3f}" if k in ("bg", "fg") else f"{m[k]:.4f}"
                      if k.endswith("rms") or k == "bnd_err" else f"{m[k]:.3f}" for k in keys)


HEAD = "bg | subject interior sharpness | subject RMS | interior RMS | subject SSIM | subject DINOv2 | boundary err | DINOv2"


# ---------------------------------------------------------------- evaluation fixes on round 10's images
def matched(grid, target):
    """grid: [(strength, measures)] with increasing strength. Measures interpolated where bg reaches target."""
    xs = [x for x, _ in grid]
    x = M.interp(xs, [m["bg"] for _, m in grid], target)
    if x is None:
        return None
    out = {k: float(np.interp(x, xs, [m[k] for _, m in grid])) for k in grid[0][1] if k != "bnd_level"}
    out["strength"] = float(x)
    return out


def evalfix():
    out = OUT / "evalfix"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meter, dino, seg = D.DofMeter("cuda"), M.Dino("cuda"), Seg("cuda")
    sc = Scorer(dino)
    idx = [i for i in range(6) if i in SUBJECTS]
    radii = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
    fams = {"global": [0, 2, 4], "masked": [0, 2, 4], "copyback_global": [0, 2, 4], "copyback_masked": [0, 2, 4],
            "posthoc_subject": radii, "uniform": radii}
    per = {f: [] for f in fams}                         # per scene: [(strength, measures)]
    mask_info, conv, ident, sheet_rows = [], [], None, []
    for i in idx:
        ref = Image.open(R10 / "none" / f"{i:02d}.png").convert("RGB")
        subj, prob = seg(ref, SUBJECTS[i])
        dfg = meter.fg_mask(ref)
        mask_info.append(dict(scene=i, noun=SUBJECTS[i], area=float(subj.mean()),
                              iou_depth=float((subj & dfg).sum() / max((subj | dfg).sum(), 1)),
                              contrast=float(prob[subj].mean() - prob[~subj].mean()) if subj.any() else 0.0))
        np.save(out / f"{i:02d}_subject.npy", np.packbits(subj))
        if not subj.any() or subj.all():
            log(f"scene {i}: degenerate subject mask, skipped")
            continue
        C = sc.ref(ref, subj, P.inv_depth(meter, ref))
        rgb = C["rgb"]
        if ident is None:
            ident = sc(C, ref)
        for f in ("global", "masked"):
            rows, rows_cb = [], []
            for s in (0, 2, 4):
                im = ref if s == 0 else Image.open(R10 / f"rfm_{f}_s{s}" / f"{i:02d}.png").convert("RGB")
                rows.append((s, sc(C, im)))
                rows_cb.append((s, sc(C, copyback(rgb, arr(im), subj))))
                if s == 4:                              # the mask convention question (round 9's caveat)
                    g0, g1 = gray(rgb), gray(arr(im))
                    pd = M.pair(g0, g1, rgb, arr(im), dfg)
                    conv.append(dict(scene=i, method=f, ratio_depth=pd["ratio"], ratio_subject=rows[-1][1]["fg"]
                                     - rows[-1][1]["bg"]))
            per[f].append(rows)
            per["copyback_" + f].append(rows_cb)
        per["posthoc_subject"].append([(r, sc(C, posthoc(rgb, subj, r))) for r in radii])
        per["uniform"].append([(r, sc(C, blur(rgb, r))) for r in radii])
        m4 = Image.open(R10 / "rfm_masked_s4" / f"{i:02d}.png").convert("RGB")
        sheet_rows.append([overlay(ref, subj), overlay(ref, dfg), m4, to_img(copyback(rgb, arr(m4), subj)),
                           to_img(posthoc(rgb, subj, 1.0))])
        log(f"scene {i} scored")
    n_sheet = P.sheet(sheet_rows, out / "evalfix_examples.png")

    # matched comparison, per scene, at each target
    res = {}
    for target in (-0.28, -0.5):
        uni = [matched(g, target) for g in per["uniform"]]
        for f in fams:
            ms = [matched(g, target) for g in per[f]]
            ok = [(m, u) for m, u in zip(ms, uni) if m is not None]
            mean = {k: float(np.mean([m[k] for m, _ in ok])) for k in ok[0][0]} if ok else {}
            ex = [m["fg"] - u["fg"] for m, u in ok if u is not None]
            if ok:
                mean["fg_minus_uniform"] = float(np.mean(ex)) if ex else None
            res[f"{f}|{target}"] = dict(reached=len(ok), n=len(ms), mean=mean)
    (out / "evalfix.json").write_text(json.dumps(dict(masks=mask_info, convention=conv, matched=res,
                                                      per_scene={f: [[(s, m) for s, m in g] for g in v]
                                                                 for f, v in per.items()}), indent=1))
    L = [f"# Round 12 evaluation fixes on round 10's smoke images ({len(per['uniform'])} scenes with a subject; "
         "diagnostic)",
         "Subject masks from CLIPSeg on the unedited reference. Every method compared at the same achieved background "
         "change (interpolated along its own strength grid, per scene); scenes where a method never reaches the target "
         "count against its achievement. 'vs uniform' = subject interior sharpness change minus that of a uniform blur "
         "reaching the same background change on the same scene (a matched control, not a constant offset). Strength: "
         "steering s for global / masked (round 10 depth mask), blur sigma (px) for post-hoc / uniform. "
         "`evalfix_examples.png` columns: CLIPSeg subject mask, depth mask (nearest 40%), masked s 4, its copy-back, "
         "post-hoc blur sigma 1.", ""]
    for target in (-0.28, -0.5):
        L += [f"## Target background change {target}", "",
              f"| method | reached | strength | {HEAD} | vs uniform |", "|---|---|---|" + "---|" * 9]
        for f in fams:
            r = res[f"{f}|{target}"]
            m = r["mean"]
            L.append(f"| {f} | {r['reached']}/{r['n']} | {m.get('strength', float('nan')):.2f} | {fmt(m)} | "
                     + ("" if m.get("fg_minus_uniform") is None else f"{m['fg_minus_uniform']:+.3f}") + " |")
        L.append("")
    L += ["## Mask convention at s 4 (subject sharpness change minus background change)", "",
          "| scene | method | depth mask (nearest 40%) | CLIPSeg subject mask |", "|---|---|---|---|"]
    L += [f"| {c['scene']} | {c['method']} | {c['ratio_depth']:+.3f} | {c['ratio_subject']:+.3f} |" for c in conv]
    L += ["", "## Subject masks", "", "| scene | noun | area | IoU with depth mask | CLIPSeg contrast |",
          "|---|---|---|---|---|"]
    L += [f"| {m['scene']} | {m['noun']} | {m['area']:.3f} | {m['iou_depth']:.2f} | {m['contrast']:.2f} |"
          for m in mask_info]
    L.append(f"\nTotal time {time.time() - t0:.0f} s.")
    ph = [dict(g)[3] for g in per["posthoc_subject"]]
    un = [dict(g)[2] for g in per["uniform"]]
    cb = [dict(g)[4] for g in per["copyback_masked"]]
    checks = [
        ("identity: reference vs itself gives 0 changes, SSIM 1",
         ident is not None and all(abs(ident[k]) < 1e-9 for k in ("bg", "fg", "subj_rms", "int_rms"))
         and abs(ident["subj_ssim"] - 1) < 1e-6, ident and f"SSIM {ident['subj_ssim']:.6f}"),
        ("every subject mask non-degenerate (area 0.5%-80%)", all(0.005 <= m["area"] <= 0.8 for m in mask_info),
         ", ".join(f"{m['area']:.3f}" for m in mask_info)),
        ("post-hoc subject blur sigma 3: background blurs, subject interior unchanged",
         all(p["bg"] < 0 and p["int_rms"] < 1e-3 for p in ph), ", ".join(f"{p['bg']:+.2f}" for p in ph)),
        ("uniform blur sigma 2 blurs the subject interior (< -0.2)", all(u["fg"] < -0.2 for u in un),
         ", ".join(f"{u['fg']:+.2f}" for u in un)),
        ("copy-back keeps the subject interior exactly (RMS < 1e-3)", all(c["int_rms"] < 1e-3 for c in cb),
         ", ".join(f"{c['int_rms']:.1e}" for c in cb)),
        ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}"),
    ]
    if not P.write(out / "evalfix_checks.md", L, checks):
        sys.exit("evalfix: some checks failed")


# ---------------------------------------------------------------- idea 3: VAE loss vs steering damage
def vaecheck():
    from diffusers import AutoencoderKL
    out = OUT / "vaecheck"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vae = AutoencoderKL.from_pretrained(D.MODEL, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    seg = Seg("cuda")

    @torch.no_grad()
    def rt(a):
        x = torch.from_numpy(np.clip(a, 0, 255) / 127.5 - 1).permute(2, 0, 1)[None].float().cuda()
        y = vae.decode(vae.encode(x).latent_dist.mean).sample.clamp(-1, 1)
        return arr(to_img((y[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5))

    psnr = lambda a, b: float(10 * np.log10(255 ** 2 / max(((a - b) ** 2).mean(), 1e-10)))
    rms = lambda d, m: float(np.sqrt((d[m] ** 2).mean()) / 255)
    rows = []
    for i in [i for i in range(6) if i in SUBJECTS]:
        ref = Image.open(R10 / "none" / f"{i:02d}.png").convert("RGB")
        subj, _ = seg(ref, SUBJECTS[i])
        if not subj.any():
            continue
        a = arr(ref)
        r = dict(scene=i, rt_psnr=psnr(rt(a), a), rt_subj=rms(rt(a) - a, subj))
        for f in ("masked", "global"):
            r[f"steer_{f}"] = rms(arr(Image.open(R10 / f"rfm_{f}_s4" / f"{i:02d}.png").convert("RGB")) - a, subj)
        rows.append(r)
    photos = []
    try:
        imgs, labels, names = D.dof_photos(n_per_class=6)
        photos = [dict(label=names[int(l)], rt_psnr=psnr(rt(arr(im)), arr(im))) for im, l in zip(imgs, labels)]
    except Exception as e:                              # photo set is optional context
        log(f"real photos unavailable: {e}")
    (out / "vaecheck.json").write_text(json.dumps(dict(generated=rows, photos=photos), indent=1))
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    L = [f"# Round 12 idea 3 check: VAE round-trip loss vs steering damage ({len(rows)} generated scenes; diagnostic)",
         "Our images are generated as D(z): they never passed through the encoder. Residual reinjection can only "
         "repair what an encode-decode cycle loses, so its ceiling is the round-trip error. Steering damage = subject "
         "RMS between the steered image (round 10, s 4) and the unsteered one.", "",
         "| scene | round-trip PSNR (dB) | round-trip subject RMS | masked-steering subject RMS | global-steering subject RMS |",
         "|---|---|---|---|---|"]
    L += [f"| {r['scene']} | {r['rt_psnr']:.1f} | {r['rt_subj']:.4f} | {r['steer_masked']:.4f} | {r['steer_global']:.4f} |"
          for r in rows]
    if rows:
        L += ["", f"Steering damage is {mean('steer_masked') / mean('rt_subj'):.1f}x (masked) and "
              f"{mean('steer_global') / mean('rt_subj'):.1f}x (global) the round-trip loss in the subject."]
    if photos:
        L += ["", f"Real photos (round 4's depth-of-field set, {len(photos)}): round-trip PSNR "
              f"{np.mean([p['rt_psnr'] for p in photos]):.1f} dB vs {mean('rt_psnr'):.1f} dB for generated images."]
    L.append(f"\nTotal time {time.time() - t0:.0f} s.")
    checks = [("round trip finite and sane (15-60 dB)", all(15 < r["rt_psnr"] < 60 for r in rows),
               ", ".join(f"{r['rt_psnr']:.1f}" for r in rows)),
              ("real photos measured", bool(photos) or None, f"{len(photos)}")]
    if not P.write(out / "vaecheck_checks.md", L, checks):
        sys.exit("vaecheck: some checks failed")


# ---------------------------------------------------------------- idea 2: clean geometry and soft masks added
def geometry2():
    from scipy.ndimage import gaussian_filter
    out = OUT / "geometry2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meter = D.DofMeter("cuda")
    imgs = PL.load_images()
    rng = np.random.default_rng(0)

    def field(shape):
        f = gaussian_filter(rng.normal(0, 1, shape), 4)
        return f / f.std()
    errs = ["clean", "noise", "smoothed", "threshold"]
    meths = ["plain", "soft", "feather_L", "ensemble8"]
    res = {(e, L, m): [] for e in errs for L in PL.LEVELS for m in meths}
    for im in imgs:
        rgb, d = arr(im), P.inv_depth(meter, im)
        q = np.quantile(d, 0.6)
        fg_true, sd_ = d >= q, d.std()                  # evaluation uses the true geometry only
        for e in errs:
            de = {"clean": d, "noise": d + 0.15 * sd_ * field(d.shape), "smoothed": gaussian_filter(d, 6),
                  "threshold": d}[e]
            qlev = 0.7 if e == "threshold" else 0.6     # the method's own (possibly wrong) cut
            qe = np.quantile(de, qlev)
            for L in PL.LEVELS:
                T = PL.render2(rgb, d, L, q)
                plain = PL.render2(rgb, de, L, qe)
                soft = (1 / (1 + np.exp(-(de - qe) / (0.05 * sd_))))[..., None]
                fm = np.clip(gaussian_filter((de >= qe).astype(float), L), 0, 1)[..., None]
                ens = []
                for _ in range(8):
                    dj = de + 0.05 * sd_ * field(d.shape)
                    # round 11 drew this cut around the TRUE level (0.6), which hid the threshold error; now it is
                    # drawn around the method's own level
                    ens.append(PL.render2(rgb, dj, L, np.quantile(dj, rng.uniform(qlev - 0.05, qlev + 0.05))))
                outs = dict(plain=plain, soft=soft * rgb + (1 - soft) * plain, feather_L=fm * rgb + (1 - fm) * plain,
                            ensemble8=np.mean(ens, 0))
                reg = PL.regions(fg_true, L)
                tb = PL.lapvar(T, reg["bg_interior"])
                for name, o in outs.items():
                    r = {k: float(np.abs(o - T)[m].mean() / 255) for k, m in reg.items()}
                    r["bg_blur_gap"] = PL.lapvar(o, reg["bg_interior"]) - tb
                    res[(e, L, name)].append(r)
    agg = {f"{e}|{L:g}|{m}": {k: float(np.mean([r[k] for r in v])) for k in v[0]} for (e, L, m), v in res.items()}
    (out / "geometry2.json").write_text(json.dumps(agg, indent=1))
    L_ = [f"# Round 12 idea 2 check: protection under depth errors, with a clean-geometry baseline ({len(imgs)} "
          "images; diagnostic)",
          "The method sees corrupted depth; the truth and the evaluation regions use the uncorrupted depth. 'clean' = "
          "the method sees the true depth (the floor each method can reach). soft = sigmoid alpha from depth (width "
          "0.05 depth SD). Truth is our own renderer, so subject-copy methods are exact inside the subject by "
          "construction; read the boundary and blur-gap columns.", "",
          "| depth error | sigma | method | subject interior | boundary | background interior | bg blur gap |",
          "|---|---|---|---|---|---|---|"]
    for e in errs:
        for L in PL.LEVELS:
            for m in meths:
                a = agg[f"{e}|{L:g}|{m}"]
                L_.append(f"| {e} | {L:g} | {m} | {a['fg_interior']:.4f} | {a['boundary']:.4f} | "
                          f"{a['bg_interior']:.4f} | {a['bg_blur_gap']:+.3f} |")
    L_.append(f"\nTotal time {time.time() - t0:.0f} s.")
    checks = [("all finite", all(np.isfinite(v) for a in agg.values() for v in a.values()), ""),
              ("clean geometry: plain render matches the truth exactly",
               all(agg[f"clean|{L:g}|plain"]["boundary"] < 1e-9 for L in PL.LEVELS), "")]
    if not P.write(out / "geometry2_checks.md", L_, checks):
        sys.exit("geometry2: some checks failed")


# ---------------------------------------------------------------- idea 1: do useful corrections exist?
def feasibility(cfg):
    from scipy.ndimage import binary_dilation
    from scipy.optimize import lsq_linear
    out = OUT / "feasibility"
    (out / "images").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    D.Hook = PL.ComboHook
    sd, meter, dino, seg = D.SD("cuda"), D.DofMeter("cuda"), M.Dino("cuda"), Seg("cuda")
    sc = Scorer(dino)
    dirs = torch.load(M.DIRS)
    log(f"direction keys: {sorted(dirs)}")
    unit = lambda v: (v.float() / v.float().norm()).view(1, 1280, 16, 16).cuda()
    dv = dict(rfm=unit(dirs["rfm_eig1"]), eig2=unit(dirs["rfm_eig2"]),
              paired=unit(torch.load(PL.SMOKE / "paired" / "paired_direction.pt")["paired_dir"]))
    names = ["rfm_bg", "rfm_fg", "eig2_fg", "eig2_bg", "paired_bg"]
    K, T_, dl, box = len(names), cfg.target, cfg.delta, cfg.box
    rows, sheet_rows = [], []

    for i in cfg.scenes:
        cap, noun, ts = CAPS[i], SUBJECTS[i], time.time()
        n_gen = 0
        g16 = lambda m: torch.from_numpy(M.bg_grid(m)).view(1, 1, 16, 16).cuda()

        def gen(comps, tag):
            nonlocal n_gen
            if not any(c != 0 for _, _, c in comps):
                comps = [(dv["rfm"], torch.zeros(1, 1, 16, 16, device="cuda"), 0.0)]
            sd.hook.comps = [(d, m, float(c)) for d, m, c in comps]
            im = sd.generate([cap], [SEED], "rfm", S, dv["rfm"], bs=1)[0]
            sd.hook.comps = None
            n_gen += 1
            im.save(out / "images" / f"{i:02d}_{tag}.png")
            a = arr(im)
            return dict(im=im, r=((a - C["rgb"])[subj] / 255).ravel(), m=sc(C, im))

        ref = sd.generate([cap], [SEED], "none", bs=1)[0]
        ref.save(out / "images" / f"{i:02d}_ref.png")
        subj, _ = seg(ref, noun)
        if subj.mean() < 0.005 or subj.mean() > 0.8:
            log(f"scene {i}: subject mask area {subj.mean():.3f}, skipped")
            rows.append(dict(scene=i, skipped=f"mask area {subj.mean():.3f}"))
            continue
        C = sc.ref(ref, subj, P.inv_depth(meter, ref))
        bgw = g16(subj)                                  # fraction of non-subject pixels per block
        fgw = 1 - bgw
        bgw_dil = g16(binary_dilation(subj, iterations=16))
        cm = dict(rfm_bg=(dv["rfm"], bgw), rfm_fg=(dv["rfm"], fgw), eig2_fg=(dv["eig2"], fgw),
                  eig2_bg=(dv["eig2"], bgw), paired_bg=(dv["paired"], bgw))
        comps_of = lambda u: [(*cm[n], u[k]) for k, n in enumerate(names)]

        def tune(make, tag):
            """Secant search on the strength c so the background change reaches the target (<= 3 generations)."""
            tried, c = [], 1.0
            for j in range(3):
                g = gen(make(c), f"{tag}_try{j}")
                tried.append((c, g))
                if abs(g["m"]["bg"] - T_) <= 0.05:
                    break
                pts = sorted([(0.0, 0.0)] + [(cc, gg["m"]["bg"]) for cc, gg in tried])
                cn = M.interp([p[0] for p in pts], [p[1] for p in pts], T_)
                if cn is None:
                    cc, gg = min(tried, key=lambda t: abs(t[1]["m"]["bg"] - T_))
                    cn = cc * T_ / gg["m"]["bg"] if gg["m"]["bg"] < -1e-3 else 2 * cc
                c = float(np.clip(cn, 0.1, 4.0))
            c, g = min(tried, key=lambda t: abs(t[1]["m"]["bg"] - T_))
            return c, g

        c_m, g_m = tune(lambda c: comps_of(c * np.eye(K)[0]), "masked")
        c_d, g_d = tune(lambda c: [(dv["rfm"], bgw_dil, c)], "masked_dilated")
        c_g, g_g = tune(lambda c: [(dv["rfm"], torch.ones(1, 1, 16, 16, device="cuda"), c)], "global")

        # measured responses around the tuned masked edit: central differences, + and - perturbations
        u0, r0, b0 = c_m * np.eye(K)[0], g_m["r"], g_m["m"]["bg"]
        J, gvec, nl_r, nl_b = np.zeros((len(r0), K)), np.zeros(K), [], []
        for k in range(K):
            gp = gen(comps_of(u0 + dl * np.eye(K)[k]), f"probe_{names[k]}_plus")
            gm = gen(comps_of(u0 - dl * np.eye(K)[k]), f"probe_{names[k]}_minus")
            J[:, k] = (gp["r"] - gm["r"]) / (2 * dl)
            gvec[k] = (gp["m"]["bg"] - gm["m"]["bg"]) / (2 * dl)
            nl_r.append(float(np.linalg.norm(gp["r"] + gm["r"] - 2 * r0) / max(np.linalg.norm(gp["r"] - gm["r"]), 1e-9)))
            nl_b.append(float(abs(gp["m"]["bg"] + gm["m"]["bg"] - 2 * b0) / max(abs(gp["m"]["bg"] - gm["m"]["bg"]), 1e-9)))
        s = 1 / np.sqrt(len(r0))
        lam = 1e-2 * float(np.mean((J * s) ** 2) * len(r0))

        def solve(r, b, lo, hi):
            A = np.vstack([J * s, np.sqrt(lam) * np.eye(K), 10.0 * gvec[None]])
            y = np.concatenate([-r * s, np.zeros(K), [10.0 * (T_ - b)]])
            x = lsq_linear(A, y, bounds=(lo, hi)).x
            return x, float(np.linalg.norm(r + J @ x) * s), float(b + gvec @ x)

        lo, hi = -box * np.ones(K), box * np.ones(K)
        lo[0] = max(-box, -c_m)                          # the background edit's total stays >= 0
        dx1, pr1, pb1 = solve(r0, b0, lo, hi)
        g1 = gen(comps_of(u0 + dx1), "corrected1")
        dx2, pr2, pb2 = solve(g1["r"], g1["m"]["bg"], lo - dx1, hi - dx1)
        g2 = gen(comps_of(u0 + dx1 + dx2), "corrected2")
        # control: a random correction with the same size and the same predicted background change
        rng = np.random.default_rng(i)
        v = rng.normal(size=K)
        v -= gvec * (gvec @ v) / max(gvec @ gvec, 1e-12)
        v = v / max(np.linalg.norm(v), 1e-12) * np.linalg.norm(dx1)
        dxr = np.clip(v + gvec * (T_ - b0) / max(gvec @ gvec, 1e-12), lo, hi)
        gr = gen(comps_of(u0 + dxr), "random_control")
        # noise floor: how much does a tiny edit (10% of the tuned one) already move the subject?
        gf = gen(comps_of(0.1 * u0), "floor")
        r_ph = min(((r, sc(C, posthoc(C["rgb"], subj, r))) for r in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]),
                   key=lambda t: abs(t[1]["bg"] - T_))
        meth = dict(masked=g_m["m"], masked_dilated=g_d["m"], global_=g_g["m"],
                    copyback=sc(C, copyback(C["rgb"], arr(g_m["im"]), subj)), posthoc=r_ph[1],
                    corrected1=g1["m"], corrected2=g2["m"], random_control=gr["m"], floor_10pct=gf["m"])
        for k_, m in meth.items():
            m["achieved"] = bool(abs(m["bg"] - T_) <= TOL)
        rows.append(dict(scene=i, caption=cap, noun=noun, c_masked=c_m, c_dilated=c_d, c_global=c_g,
                         posthoc_sigma=r_ph[0], u0=u0.tolist(), grad_bg=gvec.tolist(), nonlin_subject=nl_r,
                         nonlin_bg=nl_b, J_col_rms=(np.sqrt((J ** 2).mean(0))).tolist(),
                         dx1=dx1.tolist(), dx2=dx2.tolist(), dx_random=dxr.tolist(),
                         pred1=dict(subj_rms=pr1, bg=pb1), pred2=dict(subj_rms=pr2, bg=pb2),
                         start=dict(subj_rms=float(np.linalg.norm(r0) * s), bg=b0), methods=meth,
                         n_gen=n_gen, seconds=time.time() - ts))
        if len(sheet_rows) < 5:
            best = g2 if g2["m"]["subj_rms"] < g1["m"]["subj_rms"] else g1
            sheet_rows.append([overlay(ref, subj), g_m["im"], g_d["im"], best["im"],
                               to_img(copyback(C["rgb"], arr(g_m["im"]), subj))])
        log(f"scene {i}: masked {g_m['m']['subj_rms']:.4f} (bg {b0:+.2f}) -> corrected1 {g1['m']['subj_rms']:.4f} "
            f"(pred {pr1:.4f}; bg {g1['m']['bg']:+.2f}, pred {pb1:+.2f}) -> corrected2 {g2['m']['subj_rms']:.4f}; "
            f"random {gr['m']['subj_rms']:.4f}; {n_gen} generations, {time.time() - ts:.0f} s")
        (out / "feasibility.json").write_text(json.dumps(rows, indent=1))

    n_sheet = P.sheet(sheet_rows, out / "feasibility_examples.png") if sheet_rows else 0
    done = [r for r in rows if "methods" in r]
    order = ["masked", "masked_dilated", "global_", "posthoc", "copyback", "corrected1", "corrected2", "random_control",
             "floor_10pct"]
    L = [f"# Round 12 idea 1 feasibility: do measured-response corrections beat tuned masking? ({len(done)} scenes x "
         f"seed {SEED}, s {S:g}, target background change {T_}; diagnostic)",
         f"Basis: {', '.join(names)} (direction x 16x16 block weight; _bg = non-subject fraction, _fg = the rest). "
         f"Operating point = masked RFM tuned to the target. Responses of the signed subject pixels (vs the unedited "
         f"reference) and of the background change, by central differences (+/- {dl}). Correction: minimise "
         f"|r0 + J dx|^2 + small ridge, background change held at the target, each |dx_k| <= {box}; generated, then "
         "one more step with the same J. Random control: a correction of the same size and predicted background change "
         "in a random direction. Nonlinearity = |r(+) + r(-) - 2 r0| / |r(+) - r(-)| (0 = linear). "
         "`feasibility_examples.png` columns: subject mask, masked tuned, masked dilated tuned, best correction, "
         "copy-back.", "",
         f"| scene | method | achieved | {HEAD} |", "|---|---|---|" + "---|" * 8]
    for r in done:
        for k in order:
            m = r["methods"][k]
            L.append(f"| {r['scene']} | {k.rstrip('_')} | {'yes' if m['achieved'] else 'no'} | {fmt(m)} |")
    L += ["", "| scene | start RMS | predicted 1 | realised 1 | predicted bg 1 | realised bg 1 | predicted 2 | "
          "realised 2 | nonlinearity (subject, per component) | nonlinearity (bg) | dx1 |", "|---|" + "---|" * 10]
    for r in done:
        mt = r["methods"]
        L.append(f"| {r['scene']} | {r['start']['subj_rms']:.4f} | {r['pred1']['subj_rms']:.4f} | "
                 f"{mt['corrected1']['subj_rms']:.4f} | {r['pred1']['bg']:+.2f} | {mt['corrected1']['bg']:+.2f} | "
                 f"{r['pred2']['subj_rms']:.4f} | {mt['corrected2']['subj_rms']:.4f} | "
                 f"{', '.join(f'{x:.2f}' for x in r['nonlin_subject'])} | "
                 f"{', '.join(f'{x:.2f}' for x in r['nonlin_bg'])} | {', '.join(f'{x:+.2f}' for x in r['dx1'])} |")
    # decision, following the plan's table
    beat, reliable, beat_rand, beat_cb_bnd = 0, 0, 0, 0
    for r in done:
        mt = r["methods"]
        cands = [mt[k] for k in ("corrected1", "corrected2") if mt[k]["achieved"]]
        tuned = [mt[k]["subj_rms"] for k in ("masked", "masked_dilated") if mt[k]["achieved"]]
        if cands and tuned:
            best = min(cands, key=lambda m: m["subj_rms"])
            beat += best["subj_rms"] < 0.9 * min(tuned)
            beat_rand += best["subj_rms"] < mt["random_control"]["subj_rms"]
            beat_cb_bnd += best.get("bnd_err", 1) < mt["copyback"].get("bnd_err", 0)
        p, q = r["pred1"], mt["corrected1"]
        reliable += abs(q["subj_rms"] - p["subj_rms"]) <= 0.25 * q["subj_rms"] and abs(q["bg"] - p["bg"]) <= TOL
    n = max(len(done), 1)
    if beat <= n / 2:
        dec = "Measured-response corrections do not beat tuned masking: reconsider the basis or mechanism before training."
    elif reliable <= n / 2:
        dec = "Corrections work, but predictions are unreliable: investigate locality, step size, or nonlinear response."
    else:
        dec = "Corrections work reliably on individual scenes: next, test whether a reusable predictor transfers to unseen scenes."
    L += ["", "## Decision (diagnostic, per-scene probing; not evidence of a reusable method)", "",
          f"- Correction beats the best tuned masking by >= 10% subject RMS, at the target: {beat}/{len(done)} scenes.",
          f"- Linear prediction reliable (realised RMS within 25%, background within {TOL}): {reliable}/{len(done)}.",
          f"- Correction beats the random control: {beat_rand}/{len(done)}.",
          f"- Correction has lower boundary error than copy-back: {beat_cb_bnd}/{len(done)}.",
          "- Noise floor: subject RMS of a 10% edit "
          f"{np.mean([r['methods']['floor_10pct']['subj_rms'] for r in done]):.4f} vs the tuned masked edit "
          f"{np.mean([r['methods']['masked']['subj_rms'] for r in done]):.4f} (close values mean most subject change "
          "comes from any perturbation at all, not from the size of the edit)." if done else "",
          f"- Cost: {np.mean([r['n_gen'] for r in done]):.0f} generations per scene "
          f"({np.mean([r['seconds'] for r in done]):.0f} s); a tuned mask needs <= 3." if done else "",
          f"- **{dec}**", f"\nTotal time {time.time() - t_start:.0f} s."]
    fin = lambda m: all(np.isfinite(v) for k, v in m.items() if isinstance(v, float))
    checks = [("all scenes ran", len(done) == len(cfg.scenes), f"{len(done)}/{len(cfg.scenes)}"),
              ("all measures finite", all(fin(m) for r in done for m in r["methods"].values()), ""),
              ("tuned masking reaches the target", sum(r["methods"]["masked"]["achieved"] for r in done) >= n / 2,
               ", ".join(f"{r['methods']['masked']['bg']:+.2f}" for r in done)),
              ("responses measurable: the background responds to rfm_bg (gradient < 0)",
               all(r["grad_bg"][0] < 0 for r in done), ", ".join(f"{r['grad_bg'][0]:+.2f}" for r in done)),
              ("contact sheet <= 25 images", n_sheet <= 25, f"{n_sheet}")]
    if not P.write(out / "feasibility_checks.md", L, checks):
        sys.exit("feasibility: some checks failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["evalfix", "vaecheck", "geometry2", "feasibility"])
    ap.add_argument("--scenes", type=int, nargs="+", default=[0, 1, 3, 5, 6, 7])
    ap.add_argument("--target", type=float, default=-0.5)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--box", type=float, default=1.0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    if a.stage == "feasibility":
        feasibility(a)
    else:
        {"evalfix": evalfix, "vaecheck": vaecheck, "geometry2": geometry2}[a.stage]()


if __name__ == "__main__":
    main()
