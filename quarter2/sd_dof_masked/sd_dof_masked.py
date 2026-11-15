"""Q2 round 10: does restricting the RFM edit to the background make the depth-of-field effect selective? (SD 1.5)

Smoke test only so far (no design frozen, no results). It checks that every part runs end to end and that the
measures behave as intended on cases with a known answer:

    python quarter2/sd_dof_masked/sd_dof_masked.py smoke

Conditions (6 hand-written captions x seed 42):
  none                    unsteered (text CFG 7.5)
  rfm_global_s{2,4}       round 4/9's edit h <- h + 0.7 |h| d/|d| at down_blocks[2].resnets[-1], everywhere
  rfm_masked_s{2,4}       the same edit multiplied by a background mask: the unsteered image's depth background
                          (farthest 60% of pixels), area-pooled to the layer's 16 x 16 grid (soft, in [0, 1])
  posthoc_r{1,3}          the unsteered image blurred outside its depth foreground (round 9's oracle; no generation)
  uniform_r2              the unsteered image blurred everywhere (a measure check: not selective)

Every measure is paired with the unsteered image of the same caption and seed, inside the unsteered image's depth
mask; the paper's per-image ("own") mask is reported next to it. Uses round 4's sampler and meters
(quarter2/sd_dof/sd_dof.py, unchanged) and round 9's paper direction (outputs/q2_sd_dof_direction/directions.pt,
key rfm_eig1; identical to round 4's RFM direction, cosine 1.0000).
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
sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof"))
import sd_dof as D  # noqa: E402  (round 4: SD sampler, DofMeter, hboot)

DIRS = ROOT / "outputs" / "q2_sd_dof_direction" / "directions.pt"
DINO = "facebook/dinov2-base"
GRID = 16                    # spatial size of the steered layer (1280 x 16 x 16)
FEATHER = 8.0                # post-hoc blur: feathering of the foreground mask (Gaussian sigma, pixels)
TARGET_BG = -0.28            # matching check: round 9's paper direction at s 4 on test (log background change)
SEED = 42
CAPTIONS = [                 # 2 ordinary, 4 hard (no subject, two depths, small subject, sharp-background request)
    ("ordinary", "A dog sitting in front of a wooden fence in a garden"),
    ("ordinary", "A woman holding a red umbrella on a city street"),
    ("no_subject", "A wide mountain landscape with a calm lake"),
    ("two_depths", "Two children in a park, one close to the camera and one far away"),
    ("small_subject", "A small bird standing far away on a long empty beach"),
    ("sharp_bg", "A deer standing in a forest, the whole forest in sharp focus"),
]


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


class MaskedHook(D.Hook):
    """Round 4's hook, plus an optional spatial mask m (n, 1, 16, 16) on the edit. Records how much the layer output
    changed where m == 0 (must be exactly 0) and where m > 0 (must be > 0)."""

    def __init__(self, module):
        super().__init__(module)
        self.m, self.leak, self.hit = None, 0.0, 0.0

    def fn(self, mod, inp, out):
        if self.m is None or self.d is None or self.rows is None:
            return super().fn(mod, inp, out)
        self.last = out
        lo, hi = self.rows
        h = out[lo:hi].float()
        norm = h.flatten(1).norm(dim=1).view(-1, 1, 1, 1)      # same scale as the global edit
        new = out.clone()
        new[lo:hi] = (h + self.w * norm * self.d * self.m).to(out.dtype)
        dlt = (new[lo:hi].float() - out[lo:hi].float()).abs()
        off = (self.m == 0).expand_as(dlt)
        self.leak = max(self.leak, float(dlt[off].max()) if off.any() else 0.0)
        self.hit = max(self.hit, float(dlt[~off].max()) if (~off).any() else 0.0)
        return new


class Dino:
    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(DINO)
        self.model = AutoModel.from_pretrained(DINO).to(device).eval()
        self.dev = device

    @torch.no_grad()
    def __call__(self, img):
        x = self.proc(images=img, return_tensors="pt").to(self.dev)
        v = self.model(**x).last_hidden_state[0, 0].float()
        return (v / v.norm()).cpu().numpy()


def mask(path):
    return np.unpackbits(np.load(path))[:512 * 512].reshape(512, 512).astype(bool)


def bg_grid(fg):
    """Fraction of background pixels in each 32 x 32 block: the edit's weight at each of the 16 x 16 positions."""
    k = fg.shape[0] // GRID
    return (~fg).reshape(GRID, k, GRID, k).mean((1, 3)).astype(np.float32)


def blurred(rgb, fg, radius, selective):
    from scipy.ndimage import gaussian_filter
    b = np.stack([gaussian_filter(rgb[..., c], radius) for c in range(3)], -1)
    if not selective:
        return Image.fromarray(np.clip(b, 0, 255).round().astype(np.uint8))
    m = np.clip(gaussian_filter(fg.astype(float), FEATHER), 0, 1)[..., None]
    return Image.fromarray(np.clip(m * rgb + (1 - m) * b, 0, 255).round().astype(np.uint8))


def pair(g0, g1, rgb0, rgb1, fg):
    """Paired changes in log Laplacian variance inside the unsteered image's mask, and mean pixel change."""
    from scipy.ndimage import laplace
    l0, l1 = laplace(g0), laplace(g1)
    lv = lambda l, m: float(np.log(l[m].var() + 1e-6))
    bg, fgc = lv(l1, ~fg) - lv(l0, ~fg), lv(l1, fg) - lv(l0, fg)
    dlt = np.abs(rgb1 - rgb0).mean(2) / 255
    return dict(bg=bg, fg=fgc, ratio=fgc - bg, pix_fg=float(dlt[fg].mean()), pix_bg=float(dlt[~fg].mean()))


def interp(xs, ys, target):
    """Strength at which the mean background change reaches the target (linear between grid points), or None."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return None


def smoke(cfg):
    t_start = time.time()
    D.Hook = MaskedHook                       # SD() builds its hook from D.Hook
    sd, meter, dino = D.SD("cuda"), D.DofMeter("cuda"), Dino("cuda")
    direction = torch.load(DIRS)["rfm_eig1"]
    out, img = cfg.out, cfg.out / "images"
    caps = [c for _, c in CAPTIONS]
    perf, rec = {}, {}

    def measure(cond, i, im):
        d = img / cond
        d.mkdir(parents=True, exist_ok=True)
        im.save(d / f"{i:02d}.png")
        m = meter(im, caps[i])
        np.save(d / f"{i:02d}_fg.npy", np.packbits(m.pop("fg_mask")))
        np.save(d / f"{i:02d}_dino.npy", dino(im))
        rec[(cond, i)] = m

    def generate(cond, mode, s, masked):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, c in enumerate(caps):
            if masked:
                fg = mask(img / "none" / f"{i:02d}_fg.npy")
                sd.hook.m = torch.from_numpy(bg_grid(fg)).view(1, 1, GRID, GRID).cuda()
            im = sd.generate([c], [SEED], mode, s, direction if mode == "rfm" else None, bs=1)[0]
            sd.hook.m = None
            measure(cond, i, im)
        torch.cuda.synchronize()
        perf[cond] = dict(s_per_image=(time.time() - t0) / len(caps),
                          peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        log(f"{cond}: {perf[cond]['s_per_image']:.1f} s per image, peak {perf[cond]['peak_gb']:.2f} GB")

    generate("none", "none", 0.0, False)
    for s in cfg.s_grid:
        generate(f"rfm_global_s{s:g}", "rfm", s, False)
    sd.hook.leak, sd.hook.hit = 0.0, 0.0
    for s in cfg.s_grid:
        generate(f"rfm_masked_s{s:g}", "rfm", s, True)
    leak, hit = sd.hook.leak, sd.hook.hit
    for r in cfg.radii:
        for i in range(len(caps)):
            rgb = np.asarray(Image.open(img / "none" / f"{i:02d}.png")).astype(float)
            measure(f"posthoc_r{r:g}", i, blurred(rgb, mask(img / "none" / f"{i:02d}_fg.npy"), r, True))
    for i in range(len(caps)):
        rgb = np.asarray(Image.open(img / "none" / f"{i:02d}.png")).astype(float)
        measure("uniform_r2", i, blurred(rgb, None, 2.0, False))

    # paired measures against the unsteered image (none vs none included as the identity check)
    conds = ["none"] + [f"rfm_global_s{s:g}" for s in cfg.s_grid] + [f"rfm_masked_s{s:g}" for s in cfg.s_grid] \
        + [f"posthoc_r{r:g}" for r in cfg.radii] + ["uniform_r2"]
    res = {}
    for cond in conds:
        rows = []
        for i in range(len(caps)):
            p0, p1 = img / "none" / f"{i:02d}", img / cond / f"{i:02d}"
            rgb0 = np.asarray(Image.open(f"{p0}.png")).astype(float)
            rgb1 = np.asarray(Image.open(f"{p1}.png")).astype(float)
            g0 = np.asarray(Image.open(f"{p0}.png").convert("L")).astype(float)
            g1 = np.asarray(Image.open(f"{p1}.png").convert("L")).astype(float)
            p = pair(g0, g1, rgb0, rgb1, mask(f"{p0}_fg.npy"))
            r0, r1 = rec[("none", i)], rec[(cond, i)]
            p.update(ratio_own=float(np.log(r1["ratio"]) - np.log(r0["ratio"])),
                     dino=float(np.load(f"{p0}_dino.npy") @ np.load(f"{p1}_dino.npy")),
                     clip=r1["clip_caption"] - r0["clip_caption"], kind=CAPTIONS[i][0])
            rows.append(p)
        res[cond] = rows
    (out / "smoke_pairs.json").write_text(json.dumps(res, indent=1))
    mean = {c: {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "kind"} for c, rows in res.items()}

    # matched-blur check: strength at which each method reaches the target background change
    match = {
        "rfm_global": interp([0.0] + cfg.s_grid, [0.0] + [mean[f"rfm_global_s{s:g}"]["bg"] for s in cfg.s_grid],
                             TARGET_BG),
        "rfm_masked": interp([0.0] + cfg.s_grid, [0.0] + [mean[f"rfm_masked_s{s:g}"]["bg"] for s in cfg.s_grid],
                             TARGET_BG),
        "posthoc": interp([0.0] + cfg.radii, [0.0] + [mean[f"posthoc_r{r:g}"]["bg"] for r in cfg.radii], TARGET_BG),
    }

    # contact sheet: rows captions, columns none / global / masked at the largest s / post-hoc at the largest radius
    smax, rmax = max(cfg.s_grid), max(cfg.radii)
    cols = ["none", f"rfm_global_s{smax:g}", f"rfm_masked_s{smax:g}", f"posthoc_r{rmax:g}"]
    W = 160
    sheet = Image.new("RGB", (W * len(cols), W * len(caps)), "white")
    for r in range(len(caps)):
        for c, cond in enumerate(cols):
            sheet.paste(Image.open(img / cond / f"{r:02d}.png").resize((W, W), Image.LANCZOS), (c * W, r * W))
    sheet.save(out / "smoke_examples.png")
    n_sheet = len(cols) * len(caps)

    # checks
    vals = [v for rows in res.values() for r in rows for k, v in r.items() if k != "kind"]
    gen = [c for c in conds if c.startswith("rfm_")]
    ph, un = mean[f"posthoc_r{rmax:g}"], mean["uniform_r2"]
    checks = [
        ("identity: unsteered vs itself, every change 0",
         all(abs(v) < 1e-9 for r in res["none"] for k, v in r.items() if k not in ("kind", "dino")),
         f"max |change| {max(abs(v) for r in res['none'] for k, v in r.items() if k not in ('kind', 'dino')):.2e}"),
        ("masked hook: layer output unchanged where mask = 0", leak == 0.0, f"max |change| {leak:.2e}"),
        ("masked hook: layer output changed where mask > 0", hit > 0.0, f"max |change| {hit:.3f}"),
        ("post-hoc blur: background sharper drop than foreground (|fg| < 0.25 |bg|, bg < 0)",
         ph["bg"] < 0 and abs(ph["fg"]) < 0.25 * abs(ph["bg"]), f"bg {ph['bg']:+.3f}, fg {ph['fg']:+.3f}"),
        ("measure separates selective from uniform blur (post-hoc ratio > uniform ratio + 0.3)",
         ph["ratio"] > un["ratio"] + 0.3, f"post-hoc {ph['ratio']:+.3f}, uniform {un['ratio']:+.3f}"),
        ("fixed-mask and own-mask ratios finite for every generated condition",
         all(np.isfinite(r["ratio"]) and np.isfinite(r["ratio_own"]) for c in gen for r in res[c]), ""),
        ("matching returns a strength or says out of range for each method", True,
         ", ".join(f"{k}: {'out of range' if v is None else f'{v:.2f}'}" for k, v in match.items())),
        ("all metrics finite", all(np.isfinite(v) for v in vals), f"{len(vals)} values"),
        ("runtime and peak memory recorded", len(perf) == 1 + 2 * len(cfg.s_grid), ""),
        ("contact sheet <= 24 images", n_sheet <= 24, f"{n_sheet} images"),
    ]

    L = [f"# Round 10 smoke test ({len(caps)} captions x seed {SEED}; pipeline check, not a result)", "",
         "| check | pass | detail |", "|---|---|---|"]
    L += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n, ok, d in checks]
    L += ["", "Means over the captions, paired with the unsteered image (log changes; fixed mask unless noted):", "",
          "| condition | background | foreground | FG/BG ratio | ratio, own mask | pixel change FG / BG | DINOv2 | "
          "s per image | peak GB |", "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        m, pf = mean[c], perf.get(c, {})
        L.append(f"| {c} | {m['bg']:+.3f} | {m['fg']:+.3f} | {m['ratio']:+.3f} | {m['ratio_own']:+.3f} | "
                 f"{m['pix_fg']:.3f} / {m['pix_bg']:.3f} | {m['dino']:.3f} | "
                 f"{pf.get('s_per_image', float('nan')):.1f} | {pf.get('peak_gb', float('nan')):.2f} |")
    L += ["", f"Matched-blur strengths for target background change {TARGET_BG}: "
          + ", ".join(f"{k} {'out of range' if v is None else f'{v:.2f}'}" for k, v in match.items()),
          "", "Per caption (ratio, fixed mask):", "",
          "| caption kind | " + " | ".join(gen) + " |", "|---|" + "---|" * len(gen)]
    for i, (kind, _) in enumerate(CAPTIONS):
        L.append(f"| {kind} | " + " | ".join(f"{res[c][i]['ratio']:+.3f}" for c in gen) + " |")
    L.append(f"\nTotal time {time.time() - t_start:.0f} s.")
    (out / "smoke_checks.md").write_text("\n".join(L) + "\n")
    (out / "smoke.json").write_text(json.dumps(dict(mean=mean, perf=perf, match=match, leak=leak, hit=hit,
                                                    checks=[(n, bool(ok), d) for n, ok, d in checks]), indent=1))
    print("\n".join(L), flush=True)
    if not all(ok for _, ok, _ in checks):
        sys.exit("smoke test: some checks failed (see smoke_checks.md)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["smoke"])
    cfg = ap.parse_args()
    cfg.out = ROOT / "outputs" / "q2_sd_dof_masked_smoke"
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.s_grid, cfg.radii = [2.0, 4.0], [1.0, 3.0]
    torch.backends.cuda.matmul.allow_tf32 = True
    smoke(cfg)


if __name__ == "__main__":
    main()
