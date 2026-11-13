"""Is the measured blur more than reframing? Zoom-only control on the saved sd_dof.py images (no generation).

For each test pair (unsteered, steered; same caption and seed):
  1. Estimate scale and shift from the unsteered to the steered image (template matching over scales on
     low-passed grayscale, so blur barely affects it).
  2. Zoom-only control: the unsteered image warped by that scale and shift (bicubic) onto the steered image's grid.
     It has the steered framing but no new softening; upsampling only removes detail, so this control
     overstates the sharpness that zoom alone loses.
  3. Comparable pair: after alignment, the low-passed images correlate (NCC >= threshold) over an overlap of at
     least half the frame.
  4. Laplacian variance (the paper's measure) in the same regions at the same resolution: overlap x the
     unsteered image's depth foreground (warped), and overlap x its background.
     steered - unsteered = zoom-only (control - unsteered) + beyond zoom (steered - control).

Usage (repo root, on DSMLP):  python quarter2/sd_dof/zoom_control.py [--smoke]
Reads outputs/q2_sd_dof/{records.jsonl,images/}; writes outputs/q2_sd_dof/zoom/.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import laplace

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "q2_sd_dof"
CONDS = ["rfm", "meandiff"]
SEEDS = [42, 123, 256, 789]
NCC_MIN, OVERLAP_MIN, MIN_PIX = 0.7, 0.5, 2000
NCC_GRID = [0.5, 0.6, 0.7, 0.8]
SNAP = 0.015                 # |log scale| below this counts as no zoom (one step of the alignment grid is ~0.4%)
MIN_COMPARABLE = 25          # of 100 pairs per condition; fewer -> inconclusive


def gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float64)


def fg_mask(path):
    return np.unpackbits(np.load(path))[:512 * 512].reshape(512, 512).astype(bool)


def lowpass(a, size):
    a = cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(a, (0, 0), size / 128)


def match(none, st, size, scales):
    """Best (score, scale, mx, my, off) placing the steered centre (3/4 of the frame) inside the scaled unsteered."""
    n, s_img = lowpass(none, size), lowpass(st, size)
    off = size // 8
    tmpl = s_img[off:size - off, off:size - off]
    best = (-2, None, 0, 0, off)
    for sc in scales:
        m = int(round(size * sc))
        if m < tmpl.shape[0]:
            continue
        r = cv2.matchTemplate(cv2.resize(n, (m, m), interpolation=cv2.INTER_LINEAR), tmpl, cv2.TM_CCOEFF_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, m / size, loc[0], loc[1], off)
    return best


def align(none, st):
    """Affine map unsteered (512) -> steered (512): p_st = s * p_none + t."""
    _, sc, *_ = match(none, st, 128, np.geomspace(0.8, 2.2, 45))
    _, sc, mx, my, off = match(none, st, 256, sc * np.linspace(0.975, 1.025, 11))
    k = 512 / 256
    M = np.array([[sc, 0, -k * (mx - off)], [0, sc, -k * (my - off)]], np.float64)
    if abs(np.log(sc)) < SNAP:          # within alignment noise: no zoom, whole-pixel shift, so no resampling blur
        M[0, 0] = M[1, 1] = 1.0
        M[:, 2] = np.round(M[:, 2])
    return M


def lapvar(lap, mask):
    return float(lap[mask].var()) if mask.sum() >= MIN_PIX else float("nan")


def analyse_pair(cond, pi, x):
    d = SRC / "images"
    none, st = gray(d / "none" / f"{pi:02d}_{x}.jpg"), gray(d / cond / f"{pi:02d}_{x}.jpg")
    fg0 = fg_mask(d / "none" / f"{pi:02d}_{x}_fg.npy")
    M = align(none, st)
    ctrl = cv2.warpAffine(none, M, (512, 512), flags=cv2.INTER_CUBIC)
    ker = np.ones((9, 9), np.uint8)
    ov = cv2.erode(cv2.warpAffine(np.ones((512, 512), np.uint8), M, (512, 512), flags=cv2.INTER_NEAREST), ker) > 0
    fg = cv2.warpAffine(fg0.astype(np.float32), M, (512, 512), flags=cv2.INTER_LINEAR) > 0.5
    ov0 = cv2.warpAffine(ov.astype(np.uint8), M, (512, 512), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    a, b = cv2.GaussianBlur(ctrl, (0, 0), 4)[ov], cv2.GaussianBlur(st, (0, 0), 4)[ov]
    ncc = float(np.corrcoef(a, b)[0, 1]) if ov.sum() > MIN_PIX else float("nan")
    L0, Lc, Ls = laplace(none), laplace(ctrl), laplace(st)
    r = dict(cond=cond, prompt=pi, seed=x, scale=float(M[0, 0]), tx=float(M[0, 2]), ty=float(M[1, 2]),
             ncc=ncc, overlap=float(ov.mean()))
    for reg, m_st, m_0 in (("fg", ov & fg, ov0 & fg0), ("bg", ov & ~fg, ov0 & ~fg0)):
        r[f"none_{reg}"], r[f"ctrl_{reg}"], r[f"st_{reg}"] = lapvar(L0, m_0), lapvar(Lc, m_st), lapvar(Ls, m_st)
    return r, ctrl


def hboot_ragged(vals, keys, n=2000, seed=0):
    """Mean and 95% interval: resample captions, then the available pairs within each caption."""
    by = {}
    for v, k in zip(vals, keys):
        by.setdefault(k, []).append(v)
    groups = [np.array(g) for g in by.values()]
    if not groups:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        bs.append(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in pick]).mean())
    return float(np.mean(vals)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def f3(t):
    return "n/a" if np.isnan(t[0]) else f"{t[0]:+.3f} ({t[1]:+.3f}, {t[2]:+.3f})"


def decide(t):
    """t: dict of beyond-zoom intervals for one condition among comparable pairs."""
    if t["n"] < MIN_COMPARABLE:
        return "too few pairs align reliably: inconclusive, close this test case"
    bg_soft, fg_soft = t["bg"][2] < 0, t["fg"][2] < 0
    if bg_soft and not fg_soft and t["ratio"][1] > 0:
        return "extra background softening with the foreground kept sharp: proceed with background blur"
    if bg_soft or fg_soft:
        return "extra softening beyond zoom across the image: proceed, calling the target image softness"
    return "sharpness loss explained by zoom: close the blur experiment"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = SRC / ("zoom_smoke" if args.smoke else "zoom")
    out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for line in (SRC / "records.jsonl").open():
        r = json.loads(line)
        rec[(r["cond"], r["prompt"], r["seed"])] = r
    pidx = list(range(3 if args.smoke else 25))
    rows, ctrls = [], {}
    for c in CONDS:
        for pi in pidx:
            for x in SEEDS:
                r, ctrl = analyse_pair(c, pi, x)
                r["orig_bg"] = rec[(c, pi, x)]["log_bg"] - rec[("none", pi, x)]["log_bg"]
                r["orig_fg"] = rec[(c, pi, x)]["log_fg"] - rec[("none", pi, x)]["log_fg"]
                rows.append(r)
                ctrls[(c, pi, x)] = ctrl
        print(f"{c}: aligned {len(pidx) * len(SEEDS)} pairs", flush=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def ok(r, thr=NCC_MIN):
        return (r["ncc"] >= thr and r["overlap"] >= OVERLAP_MIN
                and all(np.isfinite(r[f"{a}_{g}"]) for a in ("none", "ctrl", "st") for g in ("fg", "bg")))

    lg = lambda r, a, g: np.log(r[f"{a}_{g}"] + 1e-6)
    L = ["# Zoom-only control for the SD 1.5 depth-of-field steering", ""]
    if args.smoke:
        L += ["**SMOKE TEST: 3 captions; no number here means anything.**", ""]
    L += [f"Pairs: {len(pidx)} test captions x seeds {SEEDS} per condition, from `outputs/q2_sd_dof/`. "
          f"Comparable: low-passed NCC >= {NCC_MIN} after alignment and overlap >= {OVERLAP_MIN:.0%} of the frame. "
          "Zoom-only control: the unsteered image warped (bicubic) by the estimated scale and shift onto the steered "
          "image's grid; scales within 1.5% of 1 count as no zoom and become a whole-pixel shift (bicubic resampling "
          "at a scale of 1.004 alone costs about 0.19 in log Laplacian variance). Changes are in log Laplacian variance; intervals bootstrap over captions, then pairs within "
          "caption.", "",
          "## Comparable pairs", "",
          "| condition | " + " | ".join(f"NCC >= {t}" for t in NCC_GRID) + " | median scale (all) | median scale (comparable) | comparable with zoom > 1.5% |",
          "|---|" + "---|" * (len(NCC_GRID) + 3)]
    for c in CONDS:
        rc = [r for r in rows if r["cond"] == c]
        cm = [r for r in rc if ok(r)]
        L.append(f"| {c} | " + " | ".join(f"{sum(ok(r, t) for r in rc)} / {len(rc)}" for t in NCC_GRID)
                 + f" | {np.median([r['scale'] for r in rc]):.2f} | "
                 + (f"{np.median([r['scale'] for r in cm]):.2f} |" if cm else "n/a |")
                 + f" {sum(r['scale'] != 1.0 for r in cm)} / {len(cm)} |")
    L += ["", "## Sharpness changes (log Laplacian variance)", "",
          "Rows 1-2: the original measure (whole image, each image's own depth mask). Rows 3-5: comparable pairs, "
          "same regions and resolution; row 3 = row 4 + row 5.", "",
          "| condition | measure | pairs | background | foreground | FG/BG ratio |", "|---|---|---|---|---|---|"]
    verdict = {}
    for c in CONDS:
        rc = [r for r in rows if r["cond"] == c]
        keys = [r["prompt"] for r in rc]
        L.append(f"| {c} | original, all pairs | {len(rc)} | {f3(hboot_ragged([r['orig_bg'] for r in rc], keys))} | "
                 f"{f3(hboot_ragged([r['orig_fg'] for r in rc], keys))} | "
                 f"{f3(hboot_ragged([r['orig_fg'] - r['orig_bg'] for r in rc], keys))} |")
        for thr in (NCC_MIN, 0.5):
            cm = [r for r in rc if ok(r, thr)]
            k = [r["prompt"] for r in cm]
            tag = "" if thr == NCC_MIN else f" (NCC >= {thr})"
            parts = [("original, comparable" + tag, lambda r, g: r[f"orig_{g}"]),
                     ("steered vs unsteered, same region" + tag, lambda r, g: lg(r, "st", g) - lg(r, "none", g)),
                     ("zoom-only control vs unsteered" + tag, lambda r, g: lg(r, "ctrl", g) - lg(r, "none", g)),
                     ("**steered vs zoom-only (beyond zoom)**" + tag, lambda r, g: lg(r, "st", g) - lg(r, "ctrl", g))]
            if thr != NCC_MIN:
                parts = parts[-1:]
            for name, fn in parts:
                bg = hboot_ragged([fn(r, "bg") for r in cm], k)
                fg = hboot_ragged([fn(r, "fg") for r in cm], k)
                rt = hboot_ragged([fn(r, "fg") - fn(r, "bg") for r in cm], k)
                L.append(f"| {c} | {name} | {len(cm)} | {f3(bg)} | {f3(fg)} | {f3(rt)} |")
                if thr == NCC_MIN and name.startswith("**"):
                    verdict[c] = dict(n=len(cm), bg=bg, fg=fg, ratio=rt)
    L += ["", "## Decision rule (from the beyond-zoom row, NCC >= %.1f)" % NCC_MIN, "",
          f"Fewer than {MIN_COMPARABLE} comparable pairs: inconclusive. Background interval below 0 and foreground "
          "interval not below 0 and ratio interval above 0: background blur. Any region below 0: image softness. "
          "Otherwise: explained by zoom.", ""]
    for c in CONDS:
        L.append(f"- {c}: {decide(verdict[c])}")

    # representative pairs: per condition, 3 comparable (random) + the least comparable; columns unsteered, control, steered
    rng = np.random.default_rng(0)
    show = []
    for c in CONDS:
        rc = [r for r in rows if r["cond"] == c]
        cm = [r for r in rc if ok(r)]
        if cm:
            show += [cm[i] for i in sorted(rng.choice(len(cm), min(3, len(cm)), replace=False))]
        show.append(min(rc, key=lambda r: r["ncc"] if np.isfinite(r["ncc"]) else -9))
    W, H = 256, 20
    sheet = Image.new("RGB", (3 * W, len(show) * (W + H)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, r in enumerate(show):
        c, pi, x = r["cond"], r["prompt"], r["seed"]
        ims = [Image.open(SRC / "images" / "none" / f"{pi:02d}_{x}.jpg").convert("L"),
               Image.fromarray(np.clip(ctrls[(c, pi, x)], 0, 255).astype(np.uint8)),
               Image.open(SRC / "images" / c / f"{pi:02d}_{x}.jpg").convert("L")]
        y = i * (W + H)
        bz = lg(r, "st", "bg") - lg(r, "ctrl", "bg")
        fz = lg(r, "st", "fg") - lg(r, "ctrl", "fg")
        dr.text((4, y + 4), f"{c} cap {pi} seed {x} | scale {r['scale']:.2f} NCC {r['ncc']:.2f} "
                            f"{'comparable' if ok(r) else 'NOT comparable'} | beyond zoom: BG {bz:+.2f} FG {fz:+.2f}",
                fill="black")
        for j, im in enumerate(ims):
            sheet.paste(im.resize((W, W), Image.LANCZOS), (j * W, y + H))
    sheet.save(out / "pairs.png")
    L += ["", f"`pairs.png` (grayscale, as measured): columns unsteered, zoom-only control, steered. Rows: per "
          "condition 3 random comparable pairs and the least comparable pair.", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
