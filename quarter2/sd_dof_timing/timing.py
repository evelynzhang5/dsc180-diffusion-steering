"""Q2 round 13: can delaying the same activation edit preserve scene content at the same background blur? (SD 1.5)

A bounded diagnostic with a stopping rule (see README.md), on 7 scenes x seed 42. Run on a GPU pod:

    python quarter2/sd_dof_timing/timing.py

Fixed: round 9's paper direction (rfm_eig1), round 4's layer and edit size (w 0.7, s 4), caption, seed, text CFG 7.5,
50-step DDIM. The edit is masked to the background as in round 12: the per-block fraction of non-subject pixels, from
a subject mask on the unedited image. Only WHEN the edit is applied changes:
  full    round 4's window (every step with sigma in [0.041, 13.12])
  early / middle / late   equal-length runs of steps (a third of the window each), at its start, centre and end
Each steered run starts from the unsteered trajectory's own latent at the window's first step (cached), so steps
before the window are identical by construction. The strength c (h <- h + 0.7 |h| c m d) is tuned separately for
each window and target (secant search, <= 4 generations) so that every window is compared at the same achieved
background change. Targets (fixed before testing): -0.5 (round 12's mild target) and -2.0 (about a 1 px blur, visible).
Not reaching a target counts as a control failure, not a preservation success. Post-hoc blur is matched to each target.

Measures (round 12's, from diag.py): background change; subject RMS, interior RMS, SSIM and DINOv2 vs the
reference; boundary error against an approximate rendered target. Plus layout: IoU between the reference subject
mask and a mask found on the edited image, and the shift of the mask's centre (px).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "quarter2" / "diagnostics"))
import diag as G  # noqa: E402  (round 12: Seg, Scorer, posthoc, overlay; puts rounds 11, 10 and 4 on the path)

D, M, P, PL = G.D, G.M, G.P, G.PL
OUT = ROOT / "outputs" / "q2_sd_dof_timing"
SEED, S = 42, 4.0
SCENES = [0, 1, 3, 4, 5, 6, 7]          # round 10's captions with a subject (4 = the small bird round 12 lost)
TARGETS = [-0.5, -2.0]
TOL = 0.15
C_MAX = 24.0
BIRD_PROMPTS = ["bird", "small bird", "a small bird standing on the beach", "seagull"]
log, arr = G.log, G.arr


# ---------------------------------------------------------------- sampler with a cached unsteered trajectory
class Sampler:
    def __init__(self, sd):
        self.sd, self.p = sd, sd.pipe
        self.ts = list(self.p.scheduler.timesteps)
        self.uncond = sd.embed([""])

    def sigma(self, k):
        return float(self.sd.sig_all[int(self.ts[k])])

    @torch.no_grad()
    def decode(self, z):
        p = self.p
        x = p.vae.decode(z / p.vae.config.scaling_factor).sample.float()
        x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return Image.fromarray(x[0])

    @torch.no_grad()
    def unsteered(self, cap):
        """Plain CFG (as SD.generate mode none); returns the latent before every step and the image."""
        p, cond = self.p, self.sd.embed([cap])
        z = torch.randn(4, 64, 64, generator=torch.Generator().manual_seed(SEED))[None]
        z = (z * p.scheduler.init_noise_sigma).to(self.sd.dev, self.sd.dt)
        zs = []
        for t in self.ts:
            zs.append(z.clone())
            out = p.unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([self.uncond, cond])).sample
            eu, ec = out[:1], out[1:]
            z = p.scheduler.step(eu + D.CFG * (ec - eu), t, z).prev_sample
        return zs, cond, self.decode(z)

    @torch.no_grad()
    def steered(self, zs, cond, steps, d, m, c):
        """Start from the cached latent at the window's first step; third branch with the masked edit inside it."""
        p, hook, steps = self.p, self.sd.hook, set(steps)
        k0 = min(steps)
        z = zs[k0].clone()
        for k in range(k0, len(self.ts)):
            t, on = self.ts[k], k in steps
            if on:
                hook.comps, hook.d, hook.w, hook.rows = [(d, m, float(c))], d, D.W_RFM, (2, 3)
                out = p.unet(torch.cat([z, z, z]), t,
                             encoder_hidden_states=torch.cat([self.uncond, cond, cond])).sample
            else:
                out = p.unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([self.uncond, cond])).sample
            hook.comps, hook.d, hook.rows = None, None, None
            eu, ec = out[:1], out[1:2]
            eps = eu + D.CFG * (ec - eu)
            if on:
                eps = eps + S * (out[2:] - ec)
            z = p.scheduler.step(eps, t, z).prev_sample
        return self.decode(z)


def windows(sd):
    w = list(sd.window)
    n = len(w)
    L = n // 3
    mid = (n - L) // 2
    return dict(full=w, early=w[:L], middle=w[mid:mid + L], late=w[-L:])


def seg_at(seg, img, prompt, thr):
    old = G.SEG_THR
    G.SEG_THR = thr
    try:
        return seg(img, prompt)
    finally:
        G.SEG_THR = old


def layout(seg, prompt, thr, ref_mask, img):
    """Find the subject on the edited image with the same prompt and threshold as on the reference."""
    m, _ = seg_at(seg, img, prompt, thr)
    if not m.any():
        return 0.0, float("nan")
    iou = float((m & ref_mask).sum() / max((m | ref_mask).sum(), 1))
    c0 = np.array(np.nonzero(ref_mask)).mean(1)
    c1 = np.array(np.nonzero(m)).mean(1)
    return iou, float(np.linalg.norm(c1 - c0))


def subject_mask(seg, img, i):
    """Round 12's CLIPSeg mask; for the bird (which round 12 lost), try other prompts and thresholds and record it."""
    noun = G.SUBJECTS[i]
    m, prob = seg(img, noun)
    tries = [dict(prompt=noun, thr=G.SEG_THR, area=float(m.mean()))]
    if 0.0005 <= m.mean() <= 0.8:
        return m, noun, G.SEG_THR, tries
    best = None
    for pr in BIRD_PROMPTS if i == 4 else [noun]:
        for thr in (0.3, 0.2, 0.1):
            mm, pp = seg_at(seg, img, pr, thr)
            ct = float(pp[mm].mean() - pp[~mm].mean()) if mm.any() and not mm.all() else 0.0
            tries.append(dict(prompt=pr, thr=thr, area=float(mm.mean()), contrast=ct))
            if 0.0005 <= mm.mean() <= 0.3 and (best is None or ct > best[2]):
                best = (mm, pr, ct, thr)
    if best is None:
        return None, noun, None, tries
    return best[0], best[1], best[3], tries


def main():
    out = OUT
    (out / "images").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    D.Hook = PL.ComboHook
    sd, meter, dino, seg = D.SD("cuda"), D.DofMeter("cuda"), M.Dino("cuda"), G.Seg("cuda")
    sc, smp = G.Scorer(dino), Sampler(sd)
    d = (lambda v: (v.float() / v.float().norm()).view(1, 1280, 16, 16).cuda())(torch.load(M.DIRS)["rfm_eig1"])
    win = windows(sd)
    win_info = {k: dict(steps=[v[0], v[-1]], n=len(v), sigma=[smp.sigma(v[0]), smp.sigma(v[-1])]) for k, v in win.items()}
    log(f"windows: {win_info}")

    # determinism: identical inputs and sampler state must give identical outputs
    cap0 = G.CAPS[SCENES[0]]
    zs, cond, a = smp.unsteered(cap0)
    _, _, b = smp.unsteered(cap0)
    gen_ref = sd.generate([cap0], [SEED], "none", bs=1)[0]
    zero = smp.steered(zs, cond, win["full"], d, torch.zeros(1, 1, 16, 16, device="cuda"), 0.0)
    w1 = smp.steered(zs, cond, win["late"], d, torch.ones(1, 1, 16, 16, device="cuda"), 1.0)
    w2 = smp.steered(zs, cond, win["late"], d, torch.ones(1, 1, 16, 16, device="cuda"), 1.0)
    mx = lambda x, y: float(np.abs(arr(x) - arr(y)).max())
    det = dict(unsteered_twice=mx(a, b), vs_sd_generate=mx(a, gen_ref), zero_edit_third_branch=mx(a, zero),
               zero_edit_mean=float(np.abs(arr(a) - arr(zero)).mean()), steered_twice=mx(w1, w2))
    log(f"determinism: {det}")

    rows, sheet_rows, mask_rows = [], [], []
    for i in SCENES:
        ts = time.time()
        cap = G.CAPS[i]
        zs, cond, ref = smp.unsteered(cap)
        ref.save(out / "images" / f"{i:02d}_ref.png")
        subj, noun, thr, mtries = subject_mask(seg, ref, i)
        if subj is None:
            rows.append(dict(scene=i, skipped="no subject mask", mask_tries=mtries))
            log(f"scene {i}: no subject mask found")
            continue
        mask_rows.append(G.overlay(ref, subj))
        np.save(out / "images" / f"{i:02d}_subject.npy", np.packbits(subj))
        C = sc.ref(ref, subj, P.inv_depth(meter, ref))
        bgw = torch.from_numpy(M.bg_grid(subj)).view(1, 1, 16, 16).cuda()
        res = {}
        for T in TARGETS:
            for wn, steps in win.items():
                tried, c = [], 1.0 if T > -1 else 2.0
                for j in range(4):
                    im = smp.steered(zs, cond, steps, d, bgw, c)
                    m = sc(C, im)
                    tried.append((c, m, im))
                    if abs(m["bg"] - T) <= 0.05:
                        break
                    pts = sorted([(0.0, 0.0)] + [(cc, mm["bg"]) for cc, mm, _ in tried])
                    cn = M.interp([q[0] for q in pts], [q[1] for q in pts], T)
                    if cn is None:                       # not bracketed: grow (at most 4x per step)
                        cc, mm, _ = max(tried, key=lambda x: x[0])
                        cn = cc * min(T / mm["bg"], 4.0) if mm["bg"] < -1e-3 else 4 * cc
                    cn = float(np.clip(cn, 0.05, C_MAX))
                    if any(abs(cn - cc) < 1e-6 for cc, _, _ in tried):
                        break
                    c = cn
                c, m, im = min(tried, key=lambda x: abs(x[1]["bg"] - T))
                im.save(out / "images" / f"{i:02d}_{wn}_T{abs(T):g}.png")
                m["iou"], m["shift"] = layout(seg, noun, thr, subj, im)
                m.update(c=c, tries=len(tried), achieved=bool(abs(m["bg"] - T) <= TOL))
                res[f"{wn}|{T}"] = m
                res[f"{wn}|{T}"]["_img"] = im
            ph = min(((r, sc(C, G.posthoc(C["rgb"], subj, r))) for r in
                      [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]),
                     key=lambda x: abs(x[1]["bg"] - T))
            pm = ph[1]
            pm.update(c=ph[0], iou=1.0, shift=0.0, tries=0, achieved=bool(abs(pm["bg"] - T) <= TOL))
            res[f"posthoc|{T}"] = pm
            res[f"posthoc|{T}"]["_img"] = G.to_img(G.posthoc(C["rgb"], subj, ph[0]))
        if len(sheet_rows) < 5:
            T = TARGETS[-1]
            sheet_rows.append([G.overlay(ref, subj)] + [res[f"{w}|{T}"]["_img"] for w in ("full", "early", "late",
                                                                                         "posthoc")])
        for v in res.values():
            v.pop("_img")
        rows.append(dict(scene=i, caption=cap, noun=noun, seg_thr=thr, mask_area=float(subj.mean()), mask_tries=mtries,
                         res=res, seconds=time.time() - ts))
        log(f"scene {i}: " + "; ".join(f"{k} bg {v['bg']:+.2f} rms {v['subj_rms']:.3f} iou {v['iou']:.2f}"
                                        for k, v in res.items()))
        (out / "timing.json").write_text(json.dumps(dict(windows=win_info, determinism=det, rows=rows), indent=1))

    n_sheet = P.sheet(sheet_rows, out / "timing_examples.png") if sheet_rows else 0
    n_mask = P.sheet([mask_rows], out / "timing_masks.png") if mask_rows else 0
    report(rows, win_info, det, n_sheet, n_mask, time.time() - t_start)


def report(rows, win_info, det, n_sheet, n_mask, secs):
    done = [r for r in rows if "res" in r]
    meths = ["full", "early", "middle", "late", "posthoc"]
    keys = ["bg", "subj_rms", "int_rms", "subj_ssim", "subj_dino", "bnd_err", "iou", "shift"]
    L = [f"# Round 13: does delaying the edit preserve the scene at the same background blur? ({len(done)} scenes x "
         f"seed {SEED}; diagnostic with a stopping rule)", "",
         "Windows (DDIM step indices of 50, sigma at first and last step):", "",
         "| window | steps | n | sigma range |", "|---|---|---|---|"]
    L += [f"| {k} | {v['steps'][0]}-{v['steps'][1]} | {v['n']} | {v['sigma'][0]:.3f} - {v['sigma'][1]:.3f} |"
          for k, v in win_info.items()]
    L += ["", "Determinism (max absolute pixel difference, 0-255): " + ", ".join(f"{k} {v:.3g}" for k, v in det.items()),
          ""]
    summary = {}
    for T in TARGETS:
        L += [f"## Target background change {T}", "",
              "| method | reached | strength c (median) | subject RMS | interior RMS | subject SSIM | subject DINOv2 | "
              "boundary err | subject-mask IoU | centre shift (px) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for mt in meths:
            ms = [r["res"][f"{mt}|{T}"] for r in done]
            ok = [m for m in ms if m["achieved"]]
            mean = lambda k, xs=ok: float(np.nanmean([m[k] for m in xs])) if xs else float("nan")
            summary[f"{mt}|{T}"] = dict(reached=len(ok), n=len(ms), **{k: mean(k) for k in keys})
            L.append(f"| {mt} | {len(ok)}/{len(ms)} | {np.median([m['c'] for m in ms]):.2f} | {mean('subj_rms'):.4f} | "
                     f"{mean('int_rms'):.4f} | {mean('subj_ssim'):.3f} | {mean('subj_dino'):.3f} | {mean('bnd_err'):.4f} "
                     f"| {mean('iou'):.2f} | {mean('shift'):.1f} |")
        L += ["", "Per scene (subject RMS / IoU; * = target not reached):", "",
              "| scene | " + " | ".join(meths) + " |", "|---|" + "---|" * len(meths)]
        for r in done:
            L.append(f"| {r['scene']} {r['noun']} | " + " | ".join(
                f"{r['res'][f'{mt}|{T}']['subj_rms']:.3f} / {r['res'][f'{mt}|{T}']['iou']:.2f}"
                + ("" if r["res"][f"{mt}|{T}"]["achieved"] else "*") for mt in meths) + " |")
        L.append("")

    # decision (frozen in README before the run)
    n = len(done)
    dec = []
    for T in TARGETS:
        best, best_wins = None, -1
        for w in ("middle", "late"):
            wins = sum(r["res"][f"{w}|{T}"]["achieved"] and r["res"][f"full|{T}"]["achieved"]
                       and r["res"][f"{w}|{T}"]["subj_rms"] <= 0.8 * r["res"][f"full|{T}"]["subj_rms"] for r in done)
            if wins > best_wins:
                best, best_wins = w, wins
        s = summary[f"{best}|{T}"]
        reach = s["reached"]
        preserved = sum(r["res"][f"{best}|{T}"]["subj_rms"] <= 0.8 * r["res"][f"full|{T}"]["subj_rms"] for r in done)
        if reach >= 5 and best_wins >= 5 and s["iou"] >= 0.8:
            row = "1: later editing reaches the target and preserves the subject better; confirm on new scenes and seeds"
        elif preserved >= 5 and reach < 5:
            row = "2: later editing preserves the subject but cannot reach the target; no demonstrated improvement"
        elif reach >= 5:
            row = "3: later editing reaches the target but still changes pose/layout; stop this direction-and-layer approach"
        else:
            row = "none clean: neither window reaches the target reliably"
        ph = summary[f"posthoc|{T}"]
        beats = all(ph["subj_rms"] <= summary[f"{w}|{T}"]["subj_rms"] and ph["bnd_err"] <= summary[f"{w}|{T}"]["bnd_err"]
                    for w in ("full", "early", "middle", "late") if summary[f"{w}|{T}"]["reached"])
        dec.append((T, best, best_wins, reach, preserved, row, beats))
    L += ["## Decision", ""]
    for T, best, bw, reach, pres, row, beats in dec:
        L.append(f"- Target {T}: best later window `{best}`; reached {reach}/{n}; subject RMS <= 0.8 x full window and "
                 f"both on target in {bw}/{n}; preserved (ignoring the target) in {pres}/{n}. **Row {row}.** "
                 f"Post-hoc blur better on subject and boundary than every window that reached the target: "
                 f"{'yes' if beats else 'no'}.")
    if all(b for *_, b in dec):
        L.append("- **Row 4: post-processing remains better at both blur levels: the task does not justify a generative "
                 "steering method.**")
    L += ["", "Six (seven) scenes decide only whether to continue; any promising setting must then hold on untouched "
          f"scenes and seeds. `timing_examples.png` columns (target {TARGETS[-1]}): subject mask, full, early, late, "
          "post-hoc. `timing_masks.png`: every subject mask used.", f"\nTotal time {secs:.0f} s."]
    checks = [
        ("determinism: unsteered sampler twice gives identical images", det["unsteered_twice"] == 0.0,
         f"{det['unsteered_twice']:.3g}"),
        ("determinism: steered sampler twice gives identical images", det["steered_twice"] == 0.0,
         f"{det['steered_twice']:.3g}"),
        ("cached sampler reproduces SD.generate (mode none)", det["vs_sd_generate"] <= 1.0, f"{det['vs_sd_generate']:.3g}"),
        ("zero edit with the third branch matches unsteered (numerics only)", det["zero_edit_mean"] < 0.5,
         f"max {det['zero_edit_third_branch']:.3g}, mean {det['zero_edit_mean']:.3g}"),
        ("windows: early, middle, late have equal length", len({v["n"] for k, v in win_info.items() if k != "full"}) == 1,
         ", ".join(f"{k} {v['n']}" for k, v in win_info.items())),
        ("every scene has a subject mask (bird included)", len(done) == len(SCENES),
         "; ".join(f"{r['scene']}: {r.get('noun', '-')}" for r in rows)),
        ("all measures finite", all(np.isfinite(v[k]) for r in done for v in r["res"].values()
                                    for k in ("bg", "subj_rms", "bnd_err")), ""),
        ("contact sheets <= 25 images", n_sheet <= 25 and n_mask <= 25, f"{n_sheet}, {n_mask}"),
    ]
    (OUT / "timing_summary.json").write_text(json.dumps(dict(summary=summary, decision=dec), indent=1))
    if not P.write(OUT / "timing_checks.md", L, checks):
        sys.exit("timing: some checks failed")


if __name__ == "__main__":
    main()
