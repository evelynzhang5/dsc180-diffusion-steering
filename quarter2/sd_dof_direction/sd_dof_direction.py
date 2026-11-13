"""Q2 round 9: at matched background blur, does another direction blur the foreground less? (SD 1.5, depth of field)

Design and decision rules: README.md (the Design section is frozen by design.sha256; tune and confirm refuse to run
if it changed). Reuses round 4's sampler and meters (quarter2/sd_dof/sd_dof.py, copied unchanged from q2-sd-dof)
and its activations (outputs/q2_sd_dof/acts.pt) and directions (outputs/q2_sd_dof/directions.pt).

    python quarter2/sd_dof_direction/sd_dof_direction.py build tune select [--smoke]
    python quarter2/sd_dof_direction/sd_dof_direction.py confirm [--smoke]
    python quarter2/sd_dof_direction/sd_dof_direction.py secondary          # text CFG 5 and 10, after the decision

Stages:
  build      refit the RFM on round 4's activations; candidate directions; refit check; probe AUCs, cosines, spectrum
  tune       dev (16 person/animal captions x seeds 1, 2): none; every candidate and random x s {1, 2, 4, 8, 16};
             prompt x s {0.5, 1, 2, 4}
  select     one s per candidate (closest dev background change to the paper direction at s 4), challenger, the
             controls' s and the oracle's blur radius; frozen (selection.json + sha256)
  confirm    test (32 new person/animal + 32 object/scene captions x seeds 42, 123, 256, 789): none, paper direction,
             challenger, random, prompt; oracle afterwards; report with the decision
  secondary  text CFG 5 and 10: none, paper direction, challenger (32 test captions x seeds 42, 123)

All sharpness measures use the unsteered image's depth mask (nearest 40% = foreground); the paper's per-image mask
is a secondary column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "quarter2" / "sd_dof"))
import sd_dof as D  # noqa: E402  (round 4: SD sampler, DofMeter, hboot, NARFM path)

R4 = ROOT / "outputs" / "q2_sd_dof"                      # round 4's activations and directions
DINO = "facebook/dinov2-base"
CANDS = ["rfm_eig1", "rfm_eig2", "rfm_eig3", "meandiff", "meandiff_std"]
BASELINE = "rfm_eig1"
S_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
PROMPT_GRID = [0.5, 1.0, 2.0, 4.0]
S_PAPER = 4.0
DUP_COS = 0.95
REFIT_MIN_COS = 0.99
REACH = 0.25                                             # within 25% of the target background change
ORACLE_RADII = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]  # Gaussian sigma in pixels
FEATHER = 8.0                                            # feathering of the oracle's mask (Gaussian sigma, pixels)
DEV_SEEDS, TEST_SEEDS, SEC_SEEDS = [1, 2], [42, 123, 256, 789], [42, 123]
N_DEV, N_TEST_P, N_TEST_O = 16, 32, 32
ROUND4_CAPTIONS = 33                                     # validation images 0-32 were round 4's test and dev captions
# decision thresholds (README)
MIN_RATIO_GAIN, MIN_BG_SHARE, DINO_FLOOR, CLIP_FLOOR = 0.15, 0.75, -0.02, -0.5

# ---------------------------------------------------------------- captions: the keyword rule (fixed before sampling)
PEOPLE = set("""man men woman women person people boy boys girl girls child children kid kids baby babies toddler
toddlers player players guy guys lady ladies skier skiers surfer surfers skateboarder skateboarders snowboarder
snowboarders rider riders couple family crowd pedestrian pedestrians chef chefs worker workers officer officers police
policeman soldier soldiers student students teenager teenagers teen teens adult adults he she someone somebody
batter catcher pitcher umpire""".split())
ANIMALS = set("""dog dogs puppy puppies cat cats kitten kittens horse horses cow cows cattle sheep lamb lambs bird birds
duck ducks goose geese pigeon pigeons seagull seagulls elephant elephants giraffe giraffes zebra zebras bear bears animal
animals bull bulls goat goats pony ponies""".split())
NOT_LIVING = [r"teddy bears?", r"hot dogs?"]


def is_person_or_animal(caption):
    t = caption.lower()
    for pat in NOT_LIVING:
        t = re.sub(pat, " ", t)
    return bool(set(re.findall(r"[a-z]+", t)) & (PEOPLE | ANIMALS))


def caption_sets():
    """Dev: the first 16 P captions after round 4's 33; test P: the next 32 P; test O: the first 32 O."""
    from datasets import load_dataset
    ds = load_dataset("yerevann/coco-karpathy", split="validation")
    P, O = [], []
    for i in range(ROUND4_CAPTIONS, len(ds)):
        c = ds[i]["sentences"][0].strip().rstrip(".")
        (P if is_person_or_animal(c) else O).append((i, c))
        if len(P) >= N_DEV + N_TEST_P and len(O) >= N_TEST_O:
            break
    return {"dev": P[:N_DEV], "test_P": P[N_DEV:N_DEV + N_TEST_P], "test_O": O[:N_TEST_O]}


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


def design_ok():
    text = (HERE / "README.md").read_text()
    sec = text[text.index("## Design"):text.index("## Status")]
    return hashlib.sha256(sec.encode()).hexdigest() == (HERE / "design.sha256").read_text().split()[0]


# ---------------------------------------------------------------- build
def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def stage_build(cfg):
    path = cfg.out / "directions.pt"
    if path.exists():
        log("directions already built")
        return
    sys.path.insert(0, str(D.NARFM))
    from na_rfm.rfm_training import RFMTrainConfig, train_binary_rfm
    a = torch.load(R4 / "acts.pt")
    r4 = torch.load(R4 / "directions.pt")
    H, y, shallow = a["H"].float(), a["labels"], a["shallow"]
    is_sh = (y == shallow).numpy()
    t0 = time.time()
    res = None
    for dev in ("cuda", "cpu"):
        try:
            conf = RFMTrainConfig(bandwidth=D.BANDWIDTH, reg=1e-3, n_iters=5, top_k=5, device=dev, verbose=True)
            res = train_binary_rfm(H, y, target_value=shallow, config=conf)
            break
        except torch.cuda.OutOfMemoryError:
            log("RFM out of GPU memory; refitting on CPU")
            torch.cuda.empty_cache()
    E = res.eigenvectors.float().cpu()
    if E.shape[0] < E.shape[1]:
        E = E.T
    lam = res.eigenvalues.float().cpu()
    log(f"refit: held-out AUC {res.best_auc:.3f} [{time.time() - t0:.0f} s]; eigenvalues {lam.tolist()}")
    base = r4["rfm"].float()
    unit = lambda v: v / v.norm()
    refit_cos = float(abs(unit(E[:, 0]) @ unit(base)))
    log(f"refit check: |cos(refit top eigenvector, round 4 direction)| = {refit_cos:.4f}")
    if refit_cos < REFIT_MIN_COS and not cfg.smoke:
        sys.exit(f"refit check failed ({refit_cos:.4f} < {REFIT_MIN_COS}): stop and report (README)")

    def orient(v):
        p = (H @ v).numpy()
        return v if p[is_sh].mean() > p[~is_sh].mean() else -v

    std = H.std(0) + 1e-6
    cands = {"rfm_eig1": base, "rfm_eig2": orient(E[:, 1]), "rfm_eig3": orient(E[:, 2]),
             "meandiff": r4["meandiff"].float(), "meandiff_std": r4["meandiff"].float() / std}
    # probe AUCs on round 4's 20% split (numpy seed 42); mean differences refit on the other 80%
    perm = np.random.default_rng(42).permutation(len(y))
    val, trn = perm[:int(len(y) * 0.2)], perm[int(len(y) * 0.2):]
    md_tr = H[trn][is_sh[trn]].mean(0) - H[trn][~is_sh[trn]].mean(0)
    probe = {"rfm_eig1": base, "rfm_eig2": cands["rfm_eig2"], "rfm_eig3": cands["rfm_eig3"], "meandiff": md_tr,
             "meandiff_std": md_tr / std}
    aucs = {c: auc((H[val] @ v).numpy(), is_sh[val]) for c, v in probe.items()}
    cos = {f"{a_}|{b_}": float(unit(cands[a_]) @ unit(cands[b_])) for i, a_ in enumerate(CANDS) for b_ in CANDS[i + 1:]}
    kept, dropped = [], {}
    for c in CANDS:
        dup = [k for k in kept if cos.get(f"{k}|{c}", 0.0) > DUP_COS]
        if dup:
            dropped[c] = dup[0]
        else:
            kept.append(c)
    g = torch.Generator().manual_seed(0)
    rand = torch.randn(base.numel(), generator=g)
    torch.save({**cands, "random": rand, "kept": kept}, path)
    info = {"refit_auc": float(res.best_auc), "eigenvalues": lam.tolist(), "lambda1_over_lambda2": float(lam[0] / lam[1]),
            "refit_cos": refit_cos, "probe_auc": aucs, "cosine": cos, "kept": kept, "dropped_as_duplicate_of": dropped,
            "round4_rfm_auc": r4["rfm_auc"], "round4_meandiff_auc": r4["meandiff_auc"]}
    (cfg.out / "build.json").write_text(json.dumps(info, indent=1))
    L = ["# Candidate directions (round 4's activations: 1,200 photos at sigma 0.344, empty prompt)", "",
         f"Refit: held-out AUC {res.best_auc:.3f}; |cos(top eigenvector, round 4 direction)| = {refit_cos:.4f}; "
         f"eigenvalues {', '.join(f'{x:.3g}' for x in lam.tolist())} (lambda1 / lambda2 = {info['lambda1_over_lambda2']:.2f}).",
         "", "| candidate | probe AUC (held-out 20%) | kept |", "|---|---|---|"]
    L += [f"| {c} | {aucs[c]:.3f} | {'yes' if c in kept else 'no, duplicate of ' + dropped[c]} |" for c in CANDS]
    L += ["", "Probe AUCs of the RFM eigenvectors use a split that overlaps the RFM's own training split, so they are "
          "optimistic; they are not used to choose.", "", "| pair | cosine |", "|---|---|"]
    L += [f"| {k} | {v:+.3f} |" for k, v in cos.items()]
    (cfg.out / "build.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ---------------------------------------------------------------- generation and measurement
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


def run_set(cfg, ctx, split, cond, caps, seeds, mode, s, direction):
    """Generate, meter and embed every (caption, seed) of one condition; resumable via records.jsonl."""
    rec = ctx["rec"]
    todo = [(i, c, x) for i, c in caps for x in seeds if (split, cond, i, x) not in rec]
    d = cfg.out / "images" / split / cond
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for j in range(0, len(todo), 4):
        chunk = todo[j:j + 4]
        imgs = ctx["sd"].generate([c for _, c, _ in chunk], [x for _, _, x in chunk], mode, s, direction)
        with (cfg.out / "records.jsonl").open("a") as fh:
            for (i, c, x), im in zip(chunk, imgs):
                stem = f"{i:05d}_{x}"
                im.save(d / f"{stem}.png")
                m = ctx["meter"](im, c)
                np.save(d / f"{stem}_fg.npy", np.packbits(m.pop("fg_mask")))
                np.save(d / f"{stem}_dino.npy", ctx["dino"](im).astype(np.float16))
                r = dict(split=split, cond=cond, mode=mode, s=s, cap=i, seed=x, cfg=D.CFG, **m)
                rec[(split, cond, i, x)] = r
                fh.write(json.dumps(r) + "\n")
    if todo:
        log(f"{split}/{cond}: {len(todo)} images, {(time.time() - t0) / len(todo):.1f} s per image")


def load_records(cfg):
    rec = {}
    p = cfg.out / "records.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            rec[(r["split"], r["cond"], r["cap"], r["seed"])] = r
    return rec


def gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float64)


def mask(path):
    return np.unpackbits(np.load(path))[:512 * 512].reshape(512, 512).astype(bool)


def pair(cfg, split, cond, i, x, none_cond="none", img=None):
    """Paired measures of one image against the unsteered image of the same caption and seed (its depth mask)."""
    from scipy.ndimage import laplace
    import zoom_control as Z
    d0, d1 = cfg.out / "images" / split / none_cond, cfg.out / "images" / split / cond
    stem = f"{i:05d}_{x}"
    g0 = gray(d0 / f"{stem}.png")
    g1 = gray(d1 / f"{stem}.png") if img is None else np.asarray(img.convert("L")).astype(np.float64)
    fg = mask(d0 / f"{stem}_fg.npy")
    l0, l1 = laplace(g0), laplace(g1)
    lv = lambda l, m: float(np.log(l[m].var() + 1e-6))
    bg = lv(l1, ~fg) - lv(l0, ~fg)
    fgc = lv(l1, fg) - lv(l0, fg)
    rgb0 = np.asarray(Image.open(d0 / f"{stem}.png")).astype(float)
    rgb1 = (np.asarray(Image.open(d1 / f"{stem}.png")) if img is None else np.asarray(img)).astype(float)
    dlt = np.abs(rgb1 - rgb0).mean(2) / 255
    scale = float(Z.align(g0, g1)[0, 0])
    return dict(bg=bg, fg=fgc, ratio=fgc - bg, pix_fg=float(dlt[fg].mean()), pix_bg=float(dlt[~fg].mean()),
                log_scale=float(np.log(scale)))


def cell(cfg, ctx, split, cond, caps, seeds, none_cond="none"):
    """(captions x seeds) arrays of every paired measure for one condition."""
    rec = ctx["rec"]
    keys = ["bg", "fg", "ratio", "pix_fg", "pix_bg", "log_scale", "dino", "clip", "bg_own", "ratio_own"]
    out = {k: np.zeros((len(caps), len(seeds))) for k in keys}
    for a, (i, _) in enumerate(caps):
        for b, x in enumerate(seeds):
            p = pair(cfg, split, cond, i, x, none_cond)
            r0, r1 = rec[(split, none_cond, i, x)], rec[(split, cond, i, x)]
            e0 = np.load(cfg.out / "images" / split / none_cond / f"{i:05d}_{x}_dino.npy").astype(float)
            e1 = np.load(cfg.out / "images" / split / cond / f"{i:05d}_{x}_dino.npy").astype(float)
            p.update(dino=float(e0 @ e1), clip=r1["clip_caption"] - r0["clip_caption"],
                     bg_own=r1["log_bg"] - r0["log_bg"], ratio_own=np.log(r1["ratio"]) - np.log(r0["ratio"]))
            for k in keys:
                out[k][a, b] = p[k]
    return out


def fmt(t, nd=3):
    return f"{t[0]:+.{nd}f} ({t[1]:+.{nd}f}, {t[2]:+.{nd}f})"


# ---------------------------------------------------------------- oracle: blur outside the unsteered depth foreground
def oracle_image(cfg, split, i, x, radius):
    from scipy.ndimage import gaussian_filter
    d0 = cfg.out / "images" / split / "none"
    rgb = np.asarray(Image.open(d0 / f"{i:05d}_{x}.png")).astype(float)
    fg = mask(d0 / f"{i:05d}_{x}_fg.npy").astype(float)
    m = np.clip(gaussian_filter(fg, FEATHER), 0, 1)[..., None]
    blur = np.stack([gaussian_filter(rgb[..., c], radius) for c in range(3)], -1)
    return Image.fromarray(np.clip(m * rgb + (1 - m) * blur, 0, 255).round().astype(np.uint8))


def oracle_cell(cfg, split, caps, seeds, radius, dino=None, meter=None):
    keys = ["bg", "fg", "ratio", "pix_fg", "pix_bg", "dino", "clip"]
    out = {k: np.full((len(caps), len(seeds)), np.nan) for k in keys}
    for a, (i, c) in enumerate(caps):
        for b, x in enumerate(seeds):
            im = oracle_image(cfg, split, i, x, radius)
            p = pair(cfg, split, "none", i, x, img=im)
            if dino is not None:
                e0 = np.load(cfg.out / "images" / split / "none" / f"{i:05d}_{x}_dino.npy").astype(float)
                p["dino"] = float(e0 @ dino(im))
            if meter is not None:
                p["clip"] = meter.clip_scores(im, c)[0] - cfg_rec(cfg)[(split, "none", i, x)]["clip_caption"]
            for k in keys:
                if k in p:
                    out[k][a, b] = p[k]
    return out


_REC = {}


def cfg_rec(cfg):
    if "rec" not in _REC:
        _REC["rec"] = load_records(cfg)
    return _REC["rec"]


# ---------------------------------------------------------------- stages
def name(c, s):
    return f"{c}_s{s:g}"


def stage_tune(cfg, ctx):
    dirs = torch.load(cfg.out / "directions.pt")
    caps, seeds = cfg.caps["dev"], cfg.dev_seeds
    run_set(cfg, ctx, "dev", "none", caps, seeds, "none", 0.0, None)
    plan = [(BASELINE, S_PAPER)]
    for c in dirs["kept"] + ["random"]:
        plan += [(c, s) for s in cfg.s_grid if (c, s) != (BASELINE, S_PAPER)]
    plan += [("prompt", s) for s in cfg.prompt_grid]
    for c, s in plan:
        if c not in cfg.cands + ["random", "prompt"]:
            continue
        run_set(cfg, ctx, "dev", name(c, s), caps, seeds, "prompt" if c == "prompt" else "rfm", s,
                None if c == "prompt" else dirs[c])
    rows = {}
    L = ["# Dev grid (16 person/animal captions x seeds 1, 2; paired with the unsteered image)", "",
         "Changes in log Laplacian variance inside the unsteered image's depth mask. ratio = foreground - background "
         "(positive = selective). DINOv2: cosine with the unsteered image. Intervals: bootstrap over captions, then "
         "seeds.", "",
         "| condition | s | background | foreground | log FG/BG ratio | DINOv2 cosine | CLIP caption | log scale | "
         "ratio, own mask |", "|---|---|---|---|---|---|---|---|---|"]
    for c, s in plan:
        if c not in cfg.cands + ["random", "prompt"]:
            continue
        m = cell(cfg, ctx, "dev", name(c, s), caps, seeds)
        rows[name(c, s)] = {"cand": c, "s": s, **{k: float(v.mean()) for k, v in m.items()}}
        L.append(f"| {c} | {s:g} | {fmt(D.hboot(m['bg']))} | {fmt(D.hboot(m['fg']))} | {fmt(D.hboot(m['ratio']))} | "
                 f"{m['dino'].mean():.3f} | {m['clip'].mean():+.2f} | {m['log_scale'].mean():+.3f} | "
                 f"{m['ratio_own'].mean():+.3f} |")
    (cfg.out / "tune.json").write_text(json.dumps(rows, indent=1))
    (cfg.out / "tune.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def stage_select(cfg, ctx):
    rows = json.loads((cfg.out / "tune.json").read_text())
    target = rows[name(BASELINE, S_PAPER)]["bg"]
    pick, reach = {}, {}
    for c in sorted({r["cand"] for r in rows.values()}):
        rc = {k: r for k, r in rows.items() if r["cand"] == c}
        k = min(rc, key=lambda k: abs(rc[k]["bg"] - target))
        pick[c] = rc[k]["s"]
        reach[c] = abs(rc[k]["bg"] - target) <= REACH * abs(target)
    elig = [c for c in pick if c not in (BASELINE, "random", "prompt") and reach[c]]
    chal = max(elig, key=lambda c: (rows[name(c, pick[c])]["ratio"], rows[name(c, pick[c])]["dino"])) if elig else None
    # oracle: the blur radius whose dev background change is closest to the target
    caps, seeds = cfg.caps["dev"], cfg.dev_seeds
    obg = {r: float(np.nanmean(oracle_cell(cfg, "dev", caps, seeds, r)["bg"])) for r in ORACLE_RADII}
    orad = min(obg, key=lambda r: abs(obg[r] - target))
    sel = {"target_dev_bg": target, "picks": pick, "reaches_target": reach, "challenger": chal,
           "challenger_s": pick.get(chal), "baseline_s": S_PAPER, "random_s": pick.get("random"),
           "prompt_s": pick.get("prompt"), "oracle_radius": orad, "oracle_dev_bg": obg,
           "rule": "closest dev background change to the paper direction at s 4; eligible within 25%; challenger: "
                   "highest dev log FG/BG ratio change (ties: DINOv2)", "time": time.strftime("%F %T")}
    text = json.dumps(sel, indent=1)
    (cfg.out / "selection.json").write_text(text)
    (cfg.out / "selection.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    print(text, flush=True)


def stage_confirm(cfg, ctx):
    text = (cfg.out / "selection.json").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (cfg.out / "selection.sha256").read_text().strip(), \
        "selection changed after freezing"
    sel = json.loads(text)
    dirs = torch.load(cfg.out / "directions.pt")
    groups = {"P": cfg.caps["test_P"], "O": cfg.caps["test_O"]}
    caps = groups["P"] + groups["O"]
    seeds = cfg.test_seeds
    conds = {"paper": (BASELINE, S_PAPER)}
    if sel["challenger"]:
        conds["challenger"] = (sel["challenger"], sel["challenger_s"])
    conds["random"] = ("random", sel["random_s"])
    conds["prompt"] = ("prompt", sel["prompt_s"])
    run_set(cfg, ctx, "test", "none", caps, seeds, "none", 0.0, None)
    for role, (c, s) in conds.items():
        run_set(cfg, ctx, "test", name(c, s), caps, seeds, "prompt" if c == "prompt" else "rfm", s,
                None if c == "prompt" else dirs[c])
    res = {role: cell(cfg, ctx, "test", name(c, s), caps, seeds) for role, (c, s) in conds.items()}
    res["oracle"] = oracle_cell(cfg, "test", caps, seeds, sel["oracle_radius"], ctx["dino"], ctx["meter"])
    nP = len(groups["P"])
    L = [f"# Test: {nP} new person/animal (P) + {len(groups['O'])} object/scene (O) captions x seeds {seeds}", "",
         "Paired with the unsteered image, inside its depth mask; intervals: bootstrap over captions, then seeds.", "",
         "| condition | setting | background | foreground | log FG/BG ratio | ratio, P | ratio, O | DINOv2 cosine | "
         "CLIP caption | pixel change FG / BG | ratio, own mask |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for role, m in res.items():
        setting = f"radius {sel['oracle_radius']:g}" if role == "oracle" else name(*conds[role])
        own = f"{fmt(D.hboot(m['ratio_own']))}" if "ratio_own" in m else "n/a"
        L.append(f"| {role} | {setting} | {fmt(D.hboot(m['bg']))} | {fmt(D.hboot(m['fg']))} | "
                 f"{fmt(D.hboot(m['ratio']))} | {fmt(D.hboot(m['ratio'][:nP]))} | {fmt(D.hboot(m['ratio'][nP:]))} | "
                 f"{fmt(D.hboot(m['dino']))} | {fmt(D.hboot(m['clip']), 2)} | "
                 f"{m['pix_fg'].mean():.3f} / {m['pix_bg'].mean():.3f} | {own} |")
    summary = {}
    if "challenger" in res:
        ch, pa = res["challenger"], res["paper"]
        dr = D.hboot(ch["ratio"] - pa["ratio"])
        drP, drO = (ch["ratio"] - pa["ratio"])[:nP].mean(), (ch["ratio"] - pa["ratio"])[nP:].mean()
        share = ch["bg"].mean() / pa["bg"].mean()
        dd, dc = D.hboot(ch["dino"] - pa["dino"]), D.hboot(ch["clip"] - pa["clip"])
        checks = {"ratio gain >= 0.15 and CI above 0": dr[0] >= MIN_RATIO_GAIN and dr[1] > 0,
                  "same sign in P and O": np.sign(drP) == np.sign(drO) and drP != 0,
                  "background change >= 75% of the paper's": share >= MIN_BG_SHARE,
                  "DINOv2 difference CI lower end > -0.02": dd[1] > DINO_FLOOR,
                  "CLIP caption difference CI lower end > -0.5": dc[1] > CLIP_FLOOR}
        summary = {"ratio_diff": dr, "ratio_diff_P": drP, "ratio_diff_O": drO, "bg_share": share, "dino_diff": dd,
                   "clip_diff": dc, "checks": checks, "beats_paper": all(checks.values())}
        L += ["", "## Challenger vs. paper direction (paired)", "",
              f"- log FG/BG ratio difference: {fmt(dr)} (P {drP:+.3f}, O {drO:+.3f})",
              f"- background change, challenger / paper: {share:.2f}",
              f"- DINOv2 cosine difference: {fmt(dd)}; CLIP caption difference: {fmt(dc, 2)}", ""]
        L += [f"- {'PASS' if ok else 'FAIL'}: {k}" for k, ok in checks.items()]
        L += ["", f"**Decision rule: {'the challenger beats the paper direction' if summary['beats_paper'] else 'the challenger does not beat the paper direction'}.**"]
    else:
        L += ["", "No eligible challenger (none reached the target background change): the challenger branch of the "
              "decision fails."]
    rp = D.hboot(res["random"]["ratio"] - res["paper"]["ratio"])
    op = D.hboot(res["oracle"]["ratio"] - res["paper"]["ratio"])
    L += ["", "For the other branches of the decision table (paired differences in log FG/BG ratio change):",
          f"- oracle - paper: {fmt(op)}", f"- random - paper: {fmt(rp)}"]
    summary.update(oracle_minus_paper=op, random_minus_paper=rp)
    (cfg.out / "confirm.json").write_text(json.dumps({"conds": conds, "summary": summary,
                                                      "means": {r: {k: float(np.nanmean(v)) for k, v in m.items()}
                                                                for r, m in res.items()}}, indent=1, default=str))
    (cfg.out / "confirm.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    # 4 captions x (none + up to 4 conditions + oracle) <= 24 images
    examples(cfg, groups["P"][:2] + groups["O"][:2], seeds[0], ["none"] + [name(*conds[r]) for r in conds], sel)


def examples(cfg, caps, seed, conds, sel):
    """Contact sheet (<= 25 images per sheet): rows captions, columns conditions plus the oracle."""
    W = 160
    cols = conds + ["oracle"]
    sheet = Image.new("RGB", (W * len(cols), W * len(caps)), "white")
    for r, (i, _) in enumerate(caps):
        for c, cond in enumerate(cols):
            im = (oracle_image(cfg, "test", i, seed, sel["oracle_radius"]) if cond == "oracle"
                  else Image.open(cfg.out / "images" / "test" / cond / f"{i:05d}_{seed}.png"))
            sheet.paste(im.resize((W, W), Image.LANCZOS), (c * W, r * W))
    sheet.save(cfg.out / "examples.png")


def stage_secondary(cfg, ctx):
    sel = json.loads((cfg.out / "selection.json").read_text())
    dirs = torch.load(cfg.out / "directions.pt")
    caps = (cfg.caps["test_P"][:16] + cfg.caps["test_O"][:16]) if not cfg.smoke else cfg.caps["test_P"][:1]
    seeds = cfg.sec_seeds
    conds = [(BASELINE, S_PAPER)] + ([(sel["challenger"], sel["challenger_s"])] if sel["challenger"] else [])
    L = ["# Secondary: text CFG 5 and 10 (32 test captions x seeds 42, 123)", "",
         "| CFG | condition | background | log FG/BG ratio | DINOv2 cosine |", "|---|---|---|---|---|"]
    base_cfg = D.CFG
    for g in (5.0, 10.0):
        D.CFG = g
        split = f"cfg{g:g}"
        run_set(cfg, ctx, split, "none", caps, seeds, "none", 0.0, None)
        for c, s in conds:
            run_set(cfg, ctx, split, name(c, s), caps, seeds, "rfm", s, dirs[c])
            m = cell(cfg, ctx, split, name(c, s), caps, seeds)
            L.append(f"| {g:g} | {name(c, s)} | {fmt(D.hboot(m['bg']))} | {fmt(D.hboot(m['ratio']))} | "
                     f"{m['dino'].mean():.3f} |")
    D.CFG = base_cfg
    (cfg.out / "secondary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=["build", "tune", "select", "confirm", "secondary"])
    ap.add_argument("--smoke", action="store_true", help="2 dev captions x 1 seed, 2 candidates x s 4, 1+1 test captions")
    cfg = ap.parse_args()
    cfg.out = ROOT / "outputs" / ("q2_sd_dof_direction_smoke" if cfg.smoke else "q2_sd_dof_direction")
    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    if any(s in cfg.stages for s in ("tune", "confirm")) and not design_ok():
        sys.exit("README design section differs from design.sha256: refusing to generate")
    if "confirm" in cfg.stages and not (cfg.out / "selection.sha256").exists() and "select" not in cfg.stages:
        sys.exit("refusing to run the confirmation before the selection is frozen (run select first)")
    caps = caption_sets()
    (cfg.out / "captions.json").write_text(json.dumps(caps, indent=1))
    log("captions: " + ", ".join(f"{k} {len(v)}" for k, v in caps.items()) + f"; dev[0] {caps['dev'][0]}")
    if cfg.smoke:
        caps = {"dev": caps["dev"][:2], "test_P": caps["test_P"][:1], "test_O": caps["test_O"][:1]}
    cfg.caps = caps
    cfg.dev_seeds, cfg.test_seeds, cfg.sec_seeds = ([1], [42], [42]) if cfg.smoke else (DEV_SEEDS, TEST_SEEDS, SEC_SEEDS)
    cfg.s_grid = [S_PAPER] if cfg.smoke else S_GRID
    cfg.prompt_grid = [1.0] if cfg.smoke else PROMPT_GRID
    cfg.cands = [BASELINE, "meandiff_std"] if cfg.smoke else CANDS
    ctx = {}
    for st in cfg.stages:
        t0 = time.time()
        log(f"stage {st}{' (smoke)' if cfg.smoke else ''}")
        if st == "build":
            stage_build(cfg)
        else:
            if "sd" not in ctx and st != "select":
                ctx["sd"] = D.SD("cuda")
                ctx["meter"] = D.DofMeter("cuda")
                ctx["dino"] = Dino("cuda")
            ctx["rec"] = load_records(cfg)
            _REC["rec"] = ctx["rec"]
            {"tune": stage_tune, "select": stage_select, "confirm": stage_confirm,
             "secondary": stage_secondary}[st](cfg, ctx)
        log(f"stage {st} done [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
