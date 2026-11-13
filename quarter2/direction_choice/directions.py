"""Q2 round 8: with strength tuned, does a different activation direction improve the accuracy-FID trade-off?
(CIFAR-10, the NA-RFM paper's code; design and decision rules in README.md)

Round 7 fixed how hard to push: NA-RFM with RFM at w = 0.5 instead of 1. This round asks whether there is a better
direction to push in. The model, NA (lambda 3 for sigma >= 3.33), the RFM window (sigma 0.01-11.69) and the
CFG-style scale (s = 2) stay fixed; only the block-9 direction and its strength w change.

  build    candidate directions for all ten classes from the block-9 features of the same 10,000 training images
           the paper's RFM was fit on (sigma 0.21); also their held-out AUC as a probe (reported, not used to select)
  tune     every candidate x w {0.25, 0.5, 0.75, 1, 1.5} on ship, truck, deer, seeds 0-255
  select   per candidate, the w with the lowest mean FID among those within 1 point of the round 7 baseline's mean
           accuracy; challenger = the non-baseline candidate with the lowest FID; frozen (selection.json + sha256)
  confirm  tuned baseline vs. challenger on all ten classes, fresh seeds 1000-2023 (1,024 per class)

The sampler scales the added direction by ||activation|| / ||direction||, so only a direction's orientation matters
and one w grid is fair to every candidate. Run from the repo root on a GPU pod after
`python cifar10_replication/replicate.py setup offline`:
    python quarter2/direction_choice/directions.py build tune select [--smoke]
    python quarter2/direction_choice/directions.py confirm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cifar10_replication"))
import replicate as R  # noqa: E402  (paths, CLASSES, run, env_paths, cifar_train)

ALL_CLASSES = list(range(10))
TUNE_CLASSES = [8, 9, 4]                         # ship, truck, deer (as in round 7)
DEV, TEST = (0, 256), (1000, 1024)
CANDS = ["rfm_top3", "rfm_eig1", "rfm_eig2", "meandiff", "meandiff_std"]
BASELINE = "rfm_top3"                            # the paper's direction (top-3 RFM eigenvectors, eigenvalue-weighted)
# rfm_eig1 is built and reported but not tuned: the build found cos(rfm_top3, rfm_eig1) >= 0.9994 in every class
# (the top eigenvalue dominates the weights), so it would only repeat the baseline. Amended before any dev image.
TUNED = [c for c in CANDS if c != "rfm_eig1"]
WS = (0.25, 0.5, 0.75, 1.0, 1.5)
ROUND7_W = 0.5                                   # round 7's tuned strength; its dev accuracy sets the accuracy floor
ACC_TOL = 1.0                                    # selection: mean accuracy at least baseline(w 0.5) - this
FIXED = "--gaussian-strength 3 --gaussian-sigma-end 3.33 --rfm-sigma-lo 0.01 --rfm-sigma-hi 11.69 --cfg-scale 2"
ROUND7 = R.WORK / "samples_interplay"            # round 7's samples of the baseline (same code and flags)
ROUND7_NAME = {0.5: "lam3_late_w0.5", 1.0: "lam3_late_w1"}


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


def name(c, w):
    return f"{c}_w{w:g}"


def concepts_root(c):
    return R.CONCEPTS if c == BASELINE else R.WORK / "directions" / c


# ---------------------------------------------------------------- build
def auc(score, y):
    """Mann-Whitney AUC of score for y = 1 vs y = 0."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stage_build(cfg):
    out = R.WORK / "directions"
    if all((out / c / f"class_{k}.pt").exists() for c in CANDS if c != BASELINE for k in ALL_CLASSES) \
            and (cfg.out / "build.json").exists():
        log("directions already built")
        return
    import os
    import torch
    os.chdir(R.REPO)
    from torch.utils.data import DataLoader, TensorDataset
    from na_rfm.activation_collection import collect_activations
    from na_rfm.model_loaders import load_openai_unet
    # the same 10,000 images, noise seed, sigma and block as the replication's RFM fit
    pixels, labels = R.cifar_train()
    first = np.concatenate([np.flatnonzero(labels == k)[:1000] for k in range(10)])
    images = torch.from_numpy(pixels[first]).permute(0, 3, 1, 2).float() / 127.5 - 1
    loader = DataLoader(TensorDataset(images, torch.from_numpy(labels[first])), batch_size=250)
    unet = load_openai_unet(str(R.CKPT), image_size=32, device="cuda")
    torch.manual_seed(0)
    H, Hy = collect_activations(unet, loader, block_name="input_blocks_9",
                                alpha_prod_t=1 / (1 + R.SIGMA_RFM ** 2), timestep=60, device="cuda")
    del unet
    torch.cuda.empty_cache()
    H = H.reshape(H.shape[0], -1).float().cuda()
    y_all = Hy.numpy().astype(int)
    print("block-9 features:", tuple(H.shape), flush=True)
    std = H.std(0) + 1e-6
    # an 80/20 split for the probe AUC (mean differences fit on 80%, scored on 20%)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(y_all), generator=g).numpy()
    tr, va = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]

    def meandiff(idx, k, standardize):
        yk = torch.from_numpy(y_all[idx] == k).cuda()
        Hi = H[torch.from_numpy(idx).cuda()]
        d = Hi[yk].mean(0) - Hi[~yk].mean(0)
        return d / std if standardize else d

    def orient(v, k):
        """Sign so that the target class projects higher than the rest (raw features, as the hook adds them)."""
        p = (H @ v).cpu().numpy()
        return v if p[y_all == k].mean() > p[y_all != k].mean() else -v

    info = {}
    for k in ALL_CLASSES:
        saved = torch.load(R.CONCEPTS / f"class_{k}.pt", weights_only=False)
        E = saved["eigenvectors"].float()
        if E.shape[0] < E.shape[1]:
            E = E.T
        E = E.cuda()
        top3 = saved["direction"].float().cuda()
        cands = {BASELINE: top3,
                 "rfm_eig1": orient(E[:, 0], k), "rfm_eig2": orient(E[:, 1], k),
                 "meandiff": meandiff(np.arange(len(y_all)), k, False),
                 "meandiff_std": meandiff(np.arange(len(y_all)), k, True)}
        eig1_paper_sign = float(torch.sign(torch.dot(E[:, 0] * int(saved["sign"]), cands["rfm_eig1"])).item())
        probe = {"rfm_top3": top3, "rfm_eig1": cands["rfm_eig1"], "rfm_eig2": cands["rfm_eig2"],
                 "meandiff": meandiff(tr, k, False), "meandiff_std": meandiff(tr, k, True)}
        yv = (y_all[va] == k).astype(int)
        Hv = H[torch.from_numpy(va).cuda()]
        aucs = {c: auc((Hv @ v).cpu().numpy(), yv) for c, v in probe.items()}
        unit = {c: v / v.norm() for c, v in cands.items()}
        cos = {f"{a}|{b}": float(torch.dot(unit[a], unit[b])) for i, a in enumerate(CANDS) for b in CANDS[i + 1:]}
        for c, v in cands.items():
            if c != BASELINE:
                (out / c).mkdir(parents=True, exist_ok=True)
                torch.save({"direction": v.cpu(), "candidate": c, "block_name": "input_blocks_9", "sigma": R.SIGMA_RFM},
                           out / c / f"class_{k}.pt")
        info[R.CLASSES[k]] = {"probe_auc_raw_projection": aucs, "rfm_auc_paper": float(saved["auc"]),
                              "cosine": cos, "eig1_sign_agrees_with_paper": eig1_paper_sign > 0}
        print(f"{R.CLASSES[k]}: probe AUC " + ", ".join(f"{c} {a:.3f}" for c, a in aucs.items())
              + f"; cos(top3, meandiff_std) {cos['rfm_top3|meandiff_std']:.2f}", flush=True)
    (cfg.out / "build.json").write_text(json.dumps(info, indent=1))
    L = ["# Candidate directions (block 9, 10,000 training images at sigma 0.21)", "",
         "Probe AUC: held-out (20%) AUC of the raw-feature projection onto each direction, class vs. the other nine. "
         "Reported for comparison only; selection uses generated images.", "",
         "| class | " + " | ".join(CANDS) + " | cos(top3, eig1) | cos(top3, meandiff) | cos(top3, meandiff_std) |",
         "|---|" + "---|" * (len(CANDS) + 3)]
    for cname, d in info.items():
        a, c = d["probe_auc_raw_projection"], d["cosine"]
        L.append(f"| {cname} | " + " | ".join(f"{a[x]:.3f}" for x in CANDS)
                 + f" | {c['rfm_top3|rfm_eig1']:.2f} | {c['rfm_top3|meandiff']:.2f} | {c['rfm_top3|meandiff_std']:.2f} |")
    (cfg.out / "build.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ---------------------------------------------------------------- generate and measure
def cell_dir(cfg, split, c, w):
    return cfg.samples / split / name(c, w)


def generate(cfg, split, c, w, classes):
    start, n = cfg.seeds[split]
    out = cell_dir(cfg, split, c, w)
    for k in classes:
        cdir = out / R.CLASSES[k]
        if (cdir / "samples.pt").exists() and (cdir / "metrics.json").exists():
            continue
        # round 7 made the baseline at w 0.5 and 1 (dev) and w 0.5 (test, ship/truck/deer) with the same code: reuse
        src = ROUND7 / split / ROUND7_NAME.get(w, "-") / R.CLASSES[k]
        if c == BASELINE and not cfg.smoke and (src / "samples.pt").exists() and (src / "metrics.json").exists():
            shutil.copytree(src, cdir)
            continue
        t0 = time.time()
        R.run(f"{sys.executable} repro/run_cifar10_na_rfm.py --classes {k} --num-samples {n} --seed-start {start} "
              f"--batch-size 64 --inference-steps 100 --eval-mode hf_convnext --output-dir {out} {FIXED} "
              f"--rfm-strength {w:g} --concepts-root {concepts_root(c)}")
        acc = json.loads((cdir / "metrics.json").read_text())["accuracy"]
        print(f"   {split} {name(c, w)} {R.CLASSES[k]}: accuracy {acc:.1f}% [{time.time() - t0:.0f} s]", flush=True)


def measure(cfg, split, c, w, classes):
    out = cell_dir(cfg, split, c, w)
    targets = ",".join(map(str, classes))
    q = out / f"quality_{targets.replace(',', '-')}.json"
    if not q.exists():
        R.run(f"{sys.executable} repro/evaluate_saved_samples.py --dataset cifar10 --samples-root {out} "
              f"--targets {targets} --output {q} --batch-size 64 --num-ref {cfg.num_ref}")
    d = json.loads(q.read_text())
    return {x["name"]: {"accuracy": x["accuracy"], "fid": x["fid"], "correct": x["correct"], "total": x["total"]}
            for x in d["classes"]}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


# ---------------------------------------------------------------- stages
def stage_tune(cfg):
    names = [R.CLASSES[k] for k in cfg.tune_classes]
    res = {}
    for c in cfg.cands:
        for w in cfg.ws:
            generate(cfg, "dev", c, w, cfg.tune_classes)
            r = measure(cfg, "dev", c, w, cfg.tune_classes)
            res[name(c, w)] = {"cand": c, "w": w, "classes": r}
            print(f"   {name(c, w)}: " + ", ".join(f"{n} {r[n]['accuracy']:.1f}% / FID {r[n]['fid']:.1f}"
                                                for n in names), flush=True)
    (cfg.out / "tune.json").write_text(json.dumps(res, indent=1))
    L = ["# Tuning grid (NA lambda 3 + direction late, seeds "
         f"{cfg.seeds['dev'][0]}-{sum(cfg.seeds['dev']) - 1})", "",
         "Accuracy % / FID per class (FID at this image count only; compare rows with each other).", "",
         "| direction | w | " + " | ".join(names) + " | mean acc | mean FID |",
         "|---|---|" + "---|" * len(names) + "---|---|"]
    for r in res.values():
        x = r["classes"]
        L.append(f"| {r['cand']} | {r['w']:g} | " + " | ".join(f"{x[n]['accuracy']:.0f} / {x[n]['fid']:.0f}"
                                                              for n in names)
                 + f" | {mean(x[n]['accuracy'] for n in names):.1f} | {mean(x[n]['fid'] for n in names):.1f} |")
    (cfg.out / "tune.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def stage_select(cfg):
    """Fixed rule (README): accuracy floor = baseline at w 0.5 (round 7's choice) minus ACC_TOL, mean over the tuning
    classes. Per candidate, the w with the lowest mean FID at or above the floor (ties: higher accuracy). Challenger =
    the non-baseline candidate whose pick has the lowest mean FID."""
    res = json.loads((cfg.out / "tune.json").read_text())
    names = [R.CLASSES[k] for k in cfg.tune_classes]
    acc = {k: mean(r["classes"][n]["accuracy"] for n in names) for k, r in res.items()}
    fid = {k: mean(r["classes"][n]["fid"] for n in names) for k, r in res.items()}
    floor = acc[name(BASELINE, ROUND7_W)] - ACC_TOL
    picks = {}
    for c in cfg.cands:
        ok = [k for k, r in res.items() if r["cand"] == c and acc[k] >= floor]
        picks[c] = min(ok, key=lambda k: (fid[k], -acc[k])) if ok else None
    others = [c for c in cfg.cands if c != BASELINE and picks[c]]
    chal = min(others, key=lambda c: fid[picks[c]]) if others else None
    # the "before strength tuning" comparison: every candidate at the paper's w = 1
    at1 = {c: {"mean_acc": acc[name(c, 1.0)], "mean_fid": fid[name(c, 1.0)]} for c in cfg.cands
           if name(c, 1.0) in res}
    sel = {"rule": f"per direction: min mean FID over {names} with mean accuracy >= {floor:.2f} "
                   f"(baseline at w {ROUND7_W:g} minus {ACC_TOL}); challenger: lowest-FID non-baseline pick",
           "baseline": {"cand": BASELINE, "w": res[picks[BASELINE]]["w"]},   # w 0.5 always meets the floor
           "challenger": {"cand": chal, "w": res[picks[chal]]["w"]} if chal else None,
           "picks": {c: ({"w": res[k]["w"], "mean_acc": acc[k], "mean_fid": fid[k]} if k else None)
                     for c, k in picks.items()},
           "at_w1": at1, "time": time.strftime("%F %T")}
    text = json.dumps(sel, indent=1)
    (cfg.out / "selection.json").write_text(text)
    (cfg.out / "selection.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    print(text, flush=True)


def stage_confirm(cfg):
    text = (cfg.out / "selection.json").read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (cfg.out / "selection.sha256").read_text().strip(), \
        "selection changed after freezing"
    sel = json.loads(text)
    cells = {"tuned baseline": sel["baseline"]}
    if sel["challenger"]:
        cells["challenger"] = sel["challenger"]
    res = {}
    for role, s in cells.items():
        generate(cfg, "test", s["cand"], s["w"], cfg.all_classes)
        res[role] = measure(cfg, "test", s["cand"], s["w"], cfg.all_classes)
    names = [R.CLASSES[k] for k in cfg.all_classes]
    (cfg.out / "confirm.json").write_text(json.dumps({"cells": cells, "results": res}, indent=1))
    L = [f"# Confirmation on seeds {cfg.seeds['test'][0]}-{sum(cfg.seeds['test']) - 1} "
         f"({cfg.seeds['test'][1]} per class)", "",
         "| class | " + " | ".join(f"{r}: {name(s['cand'], s['w'])} acc (95% CI) / FID" for r, s in cells.items())
         + " |", "|---|" + "---|" * len(cells)]
    for n in names:
        row = []
        for role in cells:
            x = res[role][n]
            p, m = x["correct"] / x["total"], x["total"]
            row.append(f"{100 * p:.1f} (±{196 * (p * (1 - p) / m) ** 0.5:.1f}) / {x['fid']:.1f}")
        L.append(f"| {n} | " + " | ".join(row) + " |")
    means = {r: (mean(res[r][n]["accuracy"] for n in names), mean(res[r][n]["fid"] for n in names)) for r in cells}
    L.append("| **mean** | " + " | ".join(f"{a:.1f} / {f:.1f}" for a, f in means.values()) + " |")
    if "challenger" in cells:
        b, c = res["tuned baseline"], res["challenger"]
        wins = sum(c[n]["fid"] < b[n]["fid"] for n in names)
        dfid = means["challenger"][1] - means["tuned baseline"][1]
        dacc = means["challenger"][0] - means["tuned baseline"][0]
        helps = dfid <= -2 and wins >= 0.8 * len(names) and dacc >= -1
        L += ["", f"Challenger - baseline: mean FID {dfid:+.1f}, mean accuracy {dacc:+.1f} points, "
                  f"lower FID in {wins} of {len(names)} classes.",
              f"Decision rule (README): {'direction choice improves the tuned baseline' if helps else 'no reliable improvement over the tuned baseline'}."]
    (cfg.out / "confirm.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["build", "tune", "select", "confirm"])
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    cfg.all_classes = [8, 4] if cfg.smoke else ALL_CLASSES
    cfg.tune_classes = [8, 4] if cfg.smoke else TUNE_CLASSES
    cfg.seeds = {"dev": (0, 16), "test": (1000, 16)} if cfg.smoke else {"dev": DEV, "test": TEST}
    cfg.num_ref = 500 if cfg.smoke else -1
    cfg.cands = [BASELINE, "meandiff_std"] if cfg.smoke else TUNED
    cfg.ws = (0.5, 1.0) if cfg.smoke else WS
    tag = "_smoke" if cfg.smoke else ""
    cfg.out = ROOT / "outputs" / f"q2_direction_choice{tag}"
    cfg.samples = R.WORK / f"samples_direction{tag}"
    cfg.out.mkdir(parents=True, exist_ok=True)
    R.env_paths()
    for s in cfg.stages:
        if s == "confirm" and not (cfg.out / "selection.sha256").exists():
            sys.exit("refusing to run the confirmation before the selection is frozen (run select first)")
        t0 = time.time()
        log(f"stage {s}")
        {"build": stage_build, "tune": stage_tune, "select": stage_select, "confirm": stage_confirm}[s](cfg)
        log(f"stage {s} done [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
