"""Q2 round 7: when do noise alignment (NA) and RFM steering help or interfere? (CIFAR-10, NA-RFM paper's code)

In the replication NA-RFM (NA lambda 3 for sigma >= 3.33, plus RFM on sigma <= 11.69) was below RFM alone (RFM at
every step) on ship (84% vs. 93%) and truck (83% vs. 90%). NA-RFM differs from RFM alone in two ways: it adds NA, and
it starts RFM late. Stages (design and decision rules in README.md):

  decompose  2 x 2 on all ten classes, seeds 0-255: NA {off, lambda 3} x RFM window {all steps, sigma <= 11.69}
  tune       grid on ship, truck, deer (deer: the combination helped), seeds 0-255:
             lambda {0, 1, 3} x RFM window {all, sigma <= 11.69} x RFM strength w {0.5, 1, 2}
  select     picks the best combined (lambda > 0) and best RFM-only (lambda = 0) setting by the fixed rule and
             freezes the choice (selection.json + sha256)
  confirm    the selected settings and the two paper settings on fresh seeds 1000-2023 (1,024 per class)

Every setting is the paper's run_cifar10_na_rfm.py with flags (w = --rfm-strength; s = 2 throughout), evaluated by
the paper's evaluate_saved_samples.py (ConvNeXt accuracy; per-class FID against that class's 6,000 CIFAR-10 images).
Run from the repo root on a GPU pod after `python cifar10_replication/replicate.py setup offline`:
    python quarter2/na_rfm_interplay/interplay.py decompose tune [--smoke]
    python quarter2/na_rfm_interplay/interplay.py select
    python quarter2/na_rfm_interplay/interplay.py confirm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cifar10_replication"))
import replicate as R  # noqa: E402  (paths, CLASSES, run, env_paths)

ALL_CLASSES = list(range(10))
TUNE_CLASSES = [8, 9, 4]                 # ship, truck, deer
FOCUS = [8, 9]                           # where NA-RFM lost accuracy
DEV = (0, 256)
TEST = (1000, 1024)
FID_TOL = 5.0                            # selection: mean FID at most the paper's RFM-only + this (dev, 256/class)
LAMBDAS, WINDOWS, STRENGTHS = (0.0, 1.0, 3.0), ("all", "late"), (0.5, 1.0, 2.0)


def setting(lam, window, w):
    return {"lam": lam, "window": window, "w": w}


def name(s):
    return f"lam{s['lam']:g}_{s['window']}_w{s['w']:g}"


def flags(s):
    win = "--rfm-sigma-lo 0 --rfm-sigma-hi 1e4" if s["window"] == "all" else "--rfm-sigma-lo 0.01 --rfm-sigma-hi 11.69"
    return f"--gaussian-strength {s['lam']:g} --gaussian-sigma-end 3.33 {win} --rfm-strength {s['w']:g}"


PAPER_RFM = setting(0.0, "all", 1.0)     # the replication's rfm_only
PAPER_NARFM = setting(3.0, "late", 1.0)  # the replication's na_rfm
DECOMPOSE = [PAPER_RFM, setting(0.0, "late", 1.0), setting(3.0, "all", 1.0), PAPER_NARFM]
GRID = [setting(l, win, w) for l, win, w in product(LAMBDAS, WINDOWS, STRENGTHS)]


def log(msg):
    print(f"== [{time.strftime('%F %T')}] {msg}", flush=True)


def cell_dir(cfg, split, s):
    return cfg.samples / split / name(s)


def generate(cfg, split, s, classes):
    start, n = cfg.seeds[split]
    out = cell_dir(cfg, split, s)
    for k in classes:
        cdir = out / R.CLASSES[k]
        if (cdir / "samples.pt").exists() and (cdir / "metrics.json").exists():
            continue
        # the replication already made the two paper settings on seeds 0-255 with the same code: reuse them
        old = {name(PAPER_RFM): "rfm_only", name(PAPER_NARFM): "na_rfm"}.get(name(s))
        src = R.WORK / "samples" / (old or "-") / R.CLASSES[k]
        if split == "dev" and not cfg.smoke and old and (src / "samples.pt").exists():
            shutil.copytree(src, cdir)
            continue
        t0 = time.time()
        R.run(f"{sys.executable} repro/run_cifar10_na_rfm.py --classes {k} --num-samples {n} --seed-start {start} "
              f"--batch-size 64 --inference-steps 100 --eval-mode hf_convnext --output-dir {out} {flags(s)}")
        acc = json.loads((cdir / "metrics.json").read_text())["accuracy"]
        print(f"   {split} {name(s)} {R.CLASSES[k]}: accuracy {acc:.1f}% [{time.time() - t0:.0f} s]", flush=True)


def measure(cfg, split, s, classes):
    """Accuracy and per-class FID (cached per class list)."""
    out = cell_dir(cfg, split, s)
    targets = ",".join(map(str, classes))
    q = out / f"quality_{targets.replace(',', '-')}.json"
    if not q.exists():
        R.run(f"{sys.executable} repro/evaluate_saved_samples.py --dataset cifar10 --samples-root {out} "
              f"--targets {targets} --output {q} --batch-size 64 --num-ref {cfg.num_ref}")
    d = json.loads(q.read_text())
    return {c["name"]: {"accuracy": c["accuracy"], "fid": c["fid"], "correct": c["correct"], "total": c["total"]}
            for c in d["classes"]}


def run_cells(cfg, split, cells, classes):
    res = {}
    for s in cells:
        generate(cfg, split, s, classes)
        res[name(s)] = {"setting": s, "classes": measure(cfg, split, s, classes)}
        r = res[name(s)]["classes"]
        print(f"   {name(s)}: " + ", ".join(f"{c} {v['accuracy']:.1f}% / FID {v['fid']:.1f}" for c, v in r.items()),
              flush=True)
    return res


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def table(res, classes, title):
    names = [R.CLASSES[k] for k in classes]
    L = [f"## {title}", "", "Accuracy % / FID per class; last columns: mean over the listed classes.", "",
         "| setting | NA lambda | RFM window | w | " + " | ".join(names) + " | mean acc | mean FID |",
         "|---|---|---|---|" + "---|" * len(names) + "---|---|"]
    for key, r in res.items():
        s, c = r["setting"], r["classes"]
        L.append(f"| {key} | {s['lam']:g} | {s['window']} | {s['w']:g} | "
                 + " | ".join(f"{c[n]['accuracy']:.0f} / {c[n]['fid']:.0f}" for n in names)
                 + f" | {mean(c[n]['accuracy'] for n in names):.1f} | {mean(c[n]['fid'] for n in names):.1f} |")
    return L + [""]


# ---------------------------------------------------------------- stages
def stage_decompose(cfg):
    res = run_cells(cfg, "dev", DECOMPOSE, cfg.all_classes)
    (cfg.out / "decompose.json").write_text(json.dumps(res, indent=1))
    L = ["# Stage A: NA x RFM window (seeds 0-255)", "",
         "Settings: lam0_all = RFM alone (paper), lam3_late = NA-RFM (paper); lam0_late = RFM alone but only for "
         "sigma <= 11.69; lam3_all = NA plus RFM at every step.", ""]
    L += table(res, cfg.all_classes, "All classes")
    (cfg.out / "decompose.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def stage_tune(cfg):
    res = run_cells(cfg, "dev", cfg.grid, cfg.tune_classes)
    (cfg.out / "tune.json").write_text(json.dumps(res, indent=1))
    L = ["# Stage B: tuning grid (seeds 0-255)", ""] + table(res, cfg.tune_classes, "Grid")
    (cfg.out / "tune.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def stage_select(cfg):
    """Fixed rule: within each family (combined: lambda > 0; RFM only: lambda = 0) take the setting with the highest
    mean accuracy over the tuning classes whose mean FID is at most the paper RFM-only's + FID_TOL (ties: lower FID)."""
    res = json.loads((cfg.out / "tune.json").read_text())
    names = [R.CLASSES[k] for k in cfg.tune_classes]
    acc = {k: mean(r["classes"][n]["accuracy"] for n in names) for k, r in res.items()}
    fid = {k: mean(r["classes"][n]["fid"] for n in names) for k, r in res.items()}
    cap = fid[name(PAPER_RFM)] + FID_TOL
    pick = {}
    for fam, test in (("combined", lambda s: s["lam"] > 0), ("rfm_only", lambda s: s["lam"] == 0)):
        ok = [k for k, r in res.items() if test(r["setting"]) and fid[k] <= cap]
        pick[fam] = max(ok, key=lambda k: (acc[k], -fid[k])) if ok else None
    sel = {"rule": f"max mean accuracy over {names} with mean FID <= paper RFM-only {fid[name(PAPER_RFM)]:.2f} + "
                   f"{FID_TOL}", "selected": {f: (res[k]["setting"] if k else None) for f, k in pick.items()},
           "dev": {f: ({"mean_acc": acc[k], "mean_fid": fid[k]} if k else None) for f, k in pick.items()},
           "time": time.strftime("%F %T")}
    text = json.dumps(sel, indent=1)
    (cfg.out / "selection.json").write_text(text)
    (cfg.out / "selection.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    print(text, flush=True)


def stage_confirm(cfg):
    f = cfg.out / "selection.json"
    text = f.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == (cfg.out / "selection.sha256").read_text().strip(), \
        "selection changed after freezing"
    sel = json.loads(text)["selected"]
    cells = {"paper RFM only": PAPER_RFM, "paper NA-RFM": PAPER_NARFM}
    for fam, s in sel.items():
        if s:
            cells[f"tuned {fam}"] = s
    uniq = {name(s): s for s in cells.values()}
    res = run_cells(cfg, "test", uniq.values(), cfg.tune_classes)
    (cfg.out / "confirm.json").write_text(json.dumps({"cells": {k: name(s) for k, s in cells.items()},
                                                      "results": res}, indent=1))
    names = [R.CLASSES[k] for k in cfg.tune_classes]
    L = [f"# Stage C: confirmation on seeds {cfg.seeds['test'][0]}-{sum(cfg.seeds['test']) - 1}", "",
         "| role | setting | " + " | ".join(f"{n} acc (95% CI) / FID" for n in names) + " |",
         "|---|---|" + "---|" * len(names)]
    for role, s in cells.items():
        c = res[name(s)]["classes"]
        cells_txt = []
        for n in names:
            p, m = c[n]["correct"] / c[n]["total"], c[n]["total"]
            half = 196 * (p * (1 - p) / m) ** 0.5
            cells_txt.append(f"{100 * p:.1f} (±{half:.1f}) / {c[n]['fid']:.1f}")
        L.append(f"| {role} | {name(s)} | " + " | ".join(cells_txt) + " |")
    (cfg.out / "confirm.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["decompose", "tune", "select", "confirm"])
    ap.add_argument("--smoke", action="store_true")
    cfg = ap.parse_args()
    cfg.all_classes = [8, 4] if cfg.smoke else ALL_CLASSES
    cfg.tune_classes = [8, 4] if cfg.smoke else TUNE_CLASSES
    cfg.seeds = {"dev": (0, 16), "test": (1000, 16)} if cfg.smoke else {"dev": DEV, "test": TEST}
    cfg.num_ref = 500 if cfg.smoke else -1
    cfg.grid = [PAPER_RFM, PAPER_NARFM, setting(1.0, "late", 2.0)] if cfg.smoke else GRID
    tag = "_smoke" if cfg.smoke else ""
    cfg.out = ROOT / "outputs" / f"q2_na_rfm_interplay{tag}"
    cfg.samples = R.WORK / f"samples_interplay{tag}"
    cfg.out.mkdir(parents=True, exist_ok=True)
    R.env_paths()
    for s in cfg.stages:
        if s == "confirm" and not (cfg.out / "selection.sha256").exists():
            sys.exit("refusing to run the confirmation before the selection is frozen (run select first)")
        t0 = time.time()
        log(f"stage {s}")
        {"decompose": stage_decompose, "tune": stage_tune, "select": stage_select, "confirm": stage_confirm}[s](cfg)
        log(f"stage {s} done [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
