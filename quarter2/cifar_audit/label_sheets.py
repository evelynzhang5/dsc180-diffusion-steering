"""Step 2 of the follow-up: can people label concrete changes consistently at CIFAR resolution?

Builds blinded labelling sheets from dev seeds 0-255 (development data only). Per class, 12 same-seed pairs whose
unsteered image is already the class and whose steered image still is; the 24 images are shuffled so a rater cannot
tell steered from unsteered. Questions are fixed here, before anyone labels. Run from the repo root:
    python quarter2/cifar_audit/label_sheets.py           # writes quarter2/cifar_audit/labeling/
    python quarter2/cifar_audit/label_sheets.py score     # agreement and steered-vs-unsteered, from filled CSVs
"""
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "labeling"
CLASSES = {4: "deer", 8: "ship"}
QUESTIONS = {
    "deer": {"antlers": "Antlers visible?",
             "whole_body": "Whole animal visible (head, body and at least two legs)?",
             "recognisable": "Clearly a deer?"},
    "ship": {"water": "Water visible?",
             "whole_hull": "Whole hull visible (both ends inside the frame)?",
             "recognisable": "Clearly a ship?"},
}
ANSWERS = ("yes", "no", "unclear")
N_PAIRS = 12


def build():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(ROOT / "outputs/q2_cifar_audit/dev_measures.npz", allow_pickle=True)["data"].item()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    key = {}
    for k, name in CLASSES.items():
        st, ung = d[f"steered_{k}"], d["unguided"]
        kept = np.flatnonzero((ung["pred"] == k) & (st["pred"] == k))
        seeds = np.sort(rng.choice(kept, N_PAIRS, replace=False))
        items = [(int(s), "steered", st["x"][s]) for s in seeds] + [(int(s), "unsteered", ung["x"][s]) for s in seeds]
        order = rng.permutation(len(items))
        key[name] = {}
        fig, axes = plt.subplots(4, 6, figsize=(12, 8.6))
        for pos, (ax, i) in enumerate(zip(axes.flat, order), start=1):
            seed, src, img = items[i]
            key[name][pos] = {"seed": seed, "source": src}
            ax.imshow(np.kron(img, np.ones((4, 4, 1))), interpolation="nearest")   # 4x, pixels kept as they are
            ax.set_title(str(pos), fontsize=11)
            ax.axis("off")
        fig.suptitle(f"{name}: 24 images (labels: {', '.join(QUESTIONS[name].values())})", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"sheet_{name}.png", dpi=100)
        plt.close(fig)
        with open(OUT / f"labels_{name}_RATER.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image"] + list(QUESTIONS[name]))
            for pos in range(1, 25):
                w.writerow([pos] + [""] * len(QUESTIONS[name]))
    (OUT / "key_DO_NOT_OPEN_BEFORE_LABELLING.json").write_text(json.dumps(key, indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


def kappa(a, b):
    """Cohen's kappa over the three answers."""
    a, b = np.array(a), np.array(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == x) * np.mean(b == x) for x in ANSWERS)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def score():
    key = json.loads((OUT / "key_DO_NOT_OPEN_BEFORE_LABELLING.json").read_text())
    L = ["# Labelling check: agreement and steered vs. unsteered", ""]
    for name, qs in QUESTIONS.items():
        raters = {}
        for p in sorted(OUT.glob(f"labels_{name}_*.csv")):
            rows = list(csv.DictReader(open(p)))
            if p.stem.endswith("_RATER") or any(r[q].strip().lower() not in ANSWERS for r in rows for q in qs):
                continue
            raters[p.stem.split("_", 2)[2]] = {int(r["image"]): {q: r[q].strip().lower() for q in qs} for r in rows}
        L += [f"## {name}: raters {', '.join(raters) or 'none complete'}", "",
              "| question | agreement (all raters same) | Cohen's kappa per rater pair | 'yes', steered | 'yes', unsteered |",
              "|---|---|---|---|---|"]
        if not raters:
            continue
        for q, text in qs.items():
            lab = {r: [raters[r][i][q] for i in range(1, 25)] for r in raters}
            agree = np.mean([len({lab[r][i] for r in raters}) == 1 for i in range(24)])
            kap = ", ".join(f"{a}-{b} {kappa(lab[a], lab[b]):+.2f}" for a, b in combinations(raters, 2)) or "-"
            src = [key[name][str(i)]["source"] for i in range(1, 25)]
            yes = {s: np.mean([lab[r][i] == "yes" for r in raters for i in range(24) if src[i] == s])
                   for s in ("steered", "unsteered")}
            L.append(f"| {text} | {agree:.0%} | {kap} | {yes['steered']:.0%} | {yes['unsteered']:.0%} |")
        L.append("")
    (OUT / "agreement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    score() if sys.argv[1:] == ["score"] else build()
