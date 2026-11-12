"""Labelling feasibility on SD 1.5 depth-of-field pairs (development; images already inspected in round 4).

Question: can people consistently answer plainly visible questions about what steering changed, on same-seed
steered / unsteered pairs, so that changes can later be confirmed on fresh images?

Images: round 4's test set (q2-sd-dof), RFM steering at the paper's setting vs. unsteered, same caption and seed.
12 pairs drawn at random (not picked for striking differences): 12 different captions, one random seed each. In each
pair the steered image is randomly placed left (A) or right (B); the key is in a separate file.

Questions, fixed here before anyone labels (the main subject is written next to each caption):
  subject_A, subject_B   Is the main subject visible?                         yes / no / unclear
  larger                 In which image does the main subject fill more of the frame?   A / B / same / unclear
  softer                 In which image is the main subject less sharp?       A / B / same / unclear
Foreground softening is part of the target (the direction was learned for shallow depth of field), not by itself a
side effect; subject loss and framing are separate outcomes.

Run from the repo root, with round 4's images in outputs/q2_sd_dof/images/{rfm,none}/ and the captions in
outputs/q2_sd_dof/captions_test.json:
    python quarter2/sd_labeling/pair_sheets.py           # writes quarter2/sd_labeling/labeling/
    python quarter2/sd_labeling/pair_sheets.py score     # agreement and steered vs. unsteered, from filled CSVs
"""
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "outputs/q2_sd_dof/images"
OUT = Path(__file__).parent / "labeling"
SEEDS = [42, 123, 256, 789]
N_PAIRS = 12
SUBJECTS = ["child and yak", "kitchen", "girl and kitten", "toilet", "two sinks", "woman on a bicycle", "bath tub",
            "parked cars", "bathroom with a tub", "red helmet on a toilet", "car with surfboards", "cat on a toilet",
            "building with birds", "woman on a motorcycle", "person at a desk", "bananas", "small airplane",
            "airplane", "car", "traffic signal", "small plane", "building", "bus with an advertisement",
            "two giraffes", "giraffe"]
QUESTIONS = {"subject_A": ("yes", "no", "unclear"), "subject_B": ("yes", "no", "unclear"),
             "larger": ("A", "B", "same", "unclear"), "softer": ("A", "B", "same", "unclear")}


def build():
    from PIL import Image, ImageDraw, ImageFont
    caps = json.load(open(ROOT / "outputs/q2_sd_dof/captions_test.json"))
    assert len(caps) == len(SUBJECTS) == 25
    rng = np.random.default_rng(2026)
    prompts = np.sort(rng.choice(25, N_PAIRS, replace=False))
    pairs = []
    for i, p in enumerate(prompts, start=1):
        seed = int(rng.choice(SEEDS))
        steered_left = bool(rng.integers(2))
        pairs.append({"pair": i, "prompt": int(p), "seed": seed, "caption": caps[p], "subject": SUBJECTS[p],
                      "A": "steered" if steered_left else "unsteered", "B": "unsteered" if steered_left else "steered"})
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18) if hasattr(ImageFont, "load_default") else None
    S, PAD, TXT = 384, 12, 52
    for sheet in range(2):
        chunk = pairs[sheet * 6:(sheet + 1) * 6]
        canvas = Image.new("RGB", (2 * S + 3 * PAD, len(chunk) * (S + TXT + PAD) + PAD), "white")
        draw = ImageDraw.Draw(canvas)
        for r, pr in enumerate(chunk):
            y = PAD + r * (S + TXT + PAD)
            draw.text((PAD, y), f"Pair {pr['pair']}: \"{pr['caption']}\"", fill="black", font=font)
            draw.text((PAD, y + 24), f"Main subject: {pr['subject']}", fill="black", font=font)
            for c, side in enumerate("AB"):
                cond = "rfm" if pr[side] == "steered" else "none"
                im = Image.open(IMG / cond / f"{pr['prompt']:02d}_{pr['seed']}.jpg").convert("RGB").resize((S, S))
                x = PAD + c * (S + PAD)
                canvas.paste(im, (x, y + TXT))
                draw.rectangle([x, y + TXT, x + 26, y + TXT + 24], fill="white")
                draw.text((x + 6, y + TXT + 2), side, fill="black", font=font)
        canvas.save(OUT / f"sheet_{sheet + 1}.jpg", quality=92)
    with open(OUT / "labels_RATER.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair"] + list(QUESTIONS))
        for pr in pairs:
            w.writerow([pr["pair"]] + [""] * len(QUESTIONS))
    (OUT / "key_DO_NOT_OPEN_BEFORE_LABELLING.json").write_text(json.dumps(pairs, indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


def kappa(a, b, answers):
    a, b = np.array(a), np.array(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == x) * np.mean(b == x) for x in answers)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def score():
    key = {p["pair"]: p for p in json.loads((OUT / "key_DO_NOT_OPEN_BEFORE_LABELLING.json").read_text())}
    raters = {}
    for p in sorted(OUT.glob("labels_*.csv")):
        if p.stem == "labels_RATER":
            continue
        rows = {int(r["pair"]): {q: r[q].strip() for q in QUESTIONS} for r in csv.DictReader(open(p))}
        norm = {i: {q: (v.upper() if v.lower() in ("a", "b") else v.lower()) for q, v in r.items()}
                for i, r in rows.items()}
        if len(norm) == N_PAIRS and all(norm[i][q] in QUESTIONS[q] for i in norm for q in QUESTIONS):
            raters[p.stem.split("_", 1)[1]] = norm
        else:
            print(f"skipping {p.name}: incomplete or invalid answers")
    ids = sorted(key)
    # per-image subject labels, mapped to steered / unsteered
    subj = {r: {(i, key[i][s]): raters[r][i][f"subject_{s}"] for i in ids for s in "AB"} for r in raters}
    # pair questions, mapped to steered / unsteered / same / unclear
    rel = {r: {q: [key[i][raters[r][i][q]] if raters[r][i][q] in "AB" else raters[r][i][q] for i in ids]
               for q in ("larger", "softer")} for r in raters}
    L = ["# SD pair labelling: agreement and steered vs. unsteered", "", f"Raters: {', '.join(raters) or 'none'}", "",
         "| question | agreement (all raters same) | Cohen's kappa per rater pair | answers, per rater |",
         "|---|---|---|---|"]
    if len(raters) >= 1:
        cells = sorted(subj[next(iter(raters))])
        lab = {r: [subj[r][c] for c in cells] for r in raters}
        agree = np.mean([len({lab[r][j] for r in raters}) == 1 for j in range(len(cells))])
        kap = ", ".join(f"{a}-{b} {kappa(lab[a], lab[b], QUESTIONS['subject_A']):+.2f}"
                        for a, b in combinations(raters, 2)) or "-"
        ans = "; ".join(f"{r}: subject missing in steered {sum(subj[r][(i, 'steered')] == 'no' for i in ids)}/12, "
                        f"in unsteered {sum(subj[r][(i, 'unsteered')] == 'no' for i in ids)}/12" for r in raters)
        L.append(f"| Main subject visible? (24 images) | {agree:.0%} | {kap} | {ans} |")
        for q, text in (("larger", "Subject fills more of the frame (12 pairs)"),
                        ("softer", "Subject less sharp (12 pairs)")):
            ans4 = ("steered", "unsteered", "same", "unclear")
            agree = np.mean([len({rel[r][q][j] for r in raters}) == 1 for j in range(N_PAIRS)])
            kap = ", ".join(f"{a}-{b} {kappa(rel[a][q], rel[b][q], ans4):+.2f}"
                            for a, b in combinations(raters, 2)) or "-"
            ans = "; ".join(f"{r}: " + ", ".join(f"{x} {rel[r][q].count(x)}" for x in ans4) for r in raters)
            L.append(f"| {text} | {agree:.0%} | {kap} | {ans} |")
    (OUT / "agreement.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    score() if sys.argv[1:] == ["score"] else build()
