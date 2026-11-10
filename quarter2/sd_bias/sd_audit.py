"""Follow-up checks on the SD 1.5 blur-direction run (sd_bias.py): is the framing effect real, and do the meters work?

    python quarter2/sd_bias/sd_audit.py --run outputs/q2_sd_bias --out outputs/q2_sd_audit [--smoke]

A. Paired re-analysis of the saved evaluation images (RUN/eval/<condition>/<subject>_<seed>.jpg, 256 px):
   re-meter every image, then compare directions on the SAME prompt and seed (biased - balanced, biased - random,
   biased - opposed, balanced - random), per prompt and pooled. Pooled intervals are a two-level bootstrap
   (prompts, then seeds within prompt), so they reflect that there are only 4 person prompts.
B. Blinded pairs: 12 person-prompt pairs (biased vs. balanced, same prompt and seed), left/right shuffled. The key is
   written to a separate file.
C. Meter validation on controlled edits of real SD images (the "deep focus" images from D): background-only blur at
   three radii, whole-image blur, and subject-centred zoom crops (no blur added). A good blur meter moves with
   background blur and not with zoom; framing should move with zoom and not with blur.
D. Prompt positive control: held-out subjects as "shallow depth of field, blurred bokeh background" vs. "deep focus,
   everything in sharp focus" on the same seeds. Shows whether the meters register a blur difference that the model
   can produce at all, and what else that prompt change brings with it (e.g. framing).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sd_bias import BLUR_T, KEYS, SHARP_T, SD, TEST_OBJ, TEST_PEOPLE, Meter  # noqa: E402

CONDS = ["none", "balanced", "biased", "opposed", "random"]
CONTRASTS = [("biased", "balanced"), ("biased", "random"), ("biased", "opposed"), ("balanced", "random"),
             ("opposed", "random")]
SHOW = ["framing", "subject", "person_area", "clip_person", "person_on", "clip_blur", "bg_sharp", "sharp"]


def slug(s):
    return s.replace(" ", "_")


def hboot(D, n=2000, seed=0):
    """Mean of a (prompts x seeds) array and a 95% two-level bootstrap interval (prompts, then seeds)."""
    D = np.asarray(D, float)
    P, S = D.shape
    rng = np.random.default_rng(seed)
    ip = rng.integers(0, P, (n, P))
    is_ = rng.integers(0, S, (n, P, S))
    with np.errstate(all="ignore"):
        m = np.nanmean(D[ip[..., None], is_], axis=(1, 2))
    return float(np.nanmean(D)), float(np.nanquantile(m, 0.025)), float(np.nanquantile(m, 0.975))


def sboot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def f3(t):
    m, lo, hi = t
    star = " *" if (lo > 0 or hi < 0) else ""
    return f"{m:+.3f} ({lo:+.3f}, {hi:+.3f}){star}"


# ------------------------------------------------------------------ A. paired re-analysis
def part_a(run, out, meter, subjects, seeds, L):
    rec = {}
    path = out / "remeter.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            rec[(r["cond"], r["prompt"], r["seed"])] = r
    with path.open("a") as fh:
        for c in CONDS:
            for s in subjects:
                for x in seeds:
                    if (c, s, x) in rec:
                        continue
                    img = Image.open(run / "eval" / c / f"{slug(s)}_{x}.jpg").convert("RGB")
                    r = dict(cond=c, prompt=s, seed=x, **meter(img, s))
                    rec[(c, s, x)] = r
                    fh.write(json.dumps(r) + "\n")
    arr = {c: {k: np.array([[rec[(c, s, x)][k] for x in seeds] for s in subjects]) for k in KEYS} for c in CONDS}
    person = np.array([s in TEST_PEOPLE for s in subjects])
    groups = [("person prompts", person), ("object prompts", ~person)]

    L += ["## A. Paired re-analysis of the saved images", "",
          f"Re-metered from the saved 256 px JPEGs ({len(subjects)} prompts x seeds {seeds[0]}-{seeds[-1]}). "
          "First, a check that re-metering reproduces the original report's change vs. unedited (the original "
          "measured 512 px images, so background and whole-image sharpness differ; CLIP and box-based meters "
          "should be close).", "",
          "| condition | framing, person prompts | CLIP person, object prompts | CLIP blur, all |", "|---|---|---|---|"]
    for c in CONDS[1:]:
        d = {k: arr[c][k] - arr["none"][k] for k in KEYS}
        L.append(f"| {c} | {f3(hboot(d['framing'][person]))} | {f3(hboot(d['clip_person'][~person]))} | "
                 f"{f3(hboot(d['clip_blur']))} |")
    L += ["", "Intervals: two-level bootstrap over prompts, then seeds within prompt. * = interval excludes 0. With "
          "4 person prompts the prompt level is coarse: read the per-prompt table too.", ""]

    for gname, g in groups:
        L += [f"### Direct contrasts between directions, {gname} (same prompt and seed)", "",
              "| contrast | " + " | ".join(SHOW) + " |", "|---|" + "---|" * len(SHOW)]
        for a, b in CONTRASTS:
            L.append(f"| {a} - {b} | " + " | ".join(f3(hboot((arr[a][k] - arr[b][k])[g])) for k in SHOW) + " |")
        L.append("")

    L += ["### Per person prompt: framing change vs. unedited (seed bootstrap)", "",
          "| prompt | unedited framing | balanced | biased | opposed | random | biased - balanced |",
          "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(subjects):
        if s not in TEST_PEOPLE:
            continue
        cells = [f3(sboot(arr[c]["framing"][i] - arr["none"]["framing"][i])) for c in CONDS[1:]]
        bb = f3(sboot(arr["biased"]["framing"][i] - arr["balanced"]["framing"][i]))
        L.append(f"| {s} | {np.nanmean(arr['none']['framing'][i]):.3f} | " + " | ".join(cells) + f" | {bb} |")
    pos = sum(np.nanmean(arr["biased"]["framing"][i] - arr["balanced"]["framing"][i]) > 0
              for i, s in enumerate(subjects) if s in TEST_PEOPLE)
    L += ["", f"Biased framing above balanced in {pos} of {int(person.sum())} person prompts.", "",
          "### Is the framing change a detection artefact?", "",
          "Framing is the area of the subject's OWLv2 box. If the edit changed what the detector picks (e.g. a face box "
          "instead of the whole body) the subject score or the person box would change too. Person prompts, biased "
          "- balanced:", "",
          f"- subject detection score: {f3(hboot((arr['biased']['subject'] - arr['balanced']['subject'])[person]))}",
          f"- OWLv2 'a person' box area: "
          f"{f3(hboot((arr['biased']['person_area'] - arr['balanced']['person_area'])[person]))}",
          f"- correlation across images between the subject-box and person-box framing changes (biased - unedited): "
          f"{np.corrcoef((arr['biased']['framing'] - arr['none']['framing'])[person].ravel(), (arr['biased']['person_area'] - arr['none']['person_area'])[person].ravel())[0, 1]:+.2f}",
          ""]
    return arr


# ------------------------------------------------------------------ B. blinded pairs
def part_b(run, out, seeds, L):
    rng = np.random.default_rng(7)
    pairs = [(s, x) for s in TEST_PEOPLE for x in seeds[:3]]
    rng.shuffle(pairs)
    side = 200
    sheet = Image.new("RGB", (2 * (2 * side + 30) + 20, 6 * (side + 26) + 10), "white")
    dr = ImageDraw.Draw(sheet)
    key = []
    for n, (s, x) in enumerate(pairs):
        left_is_biased = bool(rng.integers(0, 2))
        ims = [Image.open(run / "eval" / c / f"{slug(s)}_{x}.jpg").convert("RGB").resize((side, side))
               for c in (("biased", "balanced") if left_is_biased else ("balanced", "biased"))]
        col, row = n % 2, n // 2
        x0, y0 = 10 + col * (2 * side + 30), 10 + row * (side + 26)
        dr.text((x0, y0), f"pair {n + 1}: {s}   A | B", fill="black")
        sheet.paste(ims[0], (x0, y0 + 14))
        sheet.paste(ims[1], (x0 + side + 4, y0 + 14))
        key.append({"pair": n + 1, "prompt": s, "seed": int(x), "A": "biased" if left_is_biased else "balanced",
                    "B": "balanced" if left_is_biased else "biased"})
    sheet.save(out / "blind_pairs.png")
    (out / "blind_key.json").write_text(json.dumps(key, indent=1))
    L += ["## B. Blinded pairs", "",
          f"`blind_pairs.png`: {len(pairs)} pairs, each the same person prompt and seed under the biased and the "
          "balanced direction, left/right shuffled. For each pair, write down which image (A or B) shows the person "
          "larger in the frame (or 'same'), then compare with `blind_key.json`. Do not open the key first.", ""]


# ------------------------------------------------------------------ C + D. positive control and meter validation
def mask_from_box(size, box, feather=12):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle([int(v) for v in box], fill=255)
    return m.filter(ImageFilter.GaussianBlur(feather))


def zoom(img, box, frac):
    W, H = img.size
    cw, ch = int(W * frac), int(H * frac)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(min(max(cx - cw / 2, 0), W - cw))
    y0 = int(min(max(cy - ch / 2, 0), H - ch))
    return img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.LANCZOS)


def part_cd(out, sd, meter, subjects, seeds, L):
    prompts, subj, sds = [], [], []
    for s in subjects:
        for x in seeds:
            prompts += [BLUR_T.format(s), SHARP_T.format(s)]
            subj += [s, s]
            sds += [x, x]
    _, imgs = sd.generate(prompts, sds)
    (out / "prompt_control").mkdir(parents=True, exist_ok=True)
    rows = []
    for p, s, x, im in zip(prompts, subj, sds, imgs):
        kind = "blur" if p == BLUR_T.format(s) else "sharp"
        im.resize((256, 256)).save(out / "prompt_control" / f"{kind}_{slug(s)}_{x}.jpg", quality=90)
        rows.append(dict(kind=kind, prompt=s, seed=x, **meter(im, s)))
    with (out / "prompt_control.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    get = lambda kind, k: np.array([[next(r[k] for r in rows if r["kind"] == kind and r["prompt"] == s and
                                          r["seed"] == x) for x in seeds] for s in subjects])
    person = np.array([s in TEST_PEOPLE for s in subjects])

    L += ["## D. Prompt positive control: 'shallow depth of field, bokeh' minus 'deep focus', same subject and seed",
          "", f"{len(subjects)} held-out subjects x seeds {seeds[0]}-{seeds[-1]}, 512 px, same sampler as the steering "
          "run. Paired change (blur prompt - sharp prompt); standardized = mean / SD of the paired changes.", "",
          "| meter | all prompts | standardized | person prompts | object prompts |", "|---|---|---|---|---|"]
    for k in ["bg_sharp", "clip_blur", "sharp", "framing", "subject", "clip_person", "person_on", "warmth"]:
        d = get("blur", k) - get("sharp", k)
        sdv = np.nanstd(d)
        L.append(f"| {k} | {f3(hboot(d))} | {np.nanmean(d) / sdv if sdv > 0 else float('nan'):+.2f} | "
                 f"{f3(hboot(d[person]))} | {f3(hboot(d[~person]))} |")
    L += ["", "Lower background sharpness and higher CLIP blur score = blurrier. If both register this prompt "
          "contrast, the meters can see a blur difference the model can make; what else changes (framing, subject) "
          "is what a 'blur' reference set built from these prompts carries along.", ""]

    # C. controlled edits of the deep-focus images
    base = [(im, s) for p, s, im in zip(prompts, subj, imgs) if p == SHARP_T.format(s)]
    edits = {"background blur r=2": None, "background blur r=4": None, "background blur r=8": None,
             "whole-image blur r=2": None, "zoom 0.8 (no blur)": None, "zoom 0.6 (no blur)": None}
    res = {e: {k: [] for k in KEYS} for e in edits}
    ex = []
    for i, (im, s) in enumerate(base):
        m0 = meter(im, s)
        _, box = meter.top(im, s)
        msk = mask_from_box(im.size, box)
        variants = {}
        for r in (2, 4, 8):
            variants[f"background blur r={r}"] = Image.composite(im, im.filter(ImageFilter.GaussianBlur(r)), msk)
        variants["whole-image blur r=2"] = im.filter(ImageFilter.GaussianBlur(2))
        variants["zoom 0.8 (no blur)"] = zoom(im, box, 0.8)
        variants["zoom 0.6 (no blur)"] = zoom(im, box, 0.6)
        for e, v in variants.items():
            m1 = meter(v, s)
            for k in KEYS:
                res[e][k].append(m1[k] - m0[k])
        if i < 2:
            ex.append([im] + list(variants.values()))
    sheet = Image.new("RGB", (7 * 160, len(ex) * 160), "white")
    for r_, rowims in enumerate(ex):
        for c_, v in enumerate(rowims):
            sheet.paste(v.resize((160, 160)), (c_ * 160, r_ * 160))
    sheet.save(out / "meter_edits_examples.png")
    L += ["## C. Meter validation: controlled edits of the deep-focus images", "",
          f"{len(base)} images. Change vs. the unedited image; * = 95% interval excludes 0. Background blur is applied "
          "outside the subject's OWLv2 box (feathered); zoom crops around the subject and resizes back (Lanczos), "
          "which adds no blur but magnifies, so whole-image sharpness drops a little by construction.", "",
          "| edit | background sharpness | CLIP blur score | whole-image sharpness | framing | subject | CLIP person |",
          "|---|---|---|---|---|---|---|"]
    for e in edits:
        L.append(f"| {e} | " + " | ".join(f3(sboot(res[e][k])) for k in
                                         ["bg_sharp", "clip_blur", "sharp", "framing", "subject", "clip_person"]) + " |")
    L += ["", "Examples (`meter_edits_examples.png`): unedited, background blur r=2/4/8, whole blur, zoom 0.8, 0.6.",
          "", "What a working blur meter should show: background sharpness falls and CLIP blur rises with the radius, "
          "and neither moves much under zoom alone. Framing should rise with zoom and stay put under blur.", ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("outputs/q2_sd_bias"))
    ap.add_argument("--out", type=Path, default=Path("outputs/q2_sd_audit"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    subjects = TEST_OBJ + TEST_PEOPLE
    eval_seeds = list(range(2, 10))
    ctrl_subjects, ctrl_seeds = subjects, [2, 3, 4, 5]
    if a.smoke:
        subjects, eval_seeds = [TEST_OBJ[0], TEST_PEOPLE[0]], [2, 3]
        ctrl_subjects, ctrl_seeds = [TEST_OBJ[0], TEST_PEOPLE[0]], [2]
    meter = Meter(dev)
    L = ["# SD 1.5 blur direction: follow-up audit" + (" (SMOKE: numbers mean nothing)" if a.smoke else ""), ""]
    print("A: re-metering saved images", flush=True)
    part_a(a.run, a.out, meter, subjects, eval_seeds, L)
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    print("B: blinded pairs", flush=True)
    part_b(a.run, a.out, list(range(2, 10)), L)
    print("C/D: prompt control and meter validation", flush=True)
    sd = SD(dev)
    part_cd(a.out, sd, meter, ctrl_subjects, ctrl_seeds, L)
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
