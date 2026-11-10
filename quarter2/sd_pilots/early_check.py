"""Can the model's own early guess tell us which images will fail, and which a nudge will fix?

    python early_check.py --run RUN --prompts PROMPTS --out OUT_DIR [--seeds 0-7]

At a saved checkpoint k (after step 6, 15 or 24) the very next denoising step already computes a guess of the
finished image: z0_hat = (z - sqrt(1 - abar) * eps_hat) / sqrt(abar), with the normal CFG 7.5 prediction eps_hat.
We decode that guess ("preview"), score it with the same OWLv2 checks as the final images, and turn the score into
the signed margin from scorer_check.py (>= 0 means the preview passes). A real system gets the preview almost for
free (one VAE decode and one detector call), so it is a fair input for deciding whether to nudge.

Reports:
1. How well the preview margin at k predicts the final image's pass/fail (AUC; 0.5 = no better than chance).
2. Among final failures, how well it predicts "some real nudge at k repairs it"; among passes, "some nudge harms it".
3. Gated policies: apply (k, action) only when the preview at k fails, compared with applying it to everything.
Writes OUT_DIR/previews/*.jpg, OUT_DIR/previews.csv (resumable) and OUT_DIR/early_check.md.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from scorer_check import load, margin
from sd_branch import CFG, CHECKPOINTS, embed, load_pipe, parse_seeds, save_image


@torch.no_grad()
def previews(run, prompts, base, out, device):
    path = out / "previews.csv"
    done = {r["key"] for r in csv.DictReader(open(path))} if path.exists() else set()
    (out / "previews").mkdir(parents=True, exist_ok=True)
    todo = [(b, k) for b in base.values() for k in CHECKPOINTS if f"{b['key']}_k{k}" not in done]
    print(f"{len(todo)} previews to make ({len(done)} done)", flush=True)
    if not todo:
        return
    pipe = load_pipe(device)
    uncond = embed(pipe, [""])
    from evaluate import Scorer
    scorer = Scorer(device)
    from PIL import Image
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "base_key", "checkpoint", "det_1", "det_2", "count_scores"],
                           extrasaction="ignore")
        if new:
            w.writeheader()
        for n, (b, k) in enumerate(todo):
            p = prompts[b["prompt_id"]]
            z = torch.load(run / "latents" / f"{b['key']}.pt")[k][None].to(device)
            t = pipe.scheduler.timesteps[k]
            cond = embed(pipe, [p["prompt"]])
            e = pipe.unet(torch.cat([z] * 2), t, encoder_hidden_states=torch.cat([uncond, cond])).sample.chunk(2)
            eps = e[0] + CFG * (e[1] - e[0])
            abar = pipe.scheduler.alphas_cumprod[int(t)].to(device)
            z0 = (z - (1 - abar).sqrt() * eps) / abar.sqrt()
            key = f"{b['key']}_k{k}"
            img_path = out / "previews" / f"{key}.jpg"
            save_image(pipe, z0[0], img_path)
            s = scorer.score(Image.open(img_path).convert("RGB"), p)
            w.writerow({"key": key, "base_key": b["key"], "checkpoint": k, "det_2": "", "count_scores": "", **s})
            if n % 25 == 24:
                f.flush()
                print(f"  {n + 1}/{len(todo)}", flush=True)


def auc(pos, neg):
    """P(a random positive scores higher than a random negative); ties count half."""
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def analyze(base, br, prompts, out):
    pm = {}
    for r in csv.DictReader(open(out / "previews.csv")):
        pb = base.get(r["base_key"])
        if pb:
            pm[(r["base_key"], int(r["checkpoint"]))] = margin(r, prompts[pb["prompt_id"]])
    keys = [k for k in base if all((k, c) in pm for c in CHECKPOINTS)]
    final = {k: base[k]["m"] >= 0 for k in keys}
    br_ok = defaultdict(dict)  # base key -> (checkpoint, action) -> branch passes
    for r in br:
        if r["base_key"] in final:
            br_ok[r["base_key"]][(r["checkpoint"], f"{r['action']}:{r['strength']:g}")] = r["m"] >= 0
    acts = sorted({a for d in br_ok.values() for a in d})
    n = len(keys)
    lines = [f"## Early previews ({n} baselines)\n",
             "| checkpoint | preview passes | agrees with final | AUC: final pass | AUC: fail is repairable (n) | "
             "AUC: pass gets harmed (n) |", "| --- | --- | --- | --- | --- | --- |"]
    for c in CHECKPOINTS:
        m = {k: pm[(k, c)] for k in keys}
        rep = {k: any(v for (cc, _), v in br_ok[k].items() if cc == c) for k in keys if not final[k]}
        harm = {k: any(not v for (cc, _), v in br_ok[k].items() if cc == c) for k in keys if final[k]}
        lines.append(
            f"| {c} | {sum(v >= 0 for v in m.values()) / n:.0%} | "
            f"{sum((m[k] >= 0) == final[k] for k in keys) / n:.0%} | "
            f"{auc([m[k] for k in keys if final[k]], [m[k] for k in keys if not final[k]]):.2f} | "
            f"{auc([m[k] for k in rep if rep[k]], [m[k] for k in rep if not rep[k]]):.2f} ({sum(rep.values())} of {len(rep)}) | "
            f"{auc([m[k] for k in harm if not harm[k]], [m[k] for k in harm if harm[k]]):.2f} ({sum(harm.values())} of {len(harm)}) |")
    lines += ["\nAUC for 'repairable': does a *higher* preview margin among failures mean a nudge is more likely to "
              "fix it (above 0.5) or less likely (below 0.5)? For 'harmed', above 0.5 means images with a higher "
              "preview margin are harmed less.\n",
              "## Gated policies: nudge only when the preview at that checkpoint fails\n",
              "| policy | applied to | success |", "| --- | --- | --- |",
              f"| no intervention | 0% | {sum(final.values()) / n:.1%} |"]
    rows = []
    for c, a in acts:
        ks = [k for k in keys if (c, a) in br_ok[k]]
        always = sum(br_ok[k][(c, a)] for k in ks) / len(ks)
        gate = [k for k in ks if pm[(k, c)] < 0]
        gated = sum(br_ok[k][(c, a)] if pm[(k, c)] < 0 else final[k] for k in ks) / len(ks)
        rows.append((gated, f"| step {c}, {a}: always / gated | 100% / {len(gate) / len(ks):.0%} | "
                            f"{always:.1%} / **{gated:.1%}** |"))
    lines += [r for _, r in sorted(rows, reverse=True)]
    oracle = sum(max([final[k], *br_ok[k].values()]) for k in keys) / n
    lines += [f"| per-image hindsight oracle | | {oracle:.1%} |", "",
              "Gating uses only information available at the decision step, so it is a real (if simple) policy. "
              "The rows are chosen and evaluated on the same images, so treat the best row as optimistic.\n"]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = {p["id"]: p for p in json.load(open(a.prompts))}
    base, br = load(run, prompts)
    seeds = set(parse_seeds(a.seeds))
    base = {k: b for k, b in base.items() if b["seed"] in seeds and (run / "latents" / f"{k}.pt").exists()}
    previews(run, prompts, base, out, a.device)
    lines = [f"# Early check: {run}\n"] + analyze(base, br, prompts, out)
    (out / "early_check.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
