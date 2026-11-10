"""How much of the oracle gap survives a stricter reading of the scorer?

    python scorer_check.py --run RUN --prompts PROMPTS --out OUT_DIR [--aug]

A. Margins (no model needed). Each image gets a signed margin m: how far its verdict is from the 0.2 cutoff.
   - two_object: m = min(det_1, det_2) - 0.2 (the weaker object decides).
   - counting (asked for n): with box scores s_1 >= s_2 >= ..., the count is n exactly when s_n >= 0.2 and
     s_(n+1) < 0.2, so m = min(s_n - 0.2, 0.2 - s_(n+1)).
   Passing means m >= 0. A "confident" outcome change needs the margin to clear delta on the far side: a repair
   needs base m < -delta and branch m >= +delta (harms mirrored). Changes that don't clear it are ignored (the
   branch keeps the baseline's outcome). We then recompute best fixed policy, oracle and the gap for each delta.
B. Stability (--aug, runs OWLv2). Re-scores every baseline image, and every branch whose outcome differs from its
   baseline, after a mirror flip and after a 90% center crop. Reports how often the pass/fail verdict agrees
   with the original scoring, and how many repairs and harms hold in all three views.
Writes OUT_DIR/scorer_check.md (and OUT_DIR/scores_aug.csv with --aug). Never changes the run's files.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

TAU = 0.2
DELTAS = [0.0, 0.05, 0.1, 0.15]


def margin(s, p):
    if p.get("task") == "counting":
        v = sorted((float(x) for x in s["count_scores"].split()), reverse=True) + [0.0] * 12
        n = p["count"]
        return min(v[n - 1] - TAU, TAU - v[n])
    return min(float(s["det_1"]), float(s["det_2"])) - TAU


def load(run, prompts):
    scores = {r["key"]: r for r in csv.DictReader(open(run / "scores.csv"))}
    base = {}
    for l in open(run / "baseline.jsonl"):
        r = json.loads(l)
        if r["key"] in scores and r["prompt_id"] in prompts:
            base[r["key"]] = {**r, "m": margin(scores[r["key"]], prompts[r["prompt_id"]])}
    br = []
    for l in open(run / "branches.jsonl"):
        r = json.loads(l)
        if r["key"] in scores and r["action"] != "none" and r["base_key"] in base:
            br.append({**r, "m": margin(scores[r["key"]], prompts[r["prompt_id"]])})
    return base, br


def trajectories(base, br, delta):
    """Pass/fail per trajectory and policy, counting only outcome changes that clear delta."""
    traj = {k: {"none": b["m"] >= 0} for k, b in base.items()}
    for r in br:
        b0 = base[r["base_key"]]["m"]
        ok = b0 >= 0
        if not ok and b0 < -delta and r["m"] >= delta:
            ok = True
        elif ok and b0 >= delta and r["m"] < -delta:
            ok = False
        pol = (r["checkpoint"], f"{r['action']}:{r['strength']:g}")
        traj[r["base_key"]][pol] = ok
    return traj


def gap(traj, keys):
    pols = sorted({p for k in keys for p in traj[k]}, key=str)
    fixed = {p: sum(traj[k].get(p, traj[k]["none"]) for k in keys) / len(keys) for p in pols}
    oracle = sum(max(traj[k].values()) for k in keys) / len(keys)
    return fixed["none"], max(fixed.values()), oracle


def interval(traj, prompt_of, n=2000):
    by = defaultdict(list)
    for k in traj:
        by[prompt_of[k]].append(k)
    ps, rng, gs = sorted(by), random.Random(0), []
    for _ in range(n):
        keys = [k for p in rng.choices(ps, k=len(ps)) for k in by[p]]
        _, f, o = gap(traj, keys)
        gs.append(o - f)
    gs.sort()
    return gs[int(0.025 * n)], gs[int(0.975 * n)]


def part_a(base, br):
    br = [r for r in br if r["action"] != "placebo"]
    prompt_of = {k: b["prompt_id"] for k, b in base.items()}
    flips = [r for r in br if (r["m"] >= 0) != (base[r["base_key"]]["m"] >= 0)]
    near = sum(min(abs(r["m"]), abs(base[r["base_key"]]["m"])) < 0.1 for r in flips)
    out = ["## A. Margins: counting only outcome changes that clear the cutoff by delta\n",
           f"{len(base)} baselines, {len(br)} real-nudge branches, {len(flips)} outcome flips; "
           f"{near} of them ({near / max(len(flips), 1):.0%}) have the baseline or the branch within 0.1 of the cutoff.\n",
           "| delta | repairs | harms | no intervention | best fixed | oracle | **oracle - best fixed** (95% interval) |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for d in DELTAS:
        traj = trajectories(base, br, d)
        none, fixed, oracle = gap(traj, list(traj))
        lo, hi = interval(traj, prompt_of)
        rep = sum(traj[r["base_key"]][(r["checkpoint"], f"{r['action']}:{r['strength']:g}")] and not traj[r["base_key"]]["none"] for r in br)
        harm = sum(traj[r["base_key"]]["none"] and not traj[r["base_key"]][(r["checkpoint"], f"{r['action']}:{r['strength']:g}")] for r in br)
        out.append(f"| {d:g} | {rep} | {harm} | {none:.1%} | {fixed:.1%} | {oracle:.1%} | "
                   f"**{100 * (oracle - fixed):.1f}** ({100 * lo:.1f} to {100 * hi:.1f}) |")
    out.append("\ndelta 0 is the original analysis (the 'best fixed' here is chosen on the same trajectories, so it "
               "is the optimistic version). If the gap shrinks quickly as delta grows, most of it was borderline verdicts.\n")
    return out


def part_menus(base, br):
    """Hindsight oracle of the real nudges vs. the placebo menu, counting only changes that clear delta."""
    if not any(r["action"] == "placebo" for r in br):
        return []
    prompt_of = {k: b["prompt_id"] for k, b in base.items()}
    out = ["## Real nudges vs. placebo, counting only outcome changes that clear the cutoff by delta\n",
           "| delta | real-nudge oracle | placebo oracle | **real minus placebo** (95% interval over prompts) |",
           "| --- | --- | --- | --- |"]
    by = defaultdict(list)
    for k in base:
        by[prompt_of[k]].append(k)
    for d in DELTAS:
        traj = trajectories(base, br, d)
        orc = lambda keys, plac: sum(max(v for p, v in traj[k].items()
                                         if p == "none" or p[1].startswith("placebo") == plac) for k in keys) / len(keys)
        keys = list(traj)
        ps, rng, diffs = sorted(by), random.Random(0), []
        for _ in range(2000):
            ks = [k for p in rng.choices(ps, k=len(ps)) for k in by[p]]
            diffs.append(orc(ks, False) - orc(ks, True))
        diffs.sort()
        out.append(f"| {d:g} | {orc(keys, False):.1%} | {orc(keys, True):.1%} | "
                   f"**{100 * (orc(keys, False) - orc(keys, True)):+.1f}** ({100 * diffs[50]:+.1f} to {100 * diffs[1949]:+.1f}) |")
    return out + [""]


def part_b(run, prompts, base, br, out_dir, device):
    br = [r for r in br if r["action"] != "placebo"]
    import torch
    from PIL import Image, ImageOps
    from evaluate import Scorer
    flips = [r for r in br if (r["m"] >= 0) != (base[r["base_key"]]["m"] >= 0)]
    todo = list(base.values()) + flips
    path = out_dir / "scores_aug.csv"
    done = {}
    if path.exists():
        done = {(r["key"], r["view"]): r for r in csv.DictReader(open(path))}
    scorer = None
    views = {"flip": lambda im: ImageOps.mirror(im),
             "crop": lambda im: im.crop((int(.05 * im.width), int(.05 * im.height), int(.95 * im.width),
                                         int(.95 * im.height))).resize(im.size)}
    fields = ["key", "view", "det_1", "det_2", "count_scores"]
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        n = 0
        for r in todo:
            for v, fn in views.items():
                if (r["key"], v) in done:
                    continue
                if scorer is None:
                    scorer = Scorer(device)
                img = fn(Image.open(run / r["image"]).convert("RGB"))
                row = {"key": r["key"], "view": v, "det_2": "", "count_scores": "",
                       **scorer.score(img, prompts[r["prompt_id"]])}
                w.writerow(row)
                done[(r["key"], v)] = row
                n += 1
                if n % 100 == 0:
                    f.flush()
                    print(f"  {n} images re-scored", flush=True)
    m = {}
    for r in todo:
        m[(r["key"], "orig")] = r["m"]
        for v in views:
            m[(r["key"], v)] = margin(done[(r["key"], v)], prompts[r["prompt_id"]])
    out = ["## B. Stability: re-scoring after a mirror flip and a 90% center crop\n",
           "| images | verdict same under flip | same under crop | same under both |", "| --- | --- | --- | --- |"]
    for label, keys in [("all baselines", [b["key"] for b in base.values()]),
                        ("baselines within 0.1 of the cutoff", [b["key"] for b in base.values() if abs(b["m"]) < 0.1]),
                        ("flipped branches", [r["key"] for r in flips])]:
        if not keys:
            continue
        same = lambda k, v: (m[(k, v)] >= 0) == (m[(k, "orig")] >= 0)
        out.append(f"| {label} (n={len(keys)}) | {sum(same(k, 'flip') for k in keys) / len(keys):.0%} | "
                   f"{sum(same(k, 'crop') for k in keys) / len(keys):.0%} | "
                   f"{sum(same(k, 'flip') and same(k, 'crop') for k in keys) / len(keys):.0%} |")
    held = defaultdict(lambda: [0, 0])
    for r in flips:
        kind = "repairs" if r["m"] >= 0 else "harms"
        ok = all((m[(r["key"], v)] >= 0) != (m[(r["base_key"], v)] >= 0) for v in ["orig", *views])
        held[kind][0] += ok
        held[kind][1] += 1
    out.append("\n| outcome change | holds in all three views |\n| --- | --- |")
    out += [f"| {k} | {a} of {b} ({a / b:.0%}) |" for k, (a, b) in sorted(held.items())]
    return out + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    run, out_dir = Path(a.run), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = {p["id"]: p for p in json.load(open(a.prompts))}
    base, br = load(run, prompts)
    lines = [f"# Scorer check: {run}\n"] + part_a(base, br) + part_menus(base, br)
    if a.aug:
        lines += part_b(run, prompts, base, br, out_dir, a.device)
    (out_dir / "scorer_check.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
