"""Turn evaluate.py's raw scores into pass/fail checks and the smoke-test tables.

    python analyze.py mix    --run RUN --prompts prompts.json            # baseline success/failure mix
    python analyze.py pick   --run DEV --prompts dev_prompts.json        # strengths for the main branches
    python analyze.py report --run RUN --prompts prompts.json [--dev DEV --dev-prompts dev_prompts.json]

Checks per image depend on the prompt's "task":
- relation (default): both objects detected (OWLv2 score >= --det), each object's predicted color equals the
  requested color, and the left-right order of the box centers matches the prompt.
- two_object: both objects detected.
- counting: object detected, and the number of non-overlapping boxes scoring >= --det equals "count".
"all" = every check passes.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(run, prompts_path, det):
    run = Path(run)
    prompts = {p["id"]: p for p in json.load(open(prompts_path))}
    scores = {r["key"]: r for r in csv.DictReader(open(run / "scores.csv"))}
    def checks(meta):
        s, p = scores[meta["key"]], prompts[meta["prompt_id"]]
        task = p.get("task", "relation")
        o1 = float(s["det_1"]) >= det
        if task == "counting":
            n = sum(float(v) >= det for v in s["count_scores"].split())
            c = {"obj_1": o1, "count": n == p["count"], "under": n < p["count"], "over": n > p["count"]}
            c["all"] = c["count"]
            return {**meta, **c}
        o2 = float(s["det_2"]) >= det
        if task == "two_object":
            return {**meta, "obj_1": o1, "obj_2": o2, "all": o1 and o2}
        left = p["relation"].startswith("object_1 left")
        rel = o1 and o2 and ((float(s["cx_1"]) < float(s["cx_2"])) == left)
        c = {"obj_1": o1, "obj_2": o2, "objects": o1 and o2,
             "color_1": o1 and s["color_pred_1"] == p["color_1"],
             "color_2": o2 and s["color_pred_2"] == p["color_2"], "relation": rel}
        c["all"] = c["objects"] and c["color_1"] and c["color_2"] and c["relation"]
        return {**meta, **c}
    use = lambda r: r["key"] in scores and r["prompt_id"] in prompts  # a smaller prompt file selects a subset
    base = [checks(r) for r in map(json.loads, open(run / "baseline.jsonl")) if use(r)]
    br = []
    if (run / "branches.jsonl").exists():
        br = [checks(r) for r in map(json.loads, open(run / "branches.jsonl")) if use(r)]
    return base, br


def rate(rows, k="all"):
    return sum(r[k] for r in rows) / len(rows) if rows else float("nan")


META = {"key", "base_key", "prompt_id", "seed", "checkpoint", "action", "strength", "pulse_steps", "rel_size",
        "unet_evals", "image", "seconds"}


def mix_table(base):
    lines = ["| check | pass rate |", "| --- | --- |"]
    for k in [k for k in base[0] if k not in META] if base else []:
        lines.append(f"| {k} | {rate(base, k):.1%} |")
    return "\n".join(lines)


def cmd_mix(a):
    base, _ = load(a.run, a.prompts, a.det)
    print(f"{len(base)} baseline images\n\n{mix_table(base)}")
    r = rate(base)
    if not 0.1 <= r <= 0.9:
        print(f"\nWARNING: success rate {r:.1%} is outside the useful range (10-90%); branching results will "
              "say little.")
        if a.strict:
            sys.exit(3)


def outcomes(base, br):
    """Per branch: U = all(branch) - all(baseline) and its category."""
    b = {r["key"]: r for r in base}
    out = []
    for r in br:
        x = b[r["base_key"]]
        u = int(r["all"]) - int(x["all"])
        checks = [k for k in x if k not in META and k not in ("all", "under", "over")]  # under/over are not checks
        if u > 0:
            cat = "repair"
        elif u < 0:
            cat = "harm"
        elif any(r[k] and not x[k] for k in checks) and any(x[k] and not r[k] for k in checks):
            cat = "collateral"  # one check fixed, another lost, overall outcome unchanged
        else:
            cat = "no change"
        out.append({**r, "U": u, "cat": cat, "base_all": x["all"]})
    return out


def name(r):
    return f"{r['action']}:{r['strength']:g}" + (f":{r.get('pulse_steps', 3)}" if r.get("pulse_steps", 3) != 3 else "")


def cmd_pick(a):
    base, br = load(a.run, a.prompts, a.det)
    res = outcomes(base, br)
    groups = defaultdict(list)
    for r in res:
        groups[(r["action"], r["strength"], r.get("pulse_steps", 3))].append(r)
    picks = {}
    for (kind, s, n), rows in sorted(groups.items()):  # weaker and shorter first
        mean_u = sum(r["U"] for r in rows) / len(rows)
        size = sum(r["rel_size"] for r in rows) / len(rows)
        print(f"{name(rows[0])}  n={len(rows)}  success {rate(rows):.1%}  mean U {mean_u:+.3f}  "
              f"repairs {sum(r['cat'] == 'repair' for r in rows)}  harms {sum(r['cat'] == 'harm' for r in rows)}  "
              f"pulse size {size:.3f}", file=sys.stderr)
        if kind != "none" and (kind not in picks or mean_u > picks[kind][0]):  # ties keep the weaker one
            picks[kind] = (mean_u, name(rows[0]))
    print(",".join(v[1] for _, v in sorted(picks.items())))


def policies(base, res):
    """Success per trajectory under every fixed policy: 'none' or (checkpoint, action)."""
    traj = {r["key"]: {"none": r["all"]} for r in base}
    for r in res:
        if r["action"] != "none":
            traj[r["base_key"]][(r["checkpoint"], name(r))] = r["all"]
    return traj


def summarize(traj, prompt_of, keys=None):
    keys = keys if keys is not None else list(traj)
    pols = sorted({p for t in traj.values() for p in t}, key=str)
    fixed = {p: sum(traj[k][p] for k in keys) / len(keys) for p in pols}
    best = max(pols, key=lambda p: (fixed[p], p == "none"))  # ties go to doing nothing
    oracle = sum(max(traj[k].values()) for k in keys) / len(keys)
    return fixed, best, oracle


def bootstrap(traj, prompt_of, n=2000, seed=0):
    """Resample prompts (not branches) to get a 95% interval for oracle minus best fixed policy."""
    by_prompt = defaultdict(list)
    for k in traj:
        by_prompt[prompt_of[k]].append(k)
    ps, rng, gaps = sorted(by_prompt), random.Random(seed), []
    for _ in range(n):
        keys = [k for p in rng.choices(ps, k=len(ps)) for k in by_prompt[p]]
        fixed, best, oracle = summarize(traj, prompt_of, keys)
        gaps.append(oracle - fixed[best])
    gaps.sort()
    return gaps[int(0.025 * n)], gaps[int(0.975 * n)]


def menu_section(run, base, res, prompt_of):
    """Hindsight oracle over the nudges picked on dev vs. control menus built from the same saved states: placebo
    (random direction, same size), wrong sign (other non-placebo actions, e.g. targeted:-15, global:1), and
    re-rolling the seed with about the same number of extra U-Net calls (from other baseline seeds)."""
    if not any(r["action"] == "placebo" for r in res):
        return []
    import numpy as np
    from PIL import Image
    traj = policies(base, res)
    act = Path(run) / "actions.txt"
    picked = set(act.read_text().strip().split(",")) if act.exists() else None
    is_real = lambda n: not n.startswith("placebo") and (picked is None or n in picked)
    menus = {"nudges picked on dev": lambda p: p == "none" or is_real(p[1]),
             "placebo (random direction)": lambda p: p == "none" or p[1].startswith("placebo"),
             "wrong sign": lambda p: p == "none" or not (is_real(p[1]) or p[1].startswith("placebo"))}
    sizes = {m: len({p for t in traj.values() for p in t if p != "none" and f(p)}) for m, f in menus.items()}
    menus = {m: f for m, f in menus.items() if sizes[m]}
    # extra U-Net calls per trajectory for the picked menu, and a re-seed menu of about the same cost
    cost = defaultdict(float)
    for r in res:
        if r["action"] != "none" and is_real(name(r)):
            cost[r["base_key"]] += r["unet_evals"]
    extra = sum(cost.values()) / max(len(cost), 1)
    n_seeds = max(1, round(extra / 60))
    ok = {r["key"]: r["all"] for r in base}
    seeds_of = defaultdict(list)
    for r in base:
        seeds_of[r["prompt_id"]].append((r["seed"], r["key"]))
    def reseed(k):
        ss = sorted(seeds_of[prompt_of[k]])
        i = [key for _, key in ss].index(k)
        return max(ok[ss[(i + j) % len(ss)][1]] for j in range(n_seeds + 1))
    def oracle(keys, keep):
        if keep == "reseed":
            return sum(reseed(k) for k in keys) / len(keys)
        return sum(max(v for p, v in traj[k].items() if keep(p)) for k in keys) / len(keys)
    keys = list(traj)
    by_prompt = defaultdict(list)
    for k in keys:
        by_prompt[prompt_of[k]].append(k)
    ref = "nudges picked on dev"
    others = [m for m in menus if m != ref] + ["reseed"]
    ps, rng, diffs = sorted(by_prompt), random.Random(0), defaultdict(list)
    for _ in range(2000):
        ks = [k for p in rng.choices(ps, k=len(ps)) for k in by_prompt[p]]
        o = oracle(ks, menus[ref])
        for m in others:
            diffs[m].append(o - oracle(ks, menus.get(m, m)))
    for v in diffs.values():
        v.sort()
    bimg = {b["key"]: b["image"] for b in base}
    im = lambda p: np.asarray(Image.open(Path(run) / p).convert("L").resize((64, 64)), float)
    change = defaultdict(list)
    for r in res:  # skipped when the images are not on this machine (they are kept on DSMLP)
        if r["action"] != "none" and (Path(run) / r["image"]).exists():
            change[name(r)].append(np.abs(im(r["image"]) - im(bimg[r["base_key"]])).mean())
    label = lambda m: f"re-roll the seed, best of {n_seeds + 1} seeds" if m == "reseed" else m
    o_ref = oracle(keys, menus[ref])
    out = ["## Picked nudges vs. control menus (hindsight oracle, same saved states)\n",
           f"The picked nudges cost {extra:.0f} extra U-Net calls per image in total; re-rolling uses {n_seeds} other "
           f"seeds of the same prompt ({60 * n_seeds} calls).\n",
           "| menu | options per image | hindsight oracle | picked minus this menu (95% interval over prompts) |",
           "| --- | --- | --- | --- |",
           f"| no intervention | 0 | {oracle(keys, lambda p: p == 'none'):.1%} | |",
           f"| {ref} | {sizes[ref]} | {o_ref:.1%} | |"]
    for m in others:
        d = diffs[m]
        n_opt = n_seeds if m == "reseed" else sizes[m]
        out.append(f"| {label(m)} | {n_opt} | {oracle(keys, menus.get(m, m)):.1%} | "
                   f"**{100 * (o_ref - oracle(keys, menus.get(m, m))):+.1f}** ({100 * d[50]:+.1f} to {100 * d[1949]:+.1f}) |")
    out += ["", "Near zero against placebo means the per-image gain is what any re-roll of the image would give. "
            "Against wrong sign, near zero means the direction of the push does not matter.\n",
            "| action | mean image change (0-255 per pixel, vs. no action) |", "| --- | --- |"]
    out += [f"| {k} | {sum(v) / len(v):.1f} |" for k, v in sorted(change.items())]
    if not change:
        out[-2:] = ["(Image-change table skipped: images not on this machine.)"]
    return out + [""]


def cmd_report(a):
    base, br = load(a.run, a.prompts, a.det)
    res = outcomes(base, br)
    prompt_of = {r["key"]: r["prompt_id"] for r in base}
    out = [f"# Smoke test report\n\nDetection threshold {a.det}. {len(base)} baseline trajectories, "
           f"{len(res)} branches.\n", "## Baseline checks\n", mix_table(base), ""]

    out += ["## Each action at each checkpoint (vs. no action)\n",
            "Rates are over all trajectories; 'repair'/'harm' change the all-requirements outcome; 'collateral' "
            "fixes one check but loses another, leaving the outcome unchanged.\n",
            "| checkpoint | action | success | repair | harm | collateral | no change | pulse size | U-Net evals |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| – | none (baseline) | {rate(base):.1%} | | | | | | {base[0]['unet_evals'] if base else ''} |"]
    groups = defaultdict(list)
    for r in res:
        groups[(r["checkpoint"], r["action"], r["strength"], r.get("pulse_steps", 3))].append(r)
    for (k, kind, s, _), rows in sorted(groups.items()):
        cnt = lambda c: sum(r["cat"] == c for r in rows) / len(rows)
        size = sum(r["rel_size"] for r in rows) / len(rows)
        out.append(f"| {k} | {name(rows[0])} | {rate(rows):.1%} | {cnt('repair'):.1%} | {cnt('harm'):.1%} | "
                   f"{cnt('collateral'):.1%} | {cnt('no change'):.1%} | {size:.3f} | {rows[0]['unet_evals']} |")

    traj = policies(base, res)
    fixed, best, oracle = summarize(traj, prompt_of)
    lo, hi = bootstrap(traj, prompt_of)
    out += ["", "## Oracle headroom\n",
            "| policy | success |", "| --- | --- |",
            f"| no intervention | {fixed['none']:.1%} |",
            f"| best fixed policy, chosen on these same trajectories (optimistic) | {fixed[best]:.1%} ({best}) |"]
    if a.dev:
        dbase, dbr = load(a.dev, a.dev_prompts, a.det)
        dtraj = policies(dbase, outcomes(dbase, dbr))
        dfixed, dbest, _ = summarize(dtraj, {r["key"]: r["prompt_id"] for r in dbase})
        if dbest in fixed:
            out.append(f"| best fixed policy chosen on dev prompts | {fixed[dbest]:.1%} ({dbest}) |")
    out += [f"| per-sample hindsight oracle | {oracle:.1%} |", "",
            f"**Oracle minus best fixed policy: {100 * (oracle - fixed[best]):.1f} points** "
            f"(95% interval over prompts {100 * lo:.1f} to {100 * hi:.1f}).", ""]

    out += menu_section(a.run, base, res, prompt_of)

    # Where the oracle's gain comes from: whether, when, or which action.
    if not res:
        print("\n".join(out))
        return
    acts = sorted({p[1] for t in traj.values() for p in t if p != "none"})
    ks = sorted({p[0] for t in traj.values() for p in t if p != "none"})
    n = len(traj)
    whether = sum(max(t["none"], t.get(best, t["none"])) for t in traj.values()) / n if best != "none" else fixed["none"]
    when = max(sum(max([t["none"]] + [t.get((k, a_), 0) for k in ks]) for t in traj.values()) / n for a_ in acts)
    which = max(sum(max([t["none"]] + [t.get((k, a_), 0) for a_ in acts]) for t in traj.values()) / n for k in ks)
    out += ["## Where the oracle gain comes from\n", "| per-sample choice | success |", "| --- | --- |",
            f"| whether to apply the best fixed action | {whether:.1%} |",
            f"| when (one action, choose checkpoint) | {when:.1%} |",
            f"| which action (one checkpoint, choose action) | {which:.1%} |",
            f"| everything (oracle) | {oracle:.1%} |"]
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mix", "pick", "report"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--dev")
    ap.add_argument("--dev-prompts")
    ap.add_argument("--det", type=float, default=0.2, help="OWLv2 score needed to count an object as present")
    ap.add_argument("--strict", action="store_true", help="mix: exit with code 3 if success is outside 10-90%%")
    a = ap.parse_args()
    {"mix": cmd_mix, "pick": cmd_pick, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
