"""Turn evaluate.py's raw scores into pass/fail checks and the smoke-test tables.

    python analyze.py mix    --run RUN --prompts prompts.json            # baseline success/failure mix
    python analyze.py pick   --run DEV --prompts dev_prompts.json        # strengths for the main branches
    python analyze.py report --run RUN --prompts prompts.json [--dev DEV --dev-prompts dev_prompts.json]

Checks per image: object 1 and object 2 detected (OWLv2 score >= --det), each object's predicted color equals the
requested color, and the left-right order of the box centers matches the prompt. "all" = every check passes.
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
        o1, o2 = float(s["det_1"]) >= det, float(s["det_2"]) >= det
        left = p["relation"].startswith("object_1 left")
        rel = o1 and o2 and ((float(s["cx_1"]) < float(s["cx_2"])) == left)
        c = {"obj_1": o1, "obj_2": o2, "objects": o1 and o2,
             "color_1": o1 and s["color_pred_1"] == p["color_1"],
             "color_2": o2 and s["color_pred_2"] == p["color_2"], "relation": rel}
        c["all"] = c["objects"] and c["color_1"] and c["color_2"] and c["relation"]
        return {**meta, **c}
    base = [checks(json.loads(l)) for l in open(run / "baseline.jsonl") if json.loads(l)["key"] in scores]
    br = []
    if (run / "branches.jsonl").exists():
        br = [checks(json.loads(l)) for l in open(run / "branches.jsonl") if json.loads(l)["key"] in scores]
    return base, br


def rate(rows, k="all"):
    return sum(r[k] for r in rows) / len(rows) if rows else float("nan")


def mix_table(base):
    lines = ["| check | pass rate |", "| --- | --- |"]
    for k in ("obj_1", "obj_2", "objects", "color_1", "color_2", "relation", "all"):
        lines.append(f"| {k} | {rate(base, k):.1%} |")
    return "\n".join(lines)


def cmd_mix(a):
    base, _ = load(a.run, a.prompts, a.det)
    print(f"{len(base)} baseline images\n\n{mix_table(base)}")
    r = rate(base)
    if not 0.1 <= r <= 0.9 or rate(base, "objects") < 0.5:
        print(f"\nWARNING: success rate {r:.1%} or object rate {rate(base, 'objects'):.1%} is outside the "
              "useful range (10-90% success, >= 50% objects); branching results will say little.")


def outcomes(base, br):
    """Per branch: U = all(branch) - all(baseline) and its category."""
    b = {r["key"]: r for r in base}
    out = []
    for r in br:
        x = b[r["base_key"]]
        u = int(r["all"]) - int(x["all"])
        if u > 0:
            cat = "repair"
        elif u < 0:
            cat = "harm"
        elif not x["relation"] and r["relation"]:
            cat = "collateral"
        else:
            cat = "no change"
        out.append({**r, "U": u, "cat": cat, "base_all": x["all"]})
    return out


def cmd_pick(a):
    base, br = load(a.run, a.prompts, a.det)
    res = outcomes(base, br)
    groups = defaultdict(list)
    for r in res:
        groups[(r["action"], r["strength"])].append(r)
    picks = {}
    for (kind, s), rows in sorted(groups.items()):
        mean_u = sum(r["U"] for r in rows) / len(rows)
        size = sum(r["rel_size"] for r in rows) / len(rows)
        print(f"{kind}:{s:g}  n={len(rows)}  success {rate(rows):.1%}  mean U {mean_u:+.3f}  "
              f"repairs {sum(r['cat'] == 'repair' for r in rows)}  harms {sum(r['cat'] == 'harm' for r in rows)}  "
              f"pulse size {size:.3f}", file=sys.stderr)
        if kind != "none" and (kind not in picks or mean_u > picks[kind][0]):  # ties keep the weaker strength
            picks[kind] = (mean_u, s)
    print(",".join(f"{k}:{v[1]:g}" for k, v in sorted(picks.items())))


def policies(base, res):
    """Success per trajectory under every fixed policy: 'none' or (checkpoint, action)."""
    traj = {r["key"]: {"none": r["all"]} for r in base}
    for r in res:
        if r["action"] != "none":
            traj[r["base_key"]][(r["checkpoint"], f"{r['action']}:{r['strength']:g}")] = r["all"]
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


def cmd_report(a):
    base, br = load(a.run, a.prompts, a.det)
    res = outcomes(base, br)
    prompt_of = {r["key"]: r["prompt_id"] for r in base}
    out = [f"# Smoke test report\n\nDetection threshold {a.det}. {len(base)} baseline trajectories, "
           f"{len(res)} branches.\n", "## Baseline checks\n", mix_table(base), ""]

    out += ["## Each action at each checkpoint (vs. no action)\n",
            "Rates are over all trajectories; 'repair'/'harm' change the all-requirements outcome; 'collateral' "
            "fixes the relation but loses an object or color.\n",
            "| checkpoint | action | success | repair | harm | collateral | no change | pulse size | U-Net evals |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| – | none (baseline) | {rate(base):.1%} | | | | | | {base[0]['unet_evals'] if base else ''} |"]
    groups = defaultdict(list)
    for r in res:
        groups[(r["checkpoint"], r["action"], r["strength"])].append(r)
    for (k, kind, s), rows in sorted(groups.items()):
        cnt = lambda c: sum(r["cat"] == c for r in rows) / len(rows)
        size = sum(r["rel_size"] for r in rows) / len(rows)
        out.append(f"| {k} | {kind}:{s:g} | {rate(rows):.1%} | {cnt('repair'):.1%} | {cnt('harm'):.1%} | "
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
    a = ap.parse_args()
    {"mix": cmd_mix, "pick": cmd_pick, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
