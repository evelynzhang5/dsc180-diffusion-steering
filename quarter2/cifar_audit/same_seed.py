"""Same-seed check of the round-5 side effects (analysis of saved measures; makes no images).

The pre-registered comparison (steered images of class c vs. the model's own class-c images from other seeds) is a
comparison of two image sets: part of a difference can come from the steered set being more recognisable. Here the
change is measured on the same seed instead:

  KEPT  seeds whose unguided image is already class c and whose steered image is still class c: the class does not
        change, so any meter shift is a change beyond the class. Paired effect = mean(steered - unguided).
  ALL   every seed whose steered image is class c, vs. the unguided image on that seed (includes the class change).
  RANDOM the random direction vs. unguided on the KEPT seeds (generic perturbation).

Units and rule as in the pre-registration: SDs of the meter over 1,000 real class-c images; an effect is flagged if
|effect| > 0.25 and its paired 95% bootstrap interval excludes 0. Image meters only (the set meters' checks failed or
were inconclusive). Run from the repo root after the test stage:
    python quarter2/cifar_audit/same_seed.py     # reads outputs/q2_cifar_audit/test_measures.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import cifar_audit as A  # noqa: E402

OUT = ROOT / "outputs" / "q2_cifar_audit"
RES = Path(__file__).parent / "results"


def main():
    d = np.load(OUT / "test_measures.npz", allow_pickle=True)["data"].item()
    truth = json.load(open(RES / "truth.json"))["truth"]
    refs = A.real_refs()
    rng = np.random.default_rng(0)
    ung = d["unguided"]
    res = {}
    L = ["# Same-seed check of the side effects (test seeds 3000-3511)", "",
         "KEPT: seeds whose unguided image is already the class and whose steered image still is, so the class does "
         "not change; effect = steered - unguided on the same seed, in SDs of real class images. ALL: every seed "
         "whose steered image is the class (includes the class change). SETS: the pre-registered steered vs. native "
         "comparison. Bold: |effect| > 0.25 and the 95% interval excludes 0. Image meters only.", ""]
    for k in A.AUDIT_CLASSES:
        name = A.R.CLASSES[k]
        sd = {m: v.std() for m, v in A.image_meters(refs[k]).items()}
        st, rd = d[f"steered_{k}"], d[f"random_{k}"]
        kept = np.flatnonzero((ung["pred"] == k) & (st["pred"] == k))
        alls = np.flatnonzero(st["pred"] == k)
        r = {"n_kept": int(len(kept)), "n_all": int(len(alls)), "KEPT": {}, "ALL": {}, "RANDOM": {}}
        for m in A.IMAGE_METERS:
            r["KEPT"][m] = A.effect(st["m"][m][kept], ung["m"][m][kept], sd[m], rng, paired=True)
            r["ALL"][m] = A.effect(st["m"][m][alls], ung["m"][m][alls], sd[m], rng, paired=True)
            r["RANDOM"][m] = A.effect(rd["m"][m][kept], ung["m"][m][kept], sd[m], rng, paired=True)
        res[name] = r
        sets = truth[name]["effects"]
        conf = set(truth[name]["confirmed"])

        def cell(e):
            s = f"{e[0]:+.2f} ({e[1]:+.2f}, {e[2]:+.2f})"
            return f"**{s}**" if abs(e[0]) > A.TAU and (e[1] > 0 or e[2] < 0) else s
        L += [f"## {name}: {len(kept)} seeds keep the class, {len(alls)} steered images are the class", "",
              "| meter | SETS (pre-registered) | KEPT | ALL | RANDOM on KEPT seeds |", "|---|---|---|---|---|"]
        for m in A.IMAGE_METERS:
            L.append(f"| {m} | {cell(sets[m])}{' (confirmed)' if m in conf else ''} | {cell(r['KEPT'][m])} | "
                     f"{cell(r['ALL'][m])} | {cell(r['RANDOM'][m])} |")
        L.append("")
    # summary: do the confirmed image-meter effects hold on the same seed?
    L += ["## Summary: confirmed image-meter effects (SETS) on the same seed (KEPT)", "",
          "| class | meter | SETS | KEPT | same sign | KEPT flagged |", "|---|---|---|---|---|---|"]
    n = same = flag = 0
    for name, r in res.items():
        for m in truth[name]["confirmed"]:
            if m not in A.IMAGE_METERS:
                continue
            s, kp = truth[name]["effects"][m], r["KEPT"][m]
            sg = np.sign(s[0]) == np.sign(kp[0])
            fl = abs(kp[0]) > A.TAU and (kp[1] > 0 or kp[2] < 0)
            n += 1; same += sg; flag += fl
            L.append(f"| {name} | {m} | {s[0]:+.2f} | {kp[0]:+.2f} ({kp[1]:+.2f}, {kp[2]:+.2f}) | "
                     f"{'yes' if sg else 'no'} | {'yes' if fl else 'no'} |")
    L += ["", f"{same} of {n} confirmed image-meter effects have the same sign on the same seed; {flag} of {n} are "
          "also flagged there."]
    # decomposition: steered - native = (steered - unguided, same seeds) + (unguided on those seeds - native)
    L += ["", "## Decomposition of the pre-registered effect (point estimates, SDs of real class images)", "",
          "SETS = STEERING + SEED, exactly: STEERING is steered - unguided on the seeds whose steered image is the "
          "class (ALL above); SEED is what those seeds make without steering minus the model's own class images, "
          "i.e. what the steered images inherit from the image the seed would have been.", "",
          "| class | meter | SETS | STEERING | SEED |", "|---|---|---|---|---|"]
    for k in A.AUDIT_CLASSES:
        name = A.R.CLASSES[k]
        sd = {m: v.std() for m, v in A.image_meters(refs[k]).items()}
        st, pool = d[f"steered_{k}"], d["native_pool"]
        alls, nat = np.flatnonzero(st["pred"] == k), np.flatnonzero(pool["pred"] == k)
        res[name]["SEED"] = {}
        for m in truth[name]["confirmed"]:
            if m not in A.IMAGE_METERS:
                continue
            tot = (st["m"][m][alls].mean() - pool["m"][m][nat].mean()) / sd[m]
            steer = (st["m"][m][alls].mean() - ung["m"][m][alls].mean()) / sd[m]
            res[name]["SEED"][m] = float(tot - steer)
            L.append(f"| {name} | {m} | {tot:+.2f} | {steer:+.2f} | {tot - steer:+.2f} |")
    (RES / "same_seed.md").write_text("\n".join(L) + "\n")
    (RES / "same_seed.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
