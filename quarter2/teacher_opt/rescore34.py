"""Exploratory: rescore the outputs selected (by the old score) in rounds 31 and 32 with the revised error.
Frozen decisions are unchanged; this checks whether earlier small wins were contrast loss."""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import metrics34 as Q  # noqa: E402
O = "../../outputs"
img, cid, _ = Q.scene()
tg = {t: Q.target(img, t) for t in Q.TARGETS}
ld = lambda d, tag: np.asarray(Image.open(f"{O}/{d}/img/{tag}.png")).astype(np.float32)
c31 = json.load(open("results/r31_cases.json"))
m32 = json.load(open(f"{O}/q2_measured_steer/conds.json"))["rows"]
fin = lambda st, arm, k, t: next(r for r in m32 if r["stage"] == st and r["arm"] == arm and r["kind"] == "final" and r["cell"] == k and r["t"] == t)
out, r31, r32 = {}, [], {}
for t in Q.TARGETS:
    for k, p in Q.CASES:
        m = D.interior(cid, k)
        s = lambda d, tag: dict(new=Q.off_mae(ld(d, tag), tg[t], m), old=Q.aff_mae(ld(d, tag), tg[t], m),
                                contrast=Q.contrast(ld(d, tag), tg[t], m))
        c = c31[f"{p}@{t:g}"]
        a, b = s("q2_lora_scale", c["lora"]["tag"]), s("q2_lora_scale", c["request"]["tag"])
        r31.append((a["new"] / b["new"], a["old"] / b["old"], a["contrast"], b["contrast"]))
        rr = s("q2_measured_steer", fin("RR", "request", k, t)["tag"])
        for arm in ["mid_size", "mid_shape", "mid_random", "up0_size", "up0_shape", "up0_random"]:
            x = s("q2_measured_steer", fin("SR", arm, k, t)["tag"])
            r32.setdefault(arm, []).append((x["new"] / rr["new"], x["old"] / rr["old"]))
        out[f"{p}@{t:g}"] = dict(r31_lora=a, r31_request=b)
summ = dict(r31=dict(median_new=float(np.median([x[0] for x in r31])), wins_new=int(sum(x[0] < 1 for x in r31)),
                     median_old=float(np.median([x[1] for x in r31])), wins_old=int(sum(x[1] < 1 for x in r31)),
                     contrast_lora=[round(x[2], 3) for x in r31], contrast_request=[round(x[3], 3) for x in r31]),
            r32={a: dict(median_new=float(np.median([x[0] for x in v])), wins_new=int(sum(x[0] < 1 for x in v)),
                         median_old=float(np.median([x[1] for x in v]))) for a, v in r32.items()})
json.dump(dict(summary=summ, cases=out), open("results/rescore_r31_r32.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
