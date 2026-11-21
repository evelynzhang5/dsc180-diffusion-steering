"""Round 38 D2c (post hoc, CPU): rescore the round 29-34 comparisons against a linear-light disk reference.

Scene and cases: round 29's (gravel_0, astronaut_0, chelsea_0, coins_0 at t = 8 and 11.625). Revised error (offset-only
MAE) against (a) the sRGB disk blur at t (rounds 29-37's reference) and (b) a linear-light disk blur at t
(sRGB -> linear -> disk -> sRGB). Validity (radius tolerance, focus guard) is taken from the saved rows (sRGB radius fit;
radius shifts between the references are <= ~1 px, D2b).
  baseline      round 25's calibrated request (round 34's base image)
  request-only  best valid image of round 31's request search, selected separately under each reference
  LoRA          best valid image of round 31's per-block LoRA scan/retune, selected under each reference
  teacher       round 34's teacher (input already sRGB-blurred to t): correct by construction under (a) only
usage: python rescore_linear.py <outputs dir> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure18 as m18  # noqa: E402
import metrics34 as Q  # noqa: E402
import design as D  # noqa: E402

OUTS, RES = sys.argv[1], sys.argv[2]
to_lin = lambda s: np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
to_srgb = lambda l: np.where(l <= 0.0031308, 12.92 * l, 1.055 * np.clip(l, 0, None) ** (1 / 2.4) - 0.055)
img, cid, _ = Q.scene()
src = img.astype(np.float32)
ints = {k: D.interior(cid, k) for k, _ in Q.CASES}
REF = {t: dict(srgb=Q.target(img, t),
               linear=np.clip(to_srgb(m18.blur(to_lin(src / 255.0), m18.disk(t))), 0, 1) * 255) for t in Q.TARGETS}
L = lambda p: np.asarray(Image.open(p)).astype(np.float32)
rows = json.load(open(f"{OUTS}/q2_lora_scale/conds.json"))["rows"]
cache = {}


def err(path, t, k, ref):
    if path not in cache:
        cache[path] = L(path)
    return Q.off_mae(cache[path], REF[t][ref], ints[k])


out = {}
for k, p in Q.CASES:
    for t in Q.TARGETS:
        key = f"{p}@{t:g}"
        base = f"{OUTS}/q2_teacher_opt/img/base_t{t:.1f}.png"
        teach = f"{OUTS}/q2_teacher_opt/img/teacher_t{t:.1f}.png"
        ok = lambda r: abs(r["r_hat"] - t) <= m18.TOL(t) and r["focus_add"] <= Q.FOCUS_ADD_MAX
        req = [r for r in rows if r["cell"] == k and r["target"] == t and r["stage"] in ("input", "base") and ok(r)]
        lora = [r for r in rows if r["cell"] == k and r["target"] == t and r["stage"] in ("scan", "retune") and ok(r)]
        o = {}
        for ref in ("srgb", "linear"):
            b = err(base, t, k, ref)
            rq = min((err(f"{OUTS}/q2_lora_scale/img/{r['tag']}.png", t, k, ref), r["tag"]) for r in req)
            lo = min((err(f"{OUTS}/q2_lora_scale/img/{r['tag']}.png", t, k, ref), r["tag"]) for r in lora) if lora else (None, None)
            o[ref] = dict(baseline=b, request=rq[0], request_tag=rq[1], lora=lo[0], teacher=err(teach, t, k, ref))
        out[key] = o
json.dump(out, open(f"{RES}/rescore_linear.json", "w"), indent=1)

print(f"{'case':20s} | {'sRGB: base  req  LoRA  teach':30s} | linear: base  req  LoRA  teach")
for key, o in out.items():
    s, l = o["srgb"], o["linear"]
    print(f"{key:20s} | {s['baseline']:5.2f} {s['request']:5.2f} {s['lora']:5.2f} {s['teacher']:5.2f}            | "
          f"{l['baseline']:5.2f} {l['request']:5.2f} {l['lora']:5.2f} {l['teacher']:5.2f}")
for ref in ("srgb", "linear"):
    g = lambda f: np.array([o[ref][f] for o in out.values()])
    print(f"{ref:6s}: median base {np.median(g('baseline')):.2f} request {np.median(g('request')):.2f} "
          f"(request/base {np.median(g('request') / g('baseline')):.2f}) LoRA/request {np.median(g('lora') / g('request')):.2f} "
          f"wins {int((g('lora') < g('request')).sum())}/8 | teacher/request {np.median(g('teacher') / g('request')):.2f} "
          f"| spread of request error across cases {g('request').min():.2f}-{g('request').max():.2f}")
