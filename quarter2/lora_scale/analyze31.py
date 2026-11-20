"""Round 31 analysis: frozen decision for per-block LoRA scaling vs request-only correction.

Per case and method, candidates are kept only if
  - delivered blur within tolerance: |r_hat - target| <= max(2 px, 20 %), and
  - focused-region MAE vs the source added over the baseline <= 0.5/255;
the selected output is the valid candidate with the lowest primary error (round 29's reference-image error).
Methods: request-only (the request search, incl. the baseline request); LoRA (scan + retuned outputs, every one
with some block scaled; unscaled outputs are excluded, so a win needs the LoRA change).
Unattainable (no valid candidate) counts as a failure: LoRA unattainable -> loss (ratio inf); request-only
unattainable and LoRA attained -> win (ratio 0).
PASS: median ratio (LoRA / request-only) <= 0.80 over the 8 cases AND LoRA wins >= 6 of 8.
usage: python analyze31.py <R31_OUT> <results dir>
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D  # noqa: E402
import measure18 as m18  # noqa: E402
import metrics29 as Q  # noqa: E402

RATIO_MAX, WINS_MIN = 0.80, 6
out, res = sys.argv[1], sys.argv[2]
os.makedirs(res, exist_ok=True)
meta = json.load(open(f"{out}/conds.json"))
rows = meta["rows"]
img, cid, _ = Q.scene()
im = img.astype(np.float32)
tg = {t: Q.target(img, t) for t in Q.TARGETS}
ld = lambda tag: np.asarray(Image.open(f"{out}/img/{tag}.png")).astype(np.float32)
valid = lambda d, t: abs(d["r_hat"] - t) <= m18.TOL(t) and d["focus_add"] <= Q.FOCUS_ADD_MAX

cases, ratios, wins = {}, [], 0
for k, p in Q.CASES:
    m = D.interior(cid, k)
    for t in Q.TARGETS:
        key = f"{p}@{t:g}"
        mine = [r for r in rows if r["cell"] == k and r["target"] == t]
        base = next(r for r in mine if r["stage"] == "base")
        meth = {"request": [r for r in mine if r["stage"] == "input"],
                "lora": [r for r in mine if r["stage"] in ("scan", "retune")]}
        c = dict(baseline=dict(err=base["err"], r_hat=base["r_hat"], valid=valid(base, t)))
        for name, cand in meth.items():
            ok = [r for r in cand if valid(r, t)]
            if not ok:
                c[name] = dict(valid=False, n=len(cand), n_valid=0)
                continue
            x = min(ok, key=lambda r: r["err"])
            y = ld(x["tag"])
            ideal = np.clip(m18.blur(im, m18.disk(x["r_hat"])), 0, 255)
            c[name] = dict(valid=True, n=len(cand), n_valid=len(ok), err=x["err"], r_hat=x["r_hat"], tag=x["tag"],
                           block=x["block"], scale=x["scale"], factor=x["factor"], stage=x["stage"],
                           focus_add=x["focus_add"], others_vs_base=x["others_vs_base"],
                           err_size=Q.aff_mae(ideal, tg[t], m), err_shape=Q.aff_mae(y, ideal, m))
        a, b = c["lora"], c["request"]
        if a["valid"] and b["valid"]:
            ratio = a["err"] / b["err"]
        elif a["valid"]:
            ratio = 0.0
        else:
            ratio = float("inf")
        c["ratio"], c["win"] = ratio, ratio < 1
        ratios.append(ratio)
        wins += c["win"]
        cases[key] = c

med = float(np.median(ratios))
passed = med <= RATIO_MAX and wins >= WINS_MIN

# exploratory: each single (block, scale) applied to every case, best valid output of that setting per case
single = {}
for b in meta["blocks"]:
    for s in meta["scales"]:
        w, rs = 0, []
        for k, p in Q.CASES:
            for t in Q.TARGETS:
                ok = [r for r in rows if r["cell"] == k and r["target"] == t and r["block"] == b and r["scale"] == s
                      and r["stage"] in ("scan", "retune") and valid(r, t)]
                bq = cases[f"{p}@{t:g}"]["request"]
                if ok and bq["valid"]:
                    q = min(r["err"] for r in ok) / bq["err"]
                    rs.append(q)
                    w += q < 1
        single[f"{b}x{s:g}"] = dict(cases_attained=len(rs), wins=w, median_ratio=float(np.median(rs)) if rs else None)

summary = dict(decision="PASS" if passed else "FAIL", median_ratio=med, wins=int(wins), n_cases=len(cases),
               rule=dict(ratio_max=RATIO_MAX, wins_min=WINS_MIN), gate=meta["gate"], lora_layers=meta["lora_layers"],
               n_generations=meta["n_generations"], promising=meta["promising"],
               attained=dict(request=sum(c["request"]["valid"] for c in cases.values()),
                             lora=sum(c["lora"]["valid"] for c in cases.values()),
                             baseline=sum(c["baseline"]["valid"] for c in cases.values())),
               single_setting_exploratory=single)
json.dump(summary, open(f"{res}/lora_scale.json", "w"), indent=1)
json.dump(cases, open(f"{res}/lora_scale_cases.json", "w"), indent=1)
print(json.dumps(dict((k, v) for k, v in summary.items() if k != "single_setting_exploratory"), indent=1))
for key, c in cases.items():
    f = lambda d: (f"err {d['err']:.2f} r {d['r_hat']:.1f} [{d.get('block')}x{d.get('scale')}, f{d.get('factor')}]"
                   if d.get("valid", True) else "UNATTAINABLE")
    print(f"{key:16s} base err {c['baseline']['err']:.2f} r {c['baseline']['r_hat']:.1f} | request {f(c['request'])} | "
          f"lora {f(c['lora'])} | ratio {c['ratio']:.2f}")

# crop sheet at 11.625: input | target | request-only best | LoRA best   (4 x 4 = 16 tiles)
t = 11.625
T, tiles = 160, []
for k, p in Q.CASES:
    ys, xs = np.where(D.interior(cid, k))
    cy, cx = int(ys.mean()), int(xs.mean())
    sl = (slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))
    c = cases[f"{p}@{t:g}"]
    row = [(im, "input"), (tg[t], "target")]
    for name in ("request", "lora"):
        row.append((ld(c[name]["tag"]), name) if c[name]["valid"] else (np.zeros_like(im), name + ": none"))
    tiles.append([(a[sl], f"{p} {lab}") for a, lab in row])
sheet = Image.new("RGB", (4 * T, 4 * (T + 16)), "white")
dr = ImageDraw.Draw(sheet)
for i, row in enumerate(tiles):
    for j, (a, lab) in enumerate(row):
        sheet.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((T, T), Image.NEAREST), (j * T, i * (T + 16) + 16))
        dr.text((j * T + 3, i * (T + 16) + 2), lab, fill="black")
sheet.save(f"{res}/crops_t11.6.png")
