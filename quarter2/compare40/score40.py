"""Round 40 stage 3 (CPU): score the test phase. Frozen before the test phase runs.

Per generated test scene x method x target (outputs at the frozen settings in mapping.json):
  blur       delivered effective additional blur on BEHIND tiles (blur40); front tiles give foreground defocus,
             reported separately and never mixed in.
  control    delivered - target; success = |.| <= TOL(target). Unmeasurable outputs are counted, not dropped.
  matched    both methods' delivered blur measurable and within TOL of each other (TOL of their mean): only
             matched scene x target cells enter the rendering-quality comparison; the number excluded is reported.
  preservation (measures40.preservation vs the ORIGINAL input, subject interior >= 8 px): colour_shift, detail,
             hf_ratio, noise.
  boundary   measures40.boundary with depth gating (stored map and regions focus): halo_sharp, thin_loss (passed
             the specified synthetic controls); rim_inner, rim, bleed exploratory.
  focus      subject sharper than background, relative to the input: interior hf_ratio > median behind-tile HF
             retention.
Scene aggregation: a scene's value is the mean over its targets (matched cells only where stated); summaries are
medians over scenes and paired per-scene differences (BokehDiff - Bokehlicious) with sign counts. Every count is
reported per metric, split, method and category; existing_blur is reported separately from initially-sharp scenes.
Review rules (review40.json): mask failures out of every mask-dependent metric; measurement flag out of blur, control
and halo; mask cautions and metric cautions reported with and without.
RealBokeh test: linear-light mean |output - real capture| (raw, and after a per-channel gain fit for exposure), at the
f/2.0 and second-aperture settings frozen on the 3 development scenes; the input's own error is the baseline.
Latency: rendering-only per method; total for BokehDiff adds depth + segmentation (lat40.json).
Checks (printed, and the run fails if any is violated): exclusions applied; front tiles not used for blur; test
settings equal the frozen mapping.
Blinded pairs: 25 matched cells (initially-sharp test scenes, no mask failure), seed 0, sides shuffled; key kept apart.
usage: python score40.py <data root> <stage 3 out root> <results dir>
"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blur40 as BL  # noqa: E402
import common40 as C  # noqa: E402
import measures40 as MS  # noqa: E402

DATA, OUT, RES = sys.argv[1:4]
os.makedirs(RES, exist_ok=True)
scenes, regions, review = C.load(DATA)
mp = json.load(open(f"{OUT}/mapping.json"))
runs = {"bokehdiff": json.load(open(f"{OUT}/bd_test.json"))["rows"],
        "bokehlicious": json.load(open(f"{OUT}/bkl_test.json"))["rows"]}
KEY = {"bokehdiff": "K", "bokehlicious": "av"}
MF, FLAG = set(review["mask_failure"]), set(review["measurement_flag"])
CAUT = set(review["mask_caution"]) | set(review["metric_caution"])
checks = {}

# ---- check: frozen settings reused unchanged
viol = []
for meth, rows in runs.items():
    for r in rows:
        tag = r["file"].split("/")[-1][:-4].split("_")[-1]
        want = mp[meth]["gen"][tag[1:]]["setting"] if tag.startswith("t") else mp[meth]["rb"][tag]["setting"]
        if abs(r[KEY[meth]] - want) > 1e-9:
            viol.append((meth, r["file"], r[KEY[meth]], want))
checks["settings_equal_frozen_mapping"] = not viol
# ---- check: behind tiles only
fviol = []
for i in scenes:
    if regions[i]["tiles"]:
        d = C.depth(DATA, i, regions[i]["size"])
        fviol += [i for t in regions[i]["tiles"] if np.median(d[t["y"]:t["y"] + 64, t["x"]:t["x"] + 64])
                  <= regions[i]["focus_disp"]]
checks["blur_tiles_all_behind"] = not fviol

rows = []
gen_test = [i for i, s in sorted(scenes.items()) if s["kind"] == "gen" and s["split"] == "test"]
for i in gen_test:
    x = C.img(f"{DATA}/input/{i}.png")
    H, W = x.shape[:2]
    subj = C.subject(DATA, i, (H, W))
    dep = C.depth(DATA, i, (H, W))
    rg = regions[i]
    sb = BL.TileStack(x, rg["tiles"]) if rg["blur_measurable"] else None
    sf = BL.TileStack(x, rg["front_tiles"]) if rg["front_measurable"] else None
    for meth, rr in runs.items():
        for r in rr:
            if r["id"] != i:
                continue
            t = float(r["file"].split("_t")[-1][:-4])
            y = C.img(f"{OUT}/{r['file']}")
            row = dict(id=i, category=scenes[i]["category"], method=meth, target=t, setting=r[KEY[meth]], file=r["file"],
                       seconds=r["seconds"], mask_failure=i in MF, flag=i in FLAG, caution=i in CAUT)
            b = sb.measure(y) if sb else dict(r=float("nan"), n_good=0, n_tiles=0)
            row.update(blur=b["r"], blur_n_good=b["n_good"], blur_n_tiles=b["n_tiles"])
            row["front_blur"] = sf.measure(y)["r"] if sf else float("nan")
            blur_ok = rg["blur_measurable"] and i not in FLAG and np.isfinite(b["r"])
            row["control_err"] = b["r"] - t if blur_ok else float("nan")
            row["control_ok"] = bool(blur_ok and abs(b["r"] - t) <= C.TOL(t))
            row["control_status"] = ("excluded_flag" if i in FLAG else "no_tiles" if not rg["blur_measurable"]
                                     else "unmeasurable_output" if not np.isfinite(b["r"]) else
                                     "ok" if row["control_ok"] else "miss")
            if i not in MF and rg["n_inner"] > 0:
                row.update(MS.preservation(x, y, subj))
                bd = MS.boundary(x, y, subj, rg["tiles"], dep, rg["focus_disp"])
                row.update({k: bd[k] for k in ["halo_sharp", "thin_loss", "rim", "rim_inner", "bleed", "halo_kept",
                                               "far_hf_retention"]})
                if i in FLAG:
                    row["halo_sharp"] = float("nan")
                row["focus_ok"] = bool(np.isfinite(bd["far_hf_retention"]) and row["hf_ratio"] > bd["far_hf_retention"])
            rows.append(row)
    print(i, "scored", flush=True)

# matched cells
cell = {}
for r in rows:
    cell.setdefault((r["id"], r["target"]), {})[r["method"]] = r
for (i, t), c in cell.items():
    if len(c) == 2:
        a, b = c["bokehdiff"]["blur"], c["bokehlicious"]["blur"]
        m = bool(np.isfinite(a) and np.isfinite(b) and i not in FLAG and abs(a - b) <= C.TOL((a + b) / 2))
        for r in c.values():
            r["matched"] = m

# checks: mask-dependent metrics never computed on mask failures; flags out of control
checks["mask_failures_out_of_mask_metrics"] = all("detail" not in r for r in rows if r["mask_failure"])
checks["flag_out_of_control"] = all(not np.isfinite(r["control_err"]) for r in rows if r["flag"])


def summarise(sel, metric, by_scene=True):
    out = {}
    for meth in runs:
        v = {}
        for r in sel:
            if r["method"] == meth and metric in r and r[metric] is not None and np.isfinite(float(r[metric])):
                v.setdefault(r["id"], []).append(float(r[metric]))
        sc = {i: float(np.mean(x)) for i, x in v.items()}
        out[meth] = dict(n_scenes=len(sc), median=float(np.median(list(sc.values()))) if sc else float("nan"),
                         per_scene=sc)
    common = sorted(set(out["bokehdiff"]["per_scene"]) & set(out["bokehlicious"]["per_scene"]))
    dif = [out["bokehdiff"]["per_scene"][i] - out["bokehlicious"]["per_scene"][i] for i in common]
    out["paired"] = dict(n=len(common), median_diff=float(np.median(dif)) if dif else float("nan"),
                         bd_higher=int(sum(d > 0 for d in dif)), bkl_higher=int(sum(d < 0 for d in dif)))
    return out


summary = {}
groups = {"sharp": lambda r: r["category"] != "existing_blur", "existing_blur": lambda r: r["category"] == "existing_blur"}
for g, f in groups.items():
    G = [r for r in rows if f(r)]
    s = {}
    # control (all cells of measurable scenes; failures included)
    for meth in runs:
        for t in C.TARGETS:
            cc = [r for r in G if r["method"] == meth and r["target"] == t]
            st = {k: sum(r["control_status"] == k for r in cc) for k in ["ok", "miss", "unmeasurable_output", "no_tiles",
                                                                         "excluded_flag"]}
            err = [abs(r["control_err"]) for r in cc if np.isfinite(r["control_err"])]
            s[f"control/{meth}/{t:g}"] = dict(status=st, median_abs_err=float(np.median(err)) if err else float("nan"),
                                              mean_setting=mp[meth]["gen"][f"{t:g}"]["setting"],
                                              mapping_control_failure=mp[meth]["gen"][f"{t:g}"]["control_failure"])
    s["blur"] = summarise(G, "blur")
    s["front_blur"] = summarise(G, "front_blur")
    M = [r for r in G if r.get("matched")]
    n_cells = len({(r["id"], r["target"]) for r in G})
    s["matched_cells"] = dict(matched=len({(r["id"], r["target"]) for r in M}), total=n_cells)
    for variant, sel in [("all", M), ("without_cautions", [r for r in M if not r["caution"]])]:
        for metric in ["colour_shift", "detail", "hf_ratio", "noise", "halo_sharp", "thin_loss", "rim_inner", "rim",
                       "bleed"]:
            s[f"quality/{variant}/{metric}"] = summarise(sel, metric)
    for variant, sel in [("all", G), ("without_cautions", [r for r in G if not r["caution"]])]:
        for metric in ["colour_shift", "detail", "hf_ratio", "noise"]:
            s[f"preservation_all_targets/{variant}/{metric}"] = summarise(sel, metric)
        s[f"focus/{variant}"] = {meth: dict(ok=sum(1 for r in sel if r["method"] == meth and r.get("focus_ok")),
                                            n=sum(1 for r in sel if r["method"] == meth and "focus_ok" in r))
                                 for meth in runs}
    summary[g] = s

# RealBokeh test
rb = []
for i, sc in sorted(scenes.items()):
    if sc["kind"] != "rb" or sc["split"] != "test":
        continue
    x = C.img(f"{DATA}/input/{i}.png")
    for a in ["2.0", "second"]:
        av = 2.0 if a == "2.0" else [v for v in sc["apertures"] if v != 2.0][0]
        tg = C.to_lin(C.img(f"{DATA}/targets/{i}_f{av}.png"))
        base = float(np.abs(C.to_lin(x) - tg).mean() * 255)
        for meth, rr in runs.items():
            for r in rr:
                if r["id"] == i and r["file"].endswith(f"_{a}.png"):
                    yl = C.to_lin(C.img(f"{OUT}/{r['file']}"))
                    g = (yl * tg).sum((0, 1)) / np.maximum((yl * yl).sum((0, 1)), 1e-12)
                    rb.append(dict(id=i, aperture=a, av=av, method=meth, mask_failure=i in MF,
                                   mae_lin=float(np.abs(yl - tg).mean() * 255),
                                   mae_lin_gain=float(np.abs(yl * g - tg).mean() * 255), input_mae_lin=base,
                                   seconds=r["seconds"]))
for a in ["2.0", "second"]:
    for meth in runs:
        v = [r for r in rb if r["aperture"] == a and r["method"] == meth]
        summary[f"realbokeh/{a}/{meth}"] = dict(n=len(v), median_mae_lin=float(np.median([r["mae_lin"] for r in v])) if v else None,
                                                median_mae_lin_gain=float(np.median([r["mae_lin_gain"] for r in v])) if v else None,
                                                median_input_mae_lin=float(np.median([r["input_mae_lin"] for r in v])) if v else None,
                                                beats_input=int(sum(r["mae_lin"] < r["input_mae_lin"] for r in v)))

# latency
lat = json.load(open(f"{OUT}/lat40.json")) if os.path.exists(f"{OUT}/lat40.json") else {}
for meth, rr in runs.items():
    s_ = [r["seconds"] for r in rr]
    summary[f"latency/{meth}"] = dict(render_median_s=float(np.median(s_)), n=len(s_))
if lat:
    summary["latency/bokehdiff"]["total_median_s"] = summary["latency/bokehdiff"]["render_median_s"] + lat["prep_median_s"]
    summary["latency/bokehdiff"]["prep"] = lat
summary["latency/bokehlicious"]["total_median_s"] = summary["latency/bokehlicious"]["render_median_s"]

# blinded pairs
rng = np.random.RandomState(0)
cand = sorted({(r["id"], r["target"]) for r in rows if r.get("matched") and not r["mask_failure"]
               and r["category"] != "existing_blur"})
pick = [cand[k] for k in rng.permutation(len(cand))[:25]]
os.makedirs(f"{RES}/pairs", exist_ok=True)
key = []
for n, (i, t) in enumerate(pick):
    c = cell[(i, t)]
    order = ["bokehdiff", "bokehlicious"] if rng.rand() < 0.5 else ["bokehlicious", "bokehdiff"]
    ims = [Image.open(f"{OUT}/{c[m]['file']}").convert("RGB") for m in order]
    w, h = ims[0].size
    canvas = Image.new("RGB", (2 * w + 16, h), (255, 255, 255))
    canvas.paste(ims[0], (0, 0))
    canvas.paste(ims[1], (w + 16, 0))
    canvas.save(f"{RES}/pairs/pair_{n:02d}.png")
    key.append(dict(pair=n, scene=i, target=t, left=order[0], right=order[1]))
json.dump(key, open(f"{OUT}/pairs_key_DO_NOT_OPEN_BEFORE_RATING.json", "w"), indent=1)
with open(f"{RES}/pairs/ratings.csv", "w", newline="") as fh:
    w_ = csv.writer(fh)
    w_.writerow(["pair", "choice (left / right / tie / cannot judge)", "notes"])
    for n in range(len(pick)):
        w_.writerow([f"pair_{n:02d}", "", ""])
summary["pairs"] = dict(n=len(pick), candidates=len(cand))

summary["checks"] = checks
json.dump(dict(rows=rows, realbokeh=rb, summary=summary), open(f"{RES}/score40.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k.startswith(("control", "latency", "realbokeh", "checks", "pairs"))},
                 indent=1, default=float)[:6000])
for g in groups:
    print(g, "control:", {k: v["status"] for k, v in summary[g].items() if k.startswith("control")})
    print(g, "matched cells", summary[g]["matched_cells"])
print("checks", checks, flush=True)
if not all(checks.values()):
    print("CHECKS FAILED", viol[:5], fviol[:5], flush=True)
    sys.exit(1)
