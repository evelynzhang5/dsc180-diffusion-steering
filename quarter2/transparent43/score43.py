"""Round 43 (CPU, on saved outputs): score the transparent-subject test. FROZEN before any round 43 output is viewed.

Regions (regions43.json + review43.json): behind tiles (real background around the subject), see tiles (background
seen THROUGH the subject, chosen by eye from numbered candidates on the INPUT sheets), rim band (subject pixels within
6 px of its edge: the object's own outline).
Per output (rendered scene x condition x K):
  loss_see, loss_behind   detail loss = 1 - retention of mid-scale detail (energy of the difference of Gaussians
             sigma 2 and 5 px, output / input, all channels) over the see tiles and over the behind tiles. The mid band
             is used because BokehDiff's regeneration removes part of the FINEST detail everywhere (round 40 subjects
             keep 0.46-0.82 at sigma 0.5-1.5 px but 0.84-0.96 at 2-5 px), and because dust/scratches on a real
             object's surface sit mostly in the finest band (dry run, before any round 43 output: kept-sharp
             subjects 0.84-0.96, blurred background 0.12).
  rho        PRIMARY: loss_see / loss_behind (NaN if loss_behind < 0.3, i.e. the background itself was not blurred), the share of the surrounding background's detail loss that the
             through-content receives (1 = blurred like the background, 0 = kept as sharp as the input).
             Chosen over the blur-radius estimator because a see tile holds two layers (the object's surface: dust,
             scratches, reflections; and the scene behind), and the single-blur fit of blur40 fails or reads 0 on such
             tiles even on the REAL f/2.0 capture (dry run on rb_48).
  rho_blur   secondary: blur40 radius on see tiles / on behind tiles (NaN when the fit fails).
  rim_keep   rim-band hf retention relative to the original condition at the same K (1 = outline as sharp as before).
  rho_real   for RealBokeh scenes: rho measured on the real f/2.0 capture (what a real lens does: the reference).
  req_ratio  requested blur (median coc) over see / behind tiles under each condition's depth map (diagnostic).
Decisions (medians over the scored scenes; 95% scene-bootstrap CIs, 10,000, seed 0). Primary K = the depth-rule
setting for 12 px; 8 px reported the same way.
  FAILURE        original rho <= 0.3 at the 12 px setting AND at K = 64: BokehDiff keeps the through-content sharp
                 and no setting fixes it. Otherwise NO FAILURE (the rest is reported, not decided).
  cause, if FAILURE:
    INPUT        seethrough_rim rho >= 0.7 and its rim_keep >= 0.8: a corrected one-depth-per-pixel map fixes it
                 without blurring the outline; the failure is in the depth input, not the model.
    MODEL        seethrough rho < 0.5: even told the through-content is far, the model keeps it sharp; a model-side
                 failure (a target for steering).
    EDGE_TRADEOFF seethrough rho >= 0.7 but seethrough_rim fails a criterion: the through-content blurs only if the
                 outline is also asked to blur.
    MIXED        anything else.
  Real-lens check (RealBokeh scenes): rho_real is reported next to each condition's rho; it shows how far a real lens
  is from 1 when the object's surface stays sharp over the blurred scene.
usage (repo root): python quarter2/transparent43/score43.py <R43 data> <R43 out> <results dir> <review43.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "compare40"))
import blur40 as BL  # noqa: E402
import common40 as C  # noqa: E402
import measures40 as MS  # noqa: E402

DATA, OUT, RES, REVIEW = sys.argv[1:5]
os.makedirs(RES, exist_ok=True)
regions = json.load(open(f"{DATA}/regions43.json"))
run = json.load(open(f"{OUT}/bd43.json"))
review = json.load(open(REVIEW))
for i, rg in regions.items():
    rg["see_tiles"] = [c for c in rg.get("see_candidates", []) if c["k"] in review["see_tiles"].get(i, [])]
keep = {i for i, rg in regions.items() if rg["eligible"] and len(rg["see_tiles"]) >= 2 and i not in review["exclude"]}
run["rows"] = [r for r in run["rows"] if r["id"] in keep]
rng = np.random.default_rng(0)
RIM = 6


def req(coc, tiles, sx, sy):
    return float(np.median(np.concatenate([coc[int(t["y"] * sy):int((t["y"] + 64) * sy),
                                               int(t["x"] * sx):int((t["x"] + 64) * sx)].ravel() for t in tiles])))


def band(z, a=2.0, b=5.0):
    from scipy.ndimage import gaussian_filter
    z = z.astype(np.float64)
    return np.stack([gaussian_filter(z[..., c], a) - gaussian_filter(z[..., c], b) for c in range(z.shape[2])], -1)


def subj_mask(i, H, W):
    from PIL import Image
    return np.asarray(Image.open(f"{DATA}/mask/{i}.png").convert("L").resize((W, H))) / 255.0 >= 0.5


rows, real = [], {}
for i in sorted({r["id"] for r in run["rows"]}):
    rg = regions[i]
    x = C.img(f"{DATA}/input/{i}.png")
    H, W = x.shape[:2]
    subj = subj_mask(i, H, W)
    din, _ = MS.geometry(subj)
    rim = (din > 0) & (din <= RIM)
    hx = MS.hf(x)
    st_out, st_see = BL.TileStack(x, rg["tiles"]), BL.TileStack(x, rg["see_tiles"])
    tgt = f"{DATA}/targets/{i}_f2.0.png"
    m_see, m_out = MS.tile_mask(subj.shape, rg["see_tiles"]), MS.tile_mask(subj.shape, rg["tiles"])

    bx = band(x)

    def losses(y):
        by = band(y)
        ls, lb = 1 - MS._hfr(by, bx, m_see), 1 - MS._hfr(by, bx, m_out)
        return ls, lb, (ls / lb if lb >= 0.3 else float("nan")), MS.hf(y)

    if os.path.exists(tgt):
        yr = C.img(tgt)
        ls, lb, rho, _ = losses(yr)
        ob, sb = st_out.measure(yr)["r"], st_see.measure(yr)["r"]
        real[i] = dict(loss_see=ls, loss_behind=lb, rho_real=rho, out_blur=ob, see_blur=sb,
                       rho_blur_real=sb / ob if np.isfinite(sb) and np.isfinite(ob) and ob > 0.5 else float("nan"))
    for r in [r for r in run["rows"] if r["id"] == i]:
        y = C.img(f"{OUT}/{r['file']}")
        dd = np.load(f"{OUT}/{r['depth']}").astype(np.float32)
        coc = np.abs(dd - r["focus"]) * r["K"] / 10
        sx, sy = dd.shape[1] / W, dd.shape[0] / H
        r = dict(r, req_see=req(coc, rg["see_tiles"], sx, sy), req_behind=req(coc, rg["tiles"], sx, sy))
        ob, sb = st_out.measure(y)["r"], st_see.measure(y)["r"]
        ls, lb, rho, hy = losses(y)
        rows.append(dict(r, loss_see=ls, loss_behind=lb, rho=rho, out_blur=ob, see_blur=sb,
                         rho_blur=sb / ob if np.isfinite(sb) and np.isfinite(ob) and ob > 0.5 else float("nan"),
                         req_ratio=r["req_see"] / r["req_behind"] if r["req_behind"] > 0 else float("nan"),
                         rim_hf=MS._hfr(hy, hx, rim)))
    print(i, "scored", flush=True)


def get(i, cond, k, key):
    v = [r[key] for r in rows if r["id"] == i and r["cond"] == cond and r["k_tag"] == k]
    return v[0] if v else float("nan")


ids = sorted({r["id"] for r in rows})
for r in rows:
    base = get(r["id"], "original", r["k_tag"], "rim_hf")
    r["rim_keep"] = r["rim_hf"] / base if base > 0 else float("nan")


def boot_median(v):
    v = np.array([a for a in v if np.isfinite(a)])
    if len(v) == 0:
        return dict(n=0, est=float("nan"), lo=float("nan"), hi=float("nan"))
    b = [np.median(rng.choice(v, len(v))) for _ in range(10000)]
    return dict(n=int(len(v)), est=float(np.median(v)), lo=float(np.percentile(b, 2.5)),
                hi=float(np.percentile(b, 97.5)))


summ = {}
for cond, ks in [("original", ["8", "12", "K36", "K64"]), ("seethrough", ["8", "12"]), ("seethrough_rim", ["8", "12"])]:
    for k in ks:
        summ[f"{cond}@{k}"] = {key: boot_median([get(i, cond, k, key) for i in ids])
                               for key in ["rho", "rho_blur", "rim_keep", "req_ratio", "loss_behind", "out_blur"]}
summ["real_f2.0"] = dict(rho_real=boot_median([v["rho_real"] for v in real.values()]),
                         rho_blur_real=boot_median([v["rho_blur_real"] for v in real.values()]))


def decide(k):
    o, o64 = summ[f"original@{k}"]["rho"]["est"], summ["original@K64"]["rho"]["est"]
    s, sr = summ[f"seethrough@{k}"]["rho"]["est"], summ[f"seethrough_rim@{k}"]["rho"]["est"]
    rk = summ[f"seethrough_rim@{k}"]["rim_keep"]["est"]
    if not (o <= 0.3 and o64 <= 0.3):
        return "NO FAILURE"
    if sr >= 0.7 and rk >= 0.8:
        return "FAILURE: INPUT"
    if s < 0.5:
        return "FAILURE: MODEL"
    if s >= 0.7:
        return "FAILURE: EDGE_TRADEOFF"
    return "FAILURE: MIXED"


counts = {f"{c}@{k}": dict(rho_ge_0_7=sum(get(i, c, k, "rho") >= 0.7 for i in ids),
                           rho_le_0_3=sum(get(i, c, k, "rho") <= 0.3 for i in ids), n=len(ids))
          for c, k in [("original", "12"), ("original", "K64"), ("seethrough", "12"), ("seethrough_rim", "12")]}
decisions = dict(primary_12px=decide("12"), secondary_8px=decide("8"), scene_counts=counts,
                 n_scenes=len(ids), n_realbokeh=sum(i.startswith("rb_") for i in ids))
json.dump(dict(decisions=decisions, summary=summ, real=real, rows=rows), open(f"{RES}/score43.json", "w"),
          indent=1, default=float)
print(f"\n{len(ids)} scenes")
print(f"{'scene':10s} {'orig@12':>8s} {'orig@64':>8s} {'see@12':>8s} {'rim@12':>8s} {'rimkeep':>8s} {'real':>6s}")
for i in ids:
    print(f"{i:10s} {get(i,'original','12','rho'):8.2f} {get(i,'original','K64','rho'):8.2f} "
          f"{get(i,'seethrough','12','rho'):8.2f} {get(i,'seethrough_rim','12','rho'):8.2f} "
          f"{get(i,'seethrough_rim','12','rim_keep'):8.2f} {real.get(i, {}).get('rho_real', float('nan')):6.2f}")
for k, v in summ.items():
    print(k, {kk: f"{vv['est']:.2f} [{vv['lo']:.2f},{vv['hi']:.2f}] n={vv['n']}" for kk, vv in v.items()})
print("decisions:", json.dumps(decisions, indent=1))
