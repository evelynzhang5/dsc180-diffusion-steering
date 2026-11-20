"""Whole-scene figure of the core problem from saved round 25 outputs (no new generations).
Panels: sharp input | what an ideal lens gives (every background cell blurred by the same 11.6 px disk, centre sharp) |
what BokehDiff actually gives, each cell labelled with requested -> delivered blur. Bottom: delivered radius per cell.
usage: python problem_scene.py <round25 output dir> <out dir>"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D, measure18 as m18, measure19 as M, patches as PT

R25, out = sys.argv[1], sys.argv[2]
T = 11.625
ANCH = D.NOM * D.K * 0.5
lib = PT.library()
grp = [p for p in lib if p["split"] == "test"][0::2]
img, cid, where = PT.layout(grp, 0)
im = img.astype(np.float32)
y = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{T:.1f}.png")).astype(np.float32)

# ideal lens render: each cell takes the disk blur of its own radius (centre stays sharp)
ideal = im.copy()
bt, ba = m18.blur(im, m18.disk(T)), m18.blur(im, m18.disk(ANCH))
for k in range(9):
    m = cid == k
    if k in (0, 8):
        ideal[m] = ba[m]
    elif k != 4:
        ideal[m] = bt[m]
ideal = np.clip(ideal, 0, 255)

st = m18.stack(im, "disk")
meas = {}
for k in PT.FREE:
    m = D.interior(cid, k)
    meas[k] = M.fit_fast(y[m], M.stack_at(st, m))["v"]


def colour(ratio):
    return "#4da3ff" if ratio < 0.93 else "#ff8a3d" if ratio > 1.07 else "white"


fig = plt.figure(figsize=(16, 9.4))
gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1])
panels = [(im, "1. Sharp input"),
          (ideal, f"2. What a real lens gives\nall six background cells: same {T:.1f} px blur"),
          (y, f"3. What BokehDiff gives\nsame {T:.1f} px request, different blur")]
for j, (a_im, title) in enumerate(panels):
    a = fig.add_subplot(gs[0, j])
    a.imshow(a_im.astype(np.uint8)); a.set_title(title, fontsize=12)
    a.set_xticks([]); a.set_yticks([])
    for e in D.EDGES[1:-1]:
        a.axhline(e, color="w", lw=0.8); a.axvline(e, color="w", lw=0.8)
    a.add_patch(plt.Rectangle((171, 171), 170, 170, fill=False, ec="yellow", lw=2.5))
    for k in range(9):
        r, c = divmod(k, 3)
        x0, y0 = D.EDGES[c] + 5, D.EDGES[r] + 18
        if k == 4:
            s, col = "in focus (subject)", "yellow"
        elif k in (0, 8):
            s, col = "filler", "#bbbbbb"
        else:
            name = where[k].split("_")[0].replace("chelsea", "cat")
            if j == 2:
                q = meas[k] / T
                s, col = f"{name}\n{meas[k]:.1f} px ({q:.2f}x)", colour(q)
            elif j == 1:
                s, col = f"{name}\n{T:.1f} px", "white"
            else:
                s, col = name, "white"
        a.text(x0, y0, s, color=col, fontsize=9.5, fontweight="bold" if j == 2 else None, va="top",
               bbox=dict(fc="black", alpha=0.6, lw=0))

b = fig.add_subplot(gs[1, :])
ks = sorted(PT.FREE, key=lambda k: meas[k])
names = [where[k].split("_")[0].replace("chelsea", "cat") for k in ks]
vals = [meas[k] for k in ks]
b.barh(names, vals, color=[colour(v / T) if colour(v / T) != "white" else "#9a9a9a" for v in vals])
b.axvline(T, color="black", ls="--", lw=1.5)
b.text(T, len(ks) - 0.35, f"requested = real lens: {T:.1f} px", fontsize=9, ha="center", va="bottom", bbox=dict(fc="white", lw=0))
for i, v in enumerate(vals):
    if v < T:
        b.text(v - 0.15, i, f"{v:.1f} px", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    else:
        b.text(v + 0.15, i, f"{v:.1f} px", va="center", fontsize=9)
b.set_xlim(0, 17); b.set_ylim(-0.6, len(ks) + 0.2); b.set_xlabel("blur actually delivered (disk-equivalent radius, px)")
b.spines[["top", "right"]].set_visible(False)
fig.suptitle("Same depth, same blur request: a real lens blurs every background cell equally; "
             "BokehDiff blurs by content (blue = too sharp, orange = too blurry)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{out}/problem_scene.png", dpi=110)
print("written", f"{out}/problem_scene.png", {where[k]: round(v, 1) for k, v in meas.items()})
