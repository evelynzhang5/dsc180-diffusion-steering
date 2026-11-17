"""Figure of the core problem from saved round 25 outputs (no new generations): the same blur request (11.6 px)
on four held-out patches - input, BokehDiff output, ideal disk blur of the requested radius (sRGB and linear light) -
with the measured disk-equivalent radius, plus the kernel profiles from round 27."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design as D, measure18 as m18, measure19 as M, patches as PT

R25, out = sys.argv[1], sys.argv[2]
T = 11.625
lib = PT.library()
grp = [p for p in lib if p["split"] == "test"][0::2]
img, cid, where = PT.layout(grp, 0)
y = np.asarray(Image.open(f"{R25}/img/cor_g0_r0_G0_t{T:.1f}.png")).astype(np.float32)
im = img.astype(np.float32)
srgb = np.clip(m18.blur(im, m18.disk(T)), 0, 255)
lin = (np.clip(m18.blur((im / 255.0) ** 2.2, m18.disk(T)), 0, 1) ** (1 / 2.2)) * 255
st = m18.stack(im, "disk")
show = ["chelsea_0", "astronaut_0", "gravel_0", "coins_0"]
cell = {v: int(k) for k, v in where.items()}
fig, ax = plt.subplots(4, 4, figsize=(10, 10.6))
for i, p in enumerate(show):
    k = cell[p]
    m = D.interior(cid, k)
    r = M.fit_fast(y[m], M.stack_at(st, m))["v"]
    ys, xs = np.where(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    sl = (slice(cy - 48, cy + 48), slice(cx - 48, cx + 48))
    for j, (a, title) in enumerate([(im, "sharp input"), (y, f"BokehDiff: {r:.1f} px ({r / T:.2f}x)"),
                                    (srgb, f"ideal disk {T:.1f} px"), (lin, f"ideal disk, linear light")]):
        ax[i, j].imshow(np.clip(a[sl], 0, 255).astype(np.uint8), interpolation="nearest")
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, j].set_title(title, fontsize=9)
    ax[i, 0].set_ylabel(p, fontsize=10)
fig.suptitle(f"Same requested blur ({T:.1f} px), different content: BokehDiff (fp32 + calibration) under-blurs soft\n"
             "texture and over-blurs high-contrast texture; ideal lens blur (either colour space) does not", fontsize=10)
fig.tight_layout()
fig.savefig(f"{out}/problem_examples.png", dpi=120)
print("written", f"{out}/problem_examples.png")
