"""Round 19 design: six controlled 3x3-cell scenes and the run conditions.

Scene: 512x512, 3x3 cells (171/170/171 px). The centre cell is the focus cell. The other 8 cells form 4 pairs placed
symmetrically through the centre; both cells of a pair carry the IDENTICAL texture crop. Pair 0 (corners TL/BR) is the
"anchor" pair. Textures: scikit-image grass / gravel / brick (sharp photos, native resolution, tinted); "_r" = rotated.

Condition: focus disparity d_f, K = 40, and a signed disparity gap per pair (da, db) for (cell a, cell b) in
orientation P1; P2 swaps them (same content and request, positions exchanged). d = d_f + gap (front = gap > 0 =
nearer). Every condition's disparity map spans exactly [0, 1], as BokehDiff's loader always produces.
Requested (nominal) radius r_nom = NOM * K * |gap| px (NOM = 0.96875, from the attention code; see round 18).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

import scenes18 as s18

S = 512
EDGES = [0, 171, 341, 512]
K = 40.0
NOM = 31 * 16 / 51.2 / 10
CENTRE = (1, 1)
PAIRS = [((0, 0), (2, 2)), ((0, 2), (2, 0)), ((0, 1), (2, 1)), ((1, 0), (1, 2))]   # pair 0 = anchors
MARGIN_EDGE, MARGIN_BORDER = 48, 24

SCENES = {   # centre texture, then one texture per pair
    "S1": ("grass", ["gravel", "grass_r", "brick", "gravel_r"]),
    "S2": ("gravel", ["grass", "brick_r", "gravel_r", "grass_r"]),
    "S3": ("brick", ["gravel_r", "grass", "gravel", "brick_r"]),
    "S4": ("grass_r", ["brick", "gravel", "grass", "gravel_r"]),
    "S5": ("gravel_r", ["grass_r", "brick", "gravel", "grass"]),
    "S6": ("brick_r", ["gravel", "grass_r", "brick", "gravel_r"]),
}
DEV, TEST = ["S1", "S2"], ["S3", "S4", "S5", "S6"]

# run lists: (d_f, [(da, db) for pairs 0..3]) in orientation P1
RUNS = {
    "R0": (0.5, [(+.5, -.5), (0, 0), (0, 0), (0, 0)]),                 # calm: only anchors blurred
    "R1": (0.5, [(+.5, -.5), (+.05, -.05), (+.10, -.10), (+.15, -.15)]),
    "R2": (0.5, [(+.5, -.5), (+.20, -.20), (+.25, -.25), (+.30, -.30)]),
    "R3": (0.5, [(+.5, -.5), (+.35, -.35), (+.40, -.40), (+.45, -.45)]),  # busy
    "R4": (0.5, [(+.5, -.5), (+.075, -.075), (+.125, -.125), (+.175, -.175)]),  # fine steps near the plateau
    "R5": (0.2, [(+.8, -.2), (+.55, -.05), (+.65, -.10), (+.70, -.15)]),  # large requests in front
    "R6": (0.8, [(-.8, +.2), (-.55, +.05), (-.65, +.10), (-.70, +.15)]),  # large requests behind
}
GRID_RUNS = ["R1", "R2", "R3", "R4", "R5", "R6"]


def cell_slice(rc):
    r, c = rc
    return slice(EDGES[r], EDGES[r + 1]), slice(EDGES[c], EDGES[c + 1])


def make(name):
    """image uint8, cell-id map (0..8 = 3*row+col)."""
    centre, pt = SCENES[name]
    si = int(name[1])
    img = np.zeros((S, S, 3), np.float32)
    cid = np.zeros((S, S), np.int32)
    tex = {CENTRE: (centre, 0)}
    for p, (a, b) in enumerate(PAIRS):
        tex[a] = tex[b] = (pt[p], p + 1)
    for (r, c), (t, k) in tex.items():
        ys, xs = cell_slice((r, c))
        h, w = ys.stop - ys.start, xs.stop - xs.start
        oy, ox = (37 * si + 97 * k) % (S - h), (53 * si + 71 * k) % (S - w)   # same crop for both cells of a pair
        img[ys, xs] = s18._tex(t, oy, oy + h, ox, ox + w)
        cid[ys, xs] = 3 * r + c
    return dict(image=np.clip(np.round(img * 255), 0, 255).astype(np.uint8), cid=cid,
                texture={3 * r + c: t for (r, c), (t, _) in tex.items()})


def interior(cid, cell):
    m = cid == cell
    d = distance_transform_edt(np.pad(m, 1, mode="edge"))[1:-1, 1:-1]
    yy, xx = np.mgrid[:S, :S]
    border = np.minimum(np.minimum(yy, S - 1 - yy), np.minimum(xx, S - 1 - xx))
    return m & (d > MARGIN_EDGE) & (border >= MARGIN_BORDER)


def cells(run, perm):
    """list of dict(cell, pair, gap, side) for the 9 cells; centre gap 0."""
    d_f, gaps = RUNS[run]
    out = [dict(cell=4, pair=-1, gap=0.0, side="focus")]
    for p, ((a, b), (ga, gb)) in enumerate(zip(PAIRS, gaps)):
        if perm == "P2":
            ga, gb = gb, ga
        for rc, g in ((a, ga), (b, gb)):
            out.append(dict(cell=3 * rc[0] + rc[1], pair=p, gap=float(g),
                            side="focus" if g == 0 else "front" if g > 0 else "behind"))
    return d_f, out


def maps(cid, run, perm, coc_override=None):
    """disparity map (true geometry) and blur map. coc_override: {cell: coc} replaces the blur map per cell
    (used for calibration-corrected requests; the disparity channel keeps the true geometry)."""
    d_f, cl = cells(run, perm)
    disp = np.zeros((S, S), np.float32)
    coc = np.zeros((S, S), np.float32)
    for c in cl:
        m = cid == c["cell"]
        disp[m] = d_f + c["gap"]
        coc[m] = abs(c["gap"]) * K / 10.0
        if coc_override is not None and c["cell"] in coc_override:
            coc[m] = coc_override[c["cell"]]
    assert abs(disp.min()) < 1e-6 and abs(disp.max() - 1) < 1e-6, (run, perm, disp.min(), disp.max())
    return disp, coc


# ---------------- validation layouts for the estimator (same 3x3 cells, known disk radii)
LAYOUTS = {n: None for n in SCENES}


if __name__ == "__main__":
    import sys
    from PIL import Image
    out = sys.argv[1]
    for n in SCENES:
        s = make(n)
        Image.fromarray(s["image"]).save(f"{out}/{n}.png")
        print(n, {k: int(interior(s["cid"], k).sum()) for k in range(9)})
    for r in RUNS:
        for p in ("P1", "P2"):
            maps(make("S1")["cid"], r, p)
    print("all condition maps span [0, 1]")
