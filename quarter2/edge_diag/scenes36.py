"""Round 36 scene plan: depth (disparity) and content vary independently; blur request decorrelated from depth.

Sources (whole images held out by split, as round 24):
  development: grass, brick, camera, coffee, immunohistochemistry, page        (2 crops each = 12 patches)
  test:        gravel, astronaut, chelsea, coins, rocket, text                  (fresh images)
Layout: round 19/24's 3x3 grid; centre = focus cell, corners 0/8 = anchors at disparity 1 / 0 (the map spans [0, 1]),
6 free cells hold one crop of each of the split's 6 sources.
12 scenes per split: crop group g in {0, 1} x focus disparity d_f in {0.35, 0.5, 0.65} x 2 rotations.
Within a group, scene s (0..5) puts patch j at FREE[(j + s) % 6] with signed gap GAPS[(j - s) % 6], so
  * every patch appears at all 6 positions and with all 6 signed gaps (same texture at different depths),
  * every scene holds all 6 signed gaps: each |gap| twice, once in front and once behind (same requested blur,
    different depth), and different textures share each depth.
Disparity d = d_f + gap in [0, 1]; requested (nominal) radius r* = NOM * K * |gap| in {5.8, 9.7, 13.6} px.
Over the design, corr(d, |gap|) = 0, and the varying d_f separates d from the signed gap.
"""
import numpy as np

import design as D
import patches as PT

GAPS = [-0.35, -0.25, -0.15, 0.15, 0.25, 0.35]
DFS = [0.35, 0.5, 0.65]
ANCHOR_DISP = {0: 1.0, 8: 0.0}


def rstar(gap):
    return D.NOM * D.K * abs(gap)


def plan(split):
    lib = [p for p in PT.library() if p["split"] == split]
    srcs = PT.DEV_SRC if split == "dev" else PT.TEST_SRC
    out = []
    for i in range(12):
        g, s = i % 2, i // 2                         # crop group; scene index within the group
        d_f = DFS[s % 3]
        patches = [next(p for p in lib if p["id"] == f"{src}_{g}") for src in srcs]
        img, cid, where = PT.layout(patches, s)
        gaps = {PT.FREE[(j + s) % 6]: GAPS[(j - s) % 6] for j in range(6)}
        out.append(dict(name=f"{split}{i:02d}", split=split, group=g, rot=s, d_f=d_f, img=img, cid=cid,
                        where=where, gaps=gaps, disp={k: d_f + v for k, v in gaps.items()}))
    return out


def maps(sc, coc_free):
    """disparity map (true geometry) and blur map; coc_free {cell: coc} for the free cells (requests)."""
    cid, d_f = sc["cid"], sc["d_f"]
    disp = np.full(cid.shape, d_f, np.float32)
    coc = np.zeros(cid.shape, np.float32)
    for k, d in ANCHOR_DISP.items():
        disp[cid == k] = d
        coc[cid == k] = abs(d - d_f) * D.K / 10
    for k, g in sc["gaps"].items():
        disp[cid == k] = d_f + g
        coc[cid == k] = coc_free[k]
    assert abs(disp.min()) < 1e-6 and abs(disp.max() - 1) < 1e-6, (sc["name"], disp.min(), disp.max())
    return disp, coc


def check():
    rows = []
    for split in ("dev", "test"):
        for sc in plan(split):
            for k, g in sc["gaps"].items():
                rows.append((split, sc["where"][k].rsplit("_", 1)[0], k, sc["d_f"], g, sc["d_f"] + g, abs(g)))
    a = np.array([(r[3], r[4], r[5], r[6]) for r in rows], float)
    print("cells", len(rows), "corr(d, |gap|) %.3f corr(d, gap) %.3f corr(d, d_f) %.3f"
          % (np.corrcoef(a[:, 2], a[:, 3])[0, 1], np.corrcoef(a[:, 2], a[:, 1])[0, 1], np.corrcoef(a[:, 2], a[:, 0])[0, 1]))
    for split in ("dev", "test"):
        src = {}
        for r in rows:
            if r[0] == split:
                src.setdefault(r[1], set()).add((r[4], r[2]))
        print(split, {s: (len({g for g, _ in v}), len({k for _, k in v})) for s, v in src.items()}, "(gaps, positions) per source")
    print("disparity values", sorted({round(r[5], 2) for r in rows}))


if __name__ == "__main__":
    check()
