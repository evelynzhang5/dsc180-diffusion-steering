"""Round 24 patch library and layouts.

Patches: two 171x171 crops per source image (scikit-image images available offline), chosen as the two
non-overlapping crops with the highest mean gradient magnitude, native resolution. Whole sources are held out:
  development sources: grass, brick, camera, coffee, immunohistochemistry, page
  test sources:        gravel, astronaut, chelsea, coins, rocket, text
Layout: round 19's 3x3 grid. Centre = focus cell, corners (0,0)/(2,2) = anchors (fixed filler, not measured),
the other 6 cells ("free") hold 6 patches. Patches rotate through the 6 free positions across runs, so each patch is
seen at several positions with identical requests (content vs position). Every free cell gets the same |gap|;
pairs through the centre are front (+gap) / behind (-gap); d_f = 0.5; anchors +-0.5 keep the map spanning [0, 1].
"""
import numpy as np
from skimage import color, data

import design as D

DEV_SRC = ["grass", "brick", "camera", "coffee", "immunohistochemistry", "page"]
TEST_SRC = ["gravel", "astronaut", "chelsea", "coins", "rocket", "text"]
P = 171
FREE = [1, 2, 3, 5, 6, 7]                 # cell ids 3*row+col
SIGN = {2: +1, 6: -1, 1: +1, 7: -1, 3: +1, 5: -1}   # pairs (0,2)/(2,0), (0,1)/(2,1), (1,0)/(1,2): front / behind


def _load(name):
    a = getattr(data, name)()
    a = a[..., :3] if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
    return a.astype(np.uint8)


def _grad(a):
    g = color.rgb2gray(a).astype(np.float32)
    return np.abs(np.diff(g, axis=0))[:, :-1] + np.abs(np.diff(g, axis=1))[:-1, :]


def library():
    """list of dict(id, source, split, pix uint8 171x171x3, stats)"""
    out = []
    for split, srcs in (("dev", DEV_SRC), ("test", TEST_SRC)):
        for s in srcs:
            a = _load(s)
            H, W = a.shape[:2]
            g = _grad(a)
            ii = np.pad(np.cumsum(np.cumsum(g, 0), 1), ((1, 0), (1, 0)))
            ys_ = sorted(set(list(range(0, H - P + 1, 8)) + [H - P]))
            xs_ = sorted(set(list(range(0, W - P + 1, 8)) + [W - P]))
            best = []
            for y in ys_:
                for x in xs_:
                    yy, xx = min(y + P - 1, g.shape[0]), min(x + P - 1, g.shape[1])
                    best.append((ii[yy, xx] - ii[y, xx] - ii[yy, x] + ii[y, x], y, x))
            best.sort(reverse=True)
            top = best[:600]
            pair, score = None, -1.0
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    (v1, y1, x1), (v2, y2, x2) = top[i], top[j]
                    if (abs(y1 - y2) >= P or abs(x1 - x2) >= P) and v1 + v2 > score:
                        pair, score = (top[i], top[j]), v1 + v2
                        break                       # top is sorted: first valid j is best for this i
            chosen = list(pair)
            for k, (v, y, x) in enumerate(chosen):
                pix = a[y:y + P, x:x + P]
                out.append(dict(id=f"{s}_{k}", source=s, split=split, pix=pix, crop=(int(y), int(x)), stats=stats(pix)))
    return out


def stats(pix):
    g = color.rgb2gray(pix).astype(np.float64)
    gr = _grad(pix)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.mgrid[-P // 2 + 1:P // 2 + 1, -P // 2 + 1:P // 2 + 1][:, :F.shape[0], :F.shape[1]]
    rad = np.sqrt(yy ** 2 + xx ** 2)
    hf = float(F[rad > P / 8].sum() / max(F.sum(), 1e-12))
    return dict(log_std=float(np.log(g.std() + 1e-4)), log_grad=float(np.log(gr.mean() + 1e-5)), hf_ratio=hf)


FILL_ANCHOR = lambda: _load("hubble_deep_field")[300:471, 300:471]
FILL_FOCUS = lambda: np.rot90(_load("brick"))[:P, :P]


def layout(patches6, rot):
    """image + cid with patches6[j] at FREE[(j + rot) % 6]; returns (image, cid, {cell: patch id})"""
    img = np.zeros((D.S, D.S, 3), np.uint8)
    cid = np.zeros((D.S, D.S), np.int32)
    where = {FREE[(j + rot) % 6]: p for j, p in enumerate(patches6)}
    anch, foc = FILL_ANCHOR(), FILL_FOCUS()
    for r in range(3):
        for c in range(3):
            k = 3 * r + c
            ys, xs = D.cell_slice((r, c))
            h, w = ys.stop - ys.start, xs.stop - xs.start
            src = foc if k == 4 else anch if k in (0, 8) else where[k]["pix"]
            img[ys, xs] = src[:h, :w]
            cid[ys, xs] = k
    return img, cid, {k: p["id"] for k, p in where.items()}


def maps(cid, gaps):
    """gaps: {cell: signed gap} for free cells (focus 0, anchors +-0.5 fixed); blur map = |gap| * K / 10 unless
    gaps[cell] is given as ('coc', value) for a corrected request."""
    disp = np.full((D.S, D.S), 0.5, np.float32)
    coc = np.zeros((D.S, D.S), np.float32)
    for k, g in [(0, +0.5), (8, -0.5)]:
        disp[cid == k] = 0.5 + g
        coc[cid == k] = abs(g) * D.K / 10
    for k, g in gaps.items():
        m = cid == k
        if isinstance(g, tuple):           # ("coc", value, true signed gap)
            disp[m] = 0.5 + g[2]
            coc[m] = g[1]
        else:
            disp[m] = 0.5 + g
            coc[m] = abs(g) * D.K / 10
    assert abs(disp.min()) < 1e-6 and abs(disp.max() - 1) < 1e-6
    return disp, coc


if __name__ == "__main__":
    import sys
    from PIL import Image
    lib = library()
    for p in lib:
        print(p["id"], p["split"], p["crop"], {k: round(v, 3) for k, v in p["stats"].items()})
    if len(sys.argv) > 1:
        dev = [p for p in lib if p["split"] == "dev"]
        img, cid, w = layout(dev[:6], 0)
        Image.fromarray(img).save(sys.argv[1] + "/layout_dev0.png")
        tst = [p for p in lib if p["split"] == "test"]
        img, cid, w = layout(tst[:6], 0)
        Image.fromarray(img).save(sys.argv[1] + "/layout_test0.png")
        sheet = Image.new("RGB", (12 * 90, 2 * 90), "white")
        for i, p in enumerate(lib):
            sheet.paste(Image.fromarray(p["pix"]).resize((86, 86)), ((i % 12) * 90, (i // 12) * 90))
        sheet.save(sys.argv[1] + "/patches.png")
