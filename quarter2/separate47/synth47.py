"""Round 47: procedural transparent-surface synthesis (shared by training and the T2 test set).

Compositing model as round 46, in linear light: I = alpha * F + (1 - alpha) * B inside a pane; alpha = 0 outside.
TRAIN families (random parameters): strokes, grid, specular, tint (+ scratch texture), rim, dust; 1-2 families per pane.
HELD-OUT families (T2 only, never in training): text glyphs, dot patterns.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

TRAIN_FAMILIES = ["strokes", "grid", "specular", "tint", "rim", "dust"]
HELDOUT_FAMILIES = ["text", "dots"]


def to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def random_pane(rng, H, W):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    w, h = rng.uniform(0.35, 0.85) * W, rng.uniform(0.35, 0.85) * H
    x0, y0 = rng.uniform(0, W - w), rng.uniform(0, H - h)
    kind = rng.integers(3)
    if kind == 0:
        d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=float(rng.uniform(0, min(w, h) / 3)), fill=255)
    elif kind == 1:
        d.ellipse((x0, y0, x0 + w, y0 + h), fill=255)
    else:
        n = int(rng.integers(5, 9))
        t = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(0.7, 1.0, n)
        d.polygon([(x0 + w / 2 + r[k] * w / 2 * np.cos(t[k]), y0 + h / 2 + r[k] * h / 2 * np.sin(t[k])) for k in range(n)],
                  fill=255)
    return np.asarray(im) / 255.0


def _layer(rng, kind, pane, H, W):
    """returns (alpha_k, F_k) for one family, alpha_k already restricted to the pane."""
    col = (rng.uniform(0.7, 1.0, 3) if rng.random() < 0.6 else rng.uniform(0.1, 0.9, 3))
    F = np.ones((H, W, 3)) * col
    if kind == "strokes":
        im = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(im)
        for _ in range(int(rng.integers(3, 15))):
            p = rng.uniform(0, 1, (4, 2)) * [W, H]
            t = np.linspace(0, 1, 50)[:, None]
            c = (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
            dr.line([tuple(q) for q in c], fill=255, width=int(rng.integers(1, 9)))
        s = np.asarray(im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))) / 255.0
        a = s * rng.uniform(0.5, 0.95)
    elif kind == "grid":
        sp, wd, ang = int(rng.integers(10, 40)), int(rng.integers(1, 4)), rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:H, 0:W]
        u, v = xx * np.cos(ang) + yy * np.sin(ang), -xx * np.sin(ang) + yy * np.cos(ang)
        s = (((u % sp) < wd) | ((v % sp) < wd)).astype(float)
        a = s * rng.uniform(0.4, 0.85)
    elif kind == "specular":
        yy, xx = np.mgrid[0:H, 0:W]
        ang = rng.uniform(0, np.pi)
        u = xx * np.cos(ang) + yy * np.sin(ang)
        s = np.zeros((H, W))
        for _ in range(int(rng.integers(1, 4))):
            up = u[pane > 0.5] if (pane > 0.5).any() else u.ravel()
            s = np.maximum(s, np.exp(-((u - rng.uniform(up.min(), up.max())) / rng.uniform(4, 30)) ** 2))
        a = s * rng.uniform(0.3, 0.7)
        F = np.ones((H, W, 3)) * rng.uniform(0.85, 1.0)
    elif kind == "tint":
        tex = rng.normal(0, 1, (H // 2 + 1, W // 2 + 1)).astype(np.float32)
        tex = np.asarray(Image.fromarray(tex).resize((W, H), resample=Image.BILINEAR))
        a = np.clip(rng.uniform(0.15, 0.5) + rng.uniform(0, 0.1) * np.clip(tex / 3, -1, 1), 0, 1)
        F = np.ones((H, W, 3)) * rng.uniform(0.1, 0.9, 3)
    elif kind == "rim":
        m = Image.fromarray((pane * 255).astype(np.uint8))
        inner = np.asarray(m.filter(ImageFilter.MinFilter(2 * int(rng.integers(1, 5)) + 1))) / 255.0
        a = np.clip(pane - inner, 0, 1) * rng.uniform(0.6, 0.95)
    elif kind == "dust":
        s = ((rng.random((H, W)) < rng.uniform(0.002, 0.02)) * 255).astype(np.uint8)
        s = np.asarray(Image.fromarray(s).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.2)))) / 255.0
        a = np.clip(s / max(s.max(), 1e-6), 0, 1) * rng.uniform(0.4, 0.9)
    elif kind == "text":
        im = Image.new("L", (W // 4, H // 4), 0)
        dr = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
        for _ in range(int(rng.integers(30, 45))):
            txt = "".join(rng.choice(list(chars), int(rng.integers(3, 8))))
            dr.text((rng.uniform(0, W // 4 - 30), rng.uniform(0, H // 4 - 10)), txt, fill=255, font=font)
        s = np.asarray(im.resize((W, H), resample=Image.NEAREST).filter(ImageFilter.GaussianBlur(0.6))) / 255.0
        a = s * rng.uniform(0.6, 0.9)
    elif kind == "dots":
        im = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(im)
        for _ in range(int(rng.integers(400, 700))):
            x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(2, 7)
            dr.ellipse((x - r, y - r, x + r, y + r), fill=255)
        s = np.asarray(im.filter(ImageFilter.GaussianBlur(0.6))) / 255.0
        a = s * rng.uniform(0.6, 0.9)
    else:
        raise ValueError(kind)
    return np.clip(a * pane, 0, 1), F


def random_surface(rng, pane, families):
    """combine 1-2 families (over-compositing); returns alpha (H,W), F (H,W,3)."""
    H, W = pane.shape
    alpha, F = np.zeros((H, W)), np.zeros((H, W, 3))
    for kind in rng.choice(families, int(rng.integers(1, 3)), replace=False):
        a, f = _layer(rng, kind, pane, H, W)
        new = a + alpha * (1 - a)
        F = np.where(new[..., None] > 1e-6, (a[..., None] * f + (alpha * (1 - a))[..., None] * F)
                     / np.maximum(new, 1e-6)[..., None], 0)
        alpha = new
    return alpha, F


def surface_family(rng, pane, kind):
    return _layer(rng, kind, pane, *pane.shape)
