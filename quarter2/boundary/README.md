# Round 39: boundaries. Step 0: validate the reference renderer (CPU)

**Question for the round.** Can depth-informed steering correct boundary errors that remain after blur calibration, while preserving the rest of the image? The first task is to establish a repeatable boundary error that request adjustment cannot fix. That needs a validated reference first.

**Renderers (`lens39.py`, all in linear light).**
- **Scene model:** fronto-parallel layers with colour F, opacity α and disparity d; a thin lens focused at d_f.
- **Geometry:** a ray for image point x through aperture point a meets layer l at x + a·c_l, with signed blur radius c_l = NOM·K·(d_l − d_f). For planar layers given in projected image coordinates, this is exact thin-lens geometry.
- **raytrace (the reference):** aperture-sampled, 1024 stratified samples. Layers are composited front to back per sample, nearest first, with opacity as transmission.
- **composite:** analytic premultiplied layering, I = K_f*(αF) + (1 − K_f*α)·(K_b*B).
- **naive:** blurs colour and opacity separately (non-premultiplied); for comparison only.

**Validation (`validate39.py`).**
- **Scene:** a textured, antialiased disk foreground (radius 140, d = 0.7) over a background (d = 0.3).
- **Backgrounds:** uniform, gradient, texture.
- **Focus settings:**
  - A: foreground in focus;
  - B: background in focus;
  - C: focus between, both layers blurred, 7.75 px each.
- **Measure:** mean |difference| to the ray tracer (0–255), in bands of distance to the edge.

| Comparison | A (fg sharp) | B (fg blurred) | C (both blurred) |
|---|---|---|---|
| Sampled kernel vs analytic disk (interior) | 0.01–0.03 | | |
| composite | **0** (all backgrounds) | **0** | 0 uniform; 0.17–0.19 gradient; **2.2–2.5 textured**, within 8 px of the edge |
| naive (non-premultiplied) | 0 | 1.8–5.8 at the edge | 2.5–4.1 at the edge |
| **hidden**: background behind the foreground replaced by the visible all-in-focus image | **18–22** outside the edge (0–8 px), 5–6 at 8–16 px | **22–23** inside the edge (0–8 px), 5 at 8–16 px | 13–15 on both sides |
| composite in **sRGB** instead of linear light | 0.3–0.6 at the edge (3–4.5 everywhere on texture) | **7.7–13** at and inside the blurred foreground | 5–11 |

**Findings.**
1. **The premultiplied composite equals the ray tracer whenever one layer is in focus.** When both layers are blurred it is exact only on a uniform background, and is off by up to 2.5/255 at textured edges. **The ray tracer is the reference.**
2. **Hidden background content dominates edge differences, at 18–23/255.**
   - When the background is blurred (A), pixels just outside the edge gather background that the foreground hides.
   - When a blurred foreground becomes partly transparent (B), pixels just inside the edge reveal it.

   BokehDiff sees only the all-in-focus image, so near edges it cannot know that content. Boundary errors must be judged against an **available-information** reference, in which the background behind the foreground is estimated from visible pixels, *as well as* the true reference. Uniform and gradient backgrounds make that estimate trivial, so start there, as the review advised.
3. **Linear light matters:** an sRGB composite is off by 7–13/255 wherever blur occurs. This agrees with round 38's finding that BokehDiff matches linear-light blur.

**Next (step 1, GPU; not yet run).** Foreground cutouts (antialiased shapes with texture) over uniform or gradient backgrounds, with known disparities:
- **Settings:** A (foreground sharp, background blurred), B (the reverse), and C.
- **Requests:** calibrated.
- **References:** the ray tracer, and the available-information version.
- **Measure:**
  - error by edge band, against the interior error;
  - halo or bleed direction and width;
  - repeatability across textures, shapes and positions;
  - whether adjusting the request (globally or near the edge) removes it.

  A boundary error qualifies only if it is repeatable, larger than the interior error, and not removed by request adjustment.

Files: `lens39.py`, `validate39.py`, `results/validate39.json`, `results/validate39_log.txt`. The reference renders `results/ref_*.npy` are local only.
