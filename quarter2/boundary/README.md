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

## Step 0b (CPU, saved outputs): calibration and radius validity under a linear-light estimator (`calib_linear.py`)

The radius estimator's disk stack is rebuilt in linear light, and everything that depends on it is refitted.

**Checks.**
- The sRGB refit reproduces round 25's frozen calibration curve (max 0.01 px).
- It also reproduces round 36's 66/72 valid test cells.

| | sRGB estimator | Linear estimator |
|---|---|---|
| Round 25 curve (calibrated request for target 3 / 8 / 11.6 / 15.5 px) | 5.55 / 8.25 / 9.85 / 11.47 | 5.61 / 8.27 / 9.84 / 11.67 |
| Curve value at request 16 / 19 | 24.9 / 26.2 | 23.5 / 24.9 |
| Round 36 test cells within tolerance | 66/72 | 67/72 (65 under both) |
| r̂ linear − r̂ sRGB, round 36 test cells | | median −0.39 px, max \|·\| 2.0 px |

**Reading.**
- Over the tested targets (3–15.5 px), calibrated requests change by at most 0.2 px. Validity changes in a few cells: one more valid overall, 2 lost, 3 gained.
- Radius results are therefore **nearly**, not entirely, reference-independent. A single cell's radius can move by up to 2 px.
- The linear curve is the one frozen for step 1.

## Step 1 (GPU): plan, frozen before generation

**Question.** Does BokehDiff leave a repeatable boundary error that request adjustment cannot remove, while interior blur stays valid?

**Scenes (`scenes39.py`).** 24 scenes: a foreground cutout over a uniform or gradient background.
- **Foreground:** disk, ellipse or rounded square, filled with gravel, astronaut, chelsea or rocket texture, at disparity 1.
- **Background:** uniform or gradient, at disparity 0.
- **Input:** composited in linear light.
- **Background reconstruction:** rebuilt from visible pixels only, it matches the true background to ≤ 0.01/255. On these scenes the available-information reference therefore equals the true one, and check V4 below confirms this.

**Settings.**
- **A:** foreground sharp, background blurred to r.
- **B:** background sharp, foreground blurred to r.
- **C:** both layers blurred, each to r/2.
- **Levels:** r = 15.5 (L1) and 11.625 px (L2).
- **Requests:** from the frozen linear-light curve.

**Reference.** The aperture-sampled thin-lens ray tracer (1024 samples, linear light), on the GPU (`lens39.raytrace_torch`).

**Checks before any result counts (`run39.py`, stage V).**
- V1: the GPU ray tracer equals the numpy one (≤ 1e-3).
- V2: repeating a generation gives an identical image.
- V3: 1024 vs 4096 aperture samples differ by ≤ 0.1/255 in every edge band (6 scenes × A/B/C).
- V4: the available-information reference matches the true one to ≤ 0.1/255.

**Request adjustment.** Per scene and cell:
- **Global multiplier** on every blurred pixel's request: 0.6–1.4 in steps of 0.1.
- **Edge multiplier:** at the best valid global multiplier, the request within 16 px of the edge is multiplied by 0, 0.5, 0.75, 1.25, 1.5 or 2.

That is 15 generations per scene and cell, 2160 in total.

**Measurement (`measure39.py`).**
- Each side's own interior offset is removed before scoring.
- **Band error:** the mean |difference| to the reference in distance bands.
- **Excess E:** the maximum over the bands out8–16, out0–8, in0–8 and in8–16 of (band error − that side's interior error).
- **Signed luminance profile:** reported against distance from the edge.

**Validity.**
- A blurred foreground's linear-estimator radius must be within max(2 px, 20 %) of its target.
- A sharp layer's interior must stay within 3.0/255 of the input.
- A blurred plain background's interior must be within 3.0/255 of the reference.

**Decision (`analyze39.py`).**
- After the best valid adjustment, a scene shows a boundary error if E ≥ 2.0/255 and that band's error is ≥ 1.5× its side's interior error.
- A cell is **ESTABLISHED** if ≥ 18 of 24 scenes show one.
- The round is ESTABLISHED if any cell is. Otherwise no repeatable boundary error survives request adjustment on these scenes, and boundary steering is not needed for them.

**Changes after the smoke run (before any full-run data).**
- **Checks:** all passed. GPU vs numpy ray tracer 6e-14; repeated generation identical; 1024 vs 4096 samples ≤ 0.021/255 in every edge band; available-information reference identical.
- **Focus validity is now judged against BokehDiff's own zero-request output, as round 25's rule did.** With zero blur requested, BokehDiff's floor alone moves the gravel foreground 14/255 from the input, so the input-referenced 3.0 limit rejected every setting-A request. One zero-request generation per scene is added. The change relative to the input is still recorded.
- **The distance profile skips empty bins.** No pixel lies at distance 0, so that bin was empty and crashed the analysis.
- **Bokehlicious runs on torch 2.5.1+cu121.** The pod's GTX 1080 Ti (sm_61) has no kernels in current torch wheels.

**Second baseline: Bokehlicious, setting A only (`bkl39.py`).**
- **Model:** small checkpoint, commit `0d2c458`.
- **Why A only:** the model has no depth or focus input, so only setting A is well posed.
- **Adjustment:** the f-stop, from {2, 2.8, 4, 5.6, 8, 11, 16}, using the same inputs, references and measurement.
- **Latency:** recorded.
- It is reported alongside the decision, not used in it.

## Step 1 results (pod `q2-r39b`, Nov 22; 2,186 BokehDiff generations, 72 min, plus Bokehlicious)

**Checks: all passed.**
- GPU vs numpy ray tracer: 6e-14.
- Repeated generation: identical.
- 1024 vs 4096 aperture samples: ≤ 0.025/255 in every edge band.
- Available-information vs true reference: ≤ 0.002/255.

**Decision (frozen rule): NOT ESTABLISHED.** No cell reaches 18/24 scenes.

| Cell | Scenes with a boundary error after adjustment | No valid request | Median E: base → adjusted | Worst band (count) | Usual best adjustment |
|---|---|---|---|---|---|
| A L1 | 5/24 | 11 | 1.92 → 1.64 | out0-8 (8) | global ×0.6 |
| A L2 | 4/24 | 11 | 1.96 → 1.55 | out0-8 (7) | global ×0.6 |
| B L1 | 11/24 | 0 | 2.91 → 1.98 | **out0-8 (22)** | global ×1.1–1.2 |
| B L2 | **16/24** | 0 | 3.06 → 2.17 | **out0-8 (21)** | global ×1.1–1.2 |
| C L1 | 12/24 | 0 | 2.22 → 1.88 | out0-8 (23) | none (×1.0) |
| C L2 | 11/24 | 0 | 2.21 → 1.89 | out0-8 (21) | none |

**What the errors look like.**
- **B (blurred foreground over a sharp background).**
  - **Location:** the remaining error sits in the sharp background just outside the edge (0–8 px).
  - **Profile:** a positive (brighter) luminance residual of +0.4 to +1.6/255 from about −16 to +12 px.
  - **Adjustment:** a 10–20 % larger global request is usually best, and edge-only adjustment rarely helps.
  - **Where it qualifies:** in 10–11 of the 12 uniform-background scenes but only 1–5 of the 12 gradient ones. A uniform background has almost no interior error, so its edge excess is not diluted.
- **C (both layers blurred):** the same location (out0-8) and a smaller brightening of +0.3 to +1.2. Request adjustment almost never helps.
- **A (sharp foreground over a blurred background) is degenerate in this design.**
  - **Why:** a blurred uniform background is still uniform, and a blurred linear gradient is nearly unchanged. With the foreground sharp and in front, the correct thin-lens output is essentially the input. A therefore tests only whether the renderer leaves the scene alone. It says nothing about rendering a blurred boundary.
  - **Rim residual:** BokehDiff leaves a +3.1/255 luminance rim 1 px inside the foreground edge.
  - **Foreground change:** requesting background blur alters the sharp foreground relative to the zero-request output (median 5.2 gravel, 4.3 astronaut, 2.6 chelsea, 1.9 rocket). That makes 11/24 scenes invalid.
  - **Status:** I missed this degeneracy when designing the round. Setting A needs a textured background to be meaningful.

**Bokehlicious (setting A; small model; 0.08 s per image against BokehDiff's ~2 s per generation).**
- Because A's correct answer is about the input itself, the best valid f-stop was usually f/11–16, which barely changes the image. That gives E ≈ 0.3 and 0/24 qualifying, identical at both levels.
- This **does not** show better boundary rendering. It shows only that the weakest aperture leaves these images nearly unchanged.
- **Foreground change with aperture:**
  - Stronger apertures alter the sharp foreground progressively: median change to the input 2.1 at f/2, 1.4 at f/4, 0.9 at f/8.
  - Requests are invalid in 7/24 scenes at f/2, all gravel and astronaut.
  - On the gravel cutouts it blurs the "subject" heavily: 21/255 at f/2. Its own focus inference does not treat a high-frequency texture cutout as the subject.

**Reading.**
1. **Inconclusive, not a negative.** The required consistency (≥ 18/24) was not reached, but the residual did not disappear: B at r = 11.6 px still qualifies in 16/24 scenes. Most of B's error is a mild brightening just outside a blurred foreground over a flat background.
   - **The "adjusted" request is an oracle.** It was chosen per scene using the reference image. That shows a suitable request *exists*, not that the frozen calibration, or any method without the reference, would choose it on a new image.
   - An open question that fits a predictor: can a spatial model choose the request adjustment without seeing the correct output? Its first comparison should be against the existing calibration. Activation editing would then have to add something beyond that.
2. **The near-miss is specific:** blurred-foreground edges over flat backgrounds, error 1.9–2.5/255 after adjustment. If boundary work continues, that is the case to strengthen. It would need more scenes and textured backgrounds, which require the visible-pixel reconstruction to be validated first.
3. **Setting A must be redesigned with a textured background** before it can test anything, for BokehDiff or for Bokehlicious.
4. **A side finding:** BokehDiff changes a sharp foreground by 2–5/255 when only the background's blur is requested. This is the same image-wide leak seen in rounds 35–37.

Files: `results/boundary39.json` (analysis), `calib_linear.json`. Local only (`outputs/q2_boundary/`): `conds.json`, `bkl.json`, logs, inputs, base and zero-request images, and Bokehlicious outputs. All images and references are on the DSMLP workspace.
