# Round 13: does delaying the edit preserve the scene at the same background blur? (7 scenes x seed 42; diagnostic with a stopping rule)

| check | result | detail |
|---|---|---|
| determinism: unsteered sampler twice gives identical images | PASS | 0 |
| determinism: steered sampler twice gives identical images | PASS | 0 |
| cached sampler reproduces SD.generate (mode none) | PASS | 0 |
| zero edit with the third branch matches unsteered (numerics only) | PASS | max 1, mean 0.00059 |
| windows: early, middle, late have equal length | PASS | full 50, early 16, middle 16, late 16 |
| every scene has a subject mask (bird included) | PASS | 0: dog; 1: woman; 3: child; 4: small bird; 5: deer; 6: cat; 7: man riding a bicycle |
| all measures finite | PASS |  |
| contact sheets <= 25 images | PASS | 25, 7 |


Windows (DDIM step indices of 50, sigma at first and last step):

| window | steps | n | sigma range |
|---|---|---|---|
| full | 0-49 | 50 | 13.120 - 0.041 |
| early | 0-15 | 16 | 13.120 - 3.093 |
| middle | 17-32 | 16 | 2.655 - 0.958 |
| late | 34-49 | 16 | 0.836 - 0.041 |

Determinism (max absolute pixel difference, 0-255): unsteered_twice 0, vs_sd_generate 0, zero_edit_third_branch 1, zero_edit_mean 0.00059, steered_twice 0

## Target background change -0.5

| method | reached | strength c (median) | subject RMS | interior RMS | subject SSIM | subject DINOv2 | boundary err | subject-mask IoU | centre shift (px) |
|---|---|---|---|---|---|---|---|---|---|
| full | 7/7 | 0.54 | 0.1847 | 0.1823 | 0.470 | 0.552 | 0.1447 | 0.48 | 20.9 |
| early | 6/7 | 1.55 | 0.1814 | 0.1801 | 0.451 | 0.532 | 0.1416 | 0.51 | 37.3 |
| middle | 5/7 | 4.00 | 0.1443 | 0.1395 | 0.566 | 0.501 | 0.1005 | 0.58 | 81.5 |
| late | 1/7 | 9.39 | 0.0874 | 0.0805 | 0.643 | 0.345 | 0.0806 | 0.81 | 8.0 |
| posthoc | 3/7 | 0.40 | 0.0001 | 0.0000 | 1.000 | 0.997 | 0.0030 | 1.00 | 0.0 |

Per scene (subject RMS / IoU; * = target not reached):

| scene | full | early | middle | late | posthoc |
|---|---|---|---|---|---|
| 0 dog | 0.196 / 0.66 | 0.262 / 0.24 | 0.182 / 0.73 | 0.123 / 0.90* | 0.000 / 1.00* |
| 1 woman | 0.148 / 0.00 | 0.130 / 0.89 | 0.101 / 0.00 | 0.054 / 0.95* | 0.000 / 1.00 |
| 3 child | 0.057 / 0.98 | 0.067 / 0.98 | 0.109 / 0.94 | 0.030 / 0.98* | 0.000 / 1.00* |
| 4 small bird | 0.099 / 0.18 | 0.107 / 0.17 | 0.170 / 0.00* | 0.149 / 0.00* | 0.001 / 1.00* |
| 5 deer | 0.282 / 0.53 | 0.252 / 0.34 | 0.143 / 0.83* | 0.122 / 0.91* | 0.000 / 1.00* |
| 6 cat | 0.238 / 0.53 | 0.230 / 0.33* | 0.153 / 0.46 | 0.087 / 0.81 | 0.000 / 1.00 |
| 7 man riding a bicycle | 0.274 / 0.48 | 0.269 / 0.47 | 0.177 / 0.75 | 0.051 / 0.98* | 0.000 / 1.00 |

## Target background change -2.0

| method | reached | strength c (median) | subject RMS | interior RMS | subject SSIM | subject DINOv2 | boundary err | subject-mask IoU | centre shift (px) |
|---|---|---|---|---|---|---|---|---|---|
| full | 2/7 | 4.85 | 0.2072 | 0.2091 | 0.350 | 0.256 | 0.1736 | 0.04 | 33.3 |
| early | 1/7 | 5.19 | 0.1954 | 0.1937 | 0.443 | 0.663 | 0.1771 | 0.69 | 5.9 |
| middle | 1/7 | 8.00 | 0.1622 | 0.1632 | 0.521 | 0.486 | 0.1161 | 0.52 | 46.0 |
| late | 0/7 | 8.00 | nan | nan | nan | nan | nan | nan | nan |
| posthoc | 3/7 | 0.80 | 0.0003 | 0.0000 | 1.000 | 0.981 | 0.0150 | 1.00 | 0.0 |

Per scene (subject RMS / IoU; * = target not reached):

| scene | full | early | middle | late | posthoc |
|---|---|---|---|---|---|
| 0 dog | 0.245 / 0.23* | 0.264 / 0.24* | 0.165 / 0.88* | 0.126 / 0.91* | 0.000 / 1.00 |
| 1 woman | 0.195 / 0.00* | 0.166 / 0.88* | 0.110 / 0.93* | 0.055 / 0.00* | 0.000 / 1.00 |
| 3 child | 0.192 / 0.00 | 0.195 / 0.69 | 0.077 / 0.97* | 0.047 / 0.97* | 0.000 / 1.00* |
| 4 small bird | 0.127 / 0.00* | 0.133 / 0.00* | 0.142 / 0.00* | 0.089 / 0.00* | 0.002 / 1.00* |
| 5 deer | 0.244 / 0.32* | 0.323 / 0.00* | 0.115 / 0.93* | 0.134 / 0.87* | 0.000 / 1.00* |
| 6 cat | 0.234 / 0.00* | 0.301 / 0.46* | 0.162 / 0.52 | 0.116 / 0.84* | 0.000 / 1.00 |
| 7 man riding a bicycle | 0.223 / 0.09 | 0.190 / 0.86* | 0.164 / 0.86* | 0.080 / 0.97* | 0.000 / 1.00* |

## Decision

- Target -0.5: best later window `middle`; reached 5/7; subject RMS <= 0.8 x full window and both on target in 3/7; preserved (ignoring the target) in 4/7. **Row 3: later editing reaches the target but still changes pose/layout; stop this direction-and-layer approach.** Post-hoc blur better on subject and boundary than every window that reached the target: yes.
- Target -2.0: best later window `middle`; reached 1/7; subject RMS <= 0.8 x full window and both on target in 0/7; preserved (ignoring the target) in 6/7. **Row 2: later editing preserves the subject but cannot reach the target; no demonstrated improvement.** Post-hoc blur better on subject and boundary than every window that reached the target: yes.
- **Row 4: post-processing remains better at both blur levels: the task does not justify a generative steering method.**

Six (seven) scenes decide only whether to continue; any promising setting must then hold on untouched scenes and seeds. `timing_examples.png` columns (target -2.0): subject mask, full, early, late, post-hoc. `timing_masks.png`: every subject mask used.

Total time 3111 s.
