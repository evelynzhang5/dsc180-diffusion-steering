# Round 12 evaluation fixes on round 10's smoke images (4 scenes with a subject; diagnostic)

| check | result | detail |
|---|---|---|
| identity: reference vs itself gives 0 changes, SSIM 1 | PASS | SSIM 1.000000 |
| every subject mask non-degenerate (area 0.5%-80%) | FAIL | 0.190, 0.089, 0.197, 0.000, 0.119 |
| post-hoc subject blur sigma 3: background blurs, subject interior unchanged | PASS | -5.77, -3.88, -6.84, -6.53 |
| uniform blur sigma 2 blurs the subject interior (< -0.2) | PASS | -3.82, -3.49, -4.15, -4.92 |
| copy-back keeps the subject interior exactly (RMS < 1e-3) | PASS | 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00 |
| contact sheet <= 25 images | PASS | 20 |

Subject masks from CLIPSeg on the unedited reference. Every method compared at the same achieved background change (interpolated along its own strength grid, per scene); scenes where a method never reaches the target count against its achievement. 'vs uniform' = subject interior sharpness change minus that of a uniform blur reaching the same background change on the same scene (a matched control, not a constant offset). Strength: steering s for global / masked (round 10 depth mask), blur sigma (px) for post-hoc / uniform. `evalfix_examples.png` columns: CLIPSeg subject mask, depth mask (nearest 40%), masked s 4, its copy-back, post-hoc blur sigma 1.

## Target background change -0.28

| method | reached | strength | bg | subject interior sharpness | subject RMS | interior RMS | subject SSIM | subject DINOv2 | boundary err | DINOv2 | vs uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global | 3/4 | 2.29 | -0.280 | -0.143 | 0.1694 | 0.1704 | 0.525 | 0.810 | 0.1104 | 0.803 | +0.112 |
| masked | 3/4 | 1.42 | -0.280 | +0.014 | 0.1387 | 0.1387 | 0.611 | 0.869 | 0.0911 | 0.882 | +0.268 |
| copyback_global | 3/4 | 2.29 | -0.280 | +0.000 | 0.0013 | 0.0000 | 0.998 | 0.990 | 0.0534 | 0.946 | +0.254 |
| copyback_masked | 3/4 | 1.42 | -0.280 | +0.000 | 0.0009 | 0.0000 | 0.998 | 0.992 | 0.0458 | 0.977 | +0.254 |
| posthoc_subject | 4/4 | 0.34 | -0.280 | +0.000 | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.0036 | 0.999 | +0.278 |
| uniform | 4/4 | 0.34 | -0.280 | -0.278 | 0.0041 | 0.0036 | 0.996 | 0.998 | 0.0052 | 1.000 | +0.000 |

## Target background change -0.5

| method | reached | strength | bg | subject interior sharpness | subject RMS | interior RMS | subject SSIM | subject DINOv2 | boundary err | DINOv2 | vs uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global | 2/4 | 2.85 | -0.500 | -0.447 | 0.1974 | 0.2028 | 0.432 | 0.838 | 0.1196 | 0.798 | +0.009 |
| masked | 3/4 | 2.38 | -0.500 | +0.013 | 0.1731 | 0.1730 | 0.521 | 0.814 | 0.1173 | 0.836 | +0.467 |
| copyback_global | 2/4 | 2.85 | -0.500 | +0.000 | 0.0012 | 0.0000 | 0.998 | 0.985 | 0.0596 | 0.970 | +0.456 |
| copyback_masked | 3/4 | 2.38 | -0.500 | +0.000 | 0.0012 | 0.0000 | 0.998 | 0.979 | 0.0592 | 0.960 | +0.454 |
| posthoc_subject | 4/4 | 0.42 | -0.500 | +0.000 | 0.0001 | 0.0000 | 1.000 | 0.999 | 0.0058 | 0.999 | +0.496 |
| uniform | 4/4 | 0.42 | -0.500 | -0.496 | 0.0074 | 0.0064 | 0.993 | 0.997 | 0.0085 | 1.000 | +0.000 |

## Mask convention at s 4 (subject sharpness change minus background change)

| scene | method | depth mask (nearest 40%) | CLIPSeg subject mask |
|---|---|---|---|
| 0 | global | -0.643 | +0.186 |
| 0 | masked | +0.887 | +0.620 |
| 1 | global | -1.025 | -1.369 |
| 1 | masked | -0.103 | -0.908 |
| 3 | global | -0.428 | -1.438 |
| 3 | masked | +0.710 | +1.857 |
| 5 | global | -0.044 | +0.774 |
| 5 | masked | +1.187 | +0.575 |

## Subject masks

| scene | noun | area | IoU with depth mask | CLIPSeg contrast |
|---|---|---|---|---|
| 0 | dog | 0.190 | 0.35 | 0.84 |
| 1 | woman | 0.089 | 0.22 | 0.63 |
| 3 | child | 0.197 | 0.48 | 0.75 |
| 4 | bird | 0.000 | 0.00 | 0.00 |
| 5 | deer | 0.119 | 0.24 | 0.76 |

Total time 49 s.
