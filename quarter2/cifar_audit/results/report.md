# Q2 round 5: predicting steering side effects on new seeds (CIFAR-10, RFM-only)

Predictions frozen before the test images were generated (sha256 3940dc509f323bc6ca0e3b2364979fa395df7cf3c988c445713871877ddad33f).

## Confirmed side effects on the test seeds

Steered images assigned to the class vs. the model's own images of the class; confirmed if |effect| > 0.25 SD of real class images (set meters: |ratio - 1| > 0.1) and the 95% interval excludes 0. Generic: the random direction moves the meter the same way by at least half as much.

| class | steered accuracy | steered / native images | confirmed | of which generic |
|---|---|---|---|---|
| airplane | 97.3% | 498 / 691 | brightness, contrast, blueness, redness, feature_diversity | none |
| cat | 99.4% | 509 / 692 | border_clutter, sharpness, feature_diversity | none |
| deer | 97.9% | 501 / 624 | contrast, blueness, greenness, redness, figure_ground, sharpness, feature_diversity, background_diversity, pixel_diversity | none |
| ship | 91.2% | 467 / 515 | blueness, redness, sharpness, feature_diversity | feature_diversity |

## How well each prediction source did (summed over classes)

| source | flagged | hits | false alarms | missed | hits that are not generic |
|---|---|---|---|---|---|
| OUT_32 | 17 | 13 | 4 | 8 | 13 |
| OUT_256 | 23 | 19 | 4 | 2 | 18 |
| BOTH_32 | 17 | 10 | 7 | 11 | 9 |
| BOTH_256 | 22 | 15 | 7 | 6 | 14 |
| TRANSFER | 11 | 10 | 1 | 11 | 9 |
| REF | 0 | 0 | 0 | 21 | 0 |

Effect-size agreement over the 10 per-image meters x classes (Spearman, sign agreement, median |error| in SDs):

- OUT_32: +0.90, 85%, 0.10
- OUT_256: +0.96, 88%, 0.05
- BOTH_32: +0.77, 75%, 0.11
- BOTH_256: +0.88, 88%, 0.08
- TRANSFER: -0.15, 35%, 0.23

## Effects per class (test truth, with the random-direction shift)

### airplane

| meter | true effect (95% interval) | OUT_32 | OUT_256 | BOTH_256 | TRANSFER | random |
|---|---|---|---|---|---|---|
| brightness | **-0.42** (-0.53, -0.32) | -0.47 | -0.47 | -0.39 | -0.00 | +0.00 |
| contrast | **-0.28** (-0.40, -0.17) | -0.24 | -0.22 | -0.29 | +0.06 | -0.07 |
| saturation | +0.22 (+0.10, +0.33) | +0.09 | +0.16 | -0.00 | -0.12 | -0.05 |
| colorfulness | +0.05 (-0.06, +0.14) | +0.02 | +0.08 | -0.08 | -0.01 | -0.02 |
| blueness | **-0.33** (-0.46, -0.23) | -0.27 | -0.36 | -0.41 | -0.04 | -0.04 |
| greenness | +0.12 (-0.02, +0.26) | -0.08 | -0.01 | +0.06 | -0.08 | -0.05 |
| redness | **+0.31** (+0.21, +0.42) | +0.32 | +0.38 | +0.42 | +0.08 | +0.06 |
| border_clutter | -0.21 (-0.31, -0.10) | -0.14 | -0.16 | -0.13 | +0.08 | -0.04 |
| figure_ground | -0.10 (-0.19, +0.01) | -0.13 | -0.07 | -0.20 | +0.03 | +0.00 |
| sharpness | +0.21 (+0.10, +0.32) | +0.27 | +0.41 | -0.02 | +0.55 | -0.04 |
| feature_diversity | **-0.38** (-0.47, -0.37) | -0.43 | -0.36 | +0.43 | -0.51 | -0.13 |
| background_diversity | +0.02 (-0.05, +0.08) | -0.14 | -0.10 | -0.18 | -0.04 | +0.00 |
| pixel_diversity | -0.04 (-0.09, -0.01) | -0.08 | -0.10 | -0.13 | -0.00 | +0.02 |

### cat

| meter | true effect (95% interval) | OUT_32 | OUT_256 | BOTH_256 | TRANSFER | random |
|---|---|---|---|---|---|---|
| brightness | +0.07 (-0.02, +0.18) | +0.09 | +0.07 | +0.06 | -0.18 | +0.00 |
| contrast | -0.01 (-0.11, +0.10) | +0.13 | +0.02 | -0.10 | -0.01 | -0.08 |
| saturation | -0.02 (-0.12, +0.08) | -0.18 | -0.15 | -0.18 | -0.01 | -0.03 |
| colorfulness | -0.00 (-0.09, +0.08) | -0.21 | -0.13 | -0.17 | +0.05 | +0.01 |
| blueness | -0.11 (-0.22, -0.01) | +0.10 | +0.03 | -0.04 | -0.17 | -0.02 |
| greenness | +0.06 (-0.04, +0.15) | +0.02 | +0.04 | +0.04 | -0.09 | -0.02 |
| redness | +0.08 (-0.03, +0.18) | -0.10 | -0.05 | +0.02 | +0.22 | +0.03 |
| border_clutter | **+0.25** (+0.15, +0.35) | +0.46 | +0.25 | +0.18 | -0.05 | -0.07 |
| figure_ground | -0.13 (-0.23, -0.03) | -0.13 | -0.15 | -0.09 | +0.06 | +0.00 |
| sharpness | **+0.77** (+0.66, +0.89) | +0.51 | +0.76 | +0.38 | +0.43 | -0.15 |
| feature_diversity | **-0.88** (-0.89, -0.87) | -0.86 | -0.86 | -0.72 | -0.35 | -0.12 |
| background_diversity | -0.13 (-0.14, +0.01) | -0.26 | -0.17 | -0.21 | -0.01 | +0.07 |
| pixel_diversity | -0.04 (-0.08, -0.00) | -0.03 | -0.07 | -0.09 | -0.01 | -0.01 |

### deer

| meter | true effect (95% interval) | OUT_32 | OUT_256 | BOTH_256 | TRANSFER | random |
|---|---|---|---|---|---|---|
| brightness | +0.23 (+0.11, +0.36) | +0.23 | +0.24 | +0.18 | -0.24 | +0.00 |
| contrast | **+0.32** (+0.20, +0.44) | +0.30 | +0.27 | +0.21 | -0.10 | -0.12 |
| saturation | -0.12 (-0.23, -0.01) | -0.45 | -0.31 | -0.34 | +0.04 | -0.06 |
| colorfulness | +0.04 (-0.05, +0.15) | -0.28 | -0.11 | -0.26 | +0.05 | -0.05 |
| blueness | **+0.41** (+0.29, +0.53) | +0.59 | +0.47 | +0.44 | -0.32 | -0.02 |
| greenness | **-0.33** (-0.43, -0.22) | -0.48 | -0.38 | -0.40 | +0.05 | +0.00 |
| redness | **-0.25** (-0.40, -0.12) | -0.35 | -0.28 | -0.23 | +0.30 | +0.03 |
| border_clutter | +0.13 (+0.01, +0.23) | +0.20 | +0.10 | +0.08 | -0.00 | -0.07 |
| figure_ground | **+0.49** (+0.35, +0.61) | +0.23 | +0.30 | +0.32 | -0.09 | -0.10 |
| sharpness | **+0.31** (+0.20, +0.43) | +0.13 | +0.25 | -0.19 | +0.60 | -0.07 |
| feature_diversity | **-0.56** (-0.58, -0.50) | -0.55 | -0.53 | -0.03 | -0.46 | -0.11 |
| background_diversity | **+0.37** (+0.27, +0.48) | +0.08 | +0.16 | +0.12 | -0.12 | -0.02 |
| pixel_diversity | **+0.20** (+0.14, +0.22) | +0.12 | +0.09 | +0.09 | -0.06 | -0.02 |

### ship

| meter | true effect (95% interval) | OUT_32 | OUT_256 | BOTH_256 | TRANSFER | random |
|---|---|---|---|---|---|---|
| brightness | -0.21 (-0.33, -0.09) | -0.42 | -0.31 | -0.32 | -0.06 | +0.02 |
| contrast | -0.07 (-0.19, +0.06) | -0.12 | -0.10 | -0.09 | +0.02 | -0.07 |
| saturation | +0.09 (-0.02, +0.21) | +0.10 | +0.11 | +0.01 | -0.10 | -0.05 |
| colorfulness | +0.12 (+0.01, +0.23) | +0.17 | +0.20 | +0.04 | -0.05 | +0.00 |
| blueness | **-0.68** (-0.81, -0.55) | -0.57 | -0.63 | -0.66 | +0.05 | -0.06 |
| greenness | +0.13 (+0.00, +0.25) | +0.11 | +0.11 | +0.05 | -0.12 | -0.06 |
| redness | **+0.61** (+0.49, +0.73) | +0.51 | +0.57 | +0.63 | +0.02 | +0.08 |
| border_clutter | -0.09 (-0.22, +0.03) | +0.03 | -0.10 | -0.05 | +0.06 | -0.03 |
| figure_ground | -0.01 (-0.13, +0.10) | -0.07 | -0.05 | -0.01 | +0.03 | -0.07 |
| sharpness | **+0.70** (+0.59, +0.82) | +0.61 | +0.63 | +0.38 | +0.48 | -0.05 |
| feature_diversity | **-0.20** (-0.27, -0.14) | -0.12 | -0.15 | +0.33 | -0.58 | -0.13 |
| background_diversity | +0.05 (-0.03, +0.10) | -0.23 | -0.09 | -0.12 | -0.04 | -0.02 |
| pixel_diversity | +0.06 (+0.01, +0.07) | -0.02 | -0.02 | -0.01 | -0.03 | +0.01 |

