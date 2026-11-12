# Same-seed check of the side effects (test seeds 3000-3511)

KEPT: seeds whose unguided image is already the class and whose steered image still is, so the class does not change; effect = steered - unguided on the same seed, in SDs of real class images. ALL: every seed whose steered image is the class (includes the class change). SETS: the pre-registered steered vs. native comparison. Bold: |effect| > 0.25 and the 95% interval excludes 0. Image meters only.

## airplane: 82 seeds keep the class, 498 steered images are the class

| meter | SETS (pre-registered) | KEPT | ALL | RANDOM on KEPT seeds |
|---|---|---|---|---|
| brightness | **-0.42 (-0.53, -0.32)** (confirmed) | +0.05 (+0.03, +0.08) | +0.05 (+0.04, +0.06) | -0.01 (-0.02, +0.01) |
| contrast | **-0.28 (-0.40, -0.17)** (confirmed) | **-0.29 (-0.37, -0.20)** | **-0.41 (-0.45, -0.38)** | -0.02 (-0.07, +0.04) |
| saturation | +0.22 (+0.10, +0.33) | +0.22 (+0.11, +0.32) | -0.02 (-0.07, +0.03) | -0.07 (-0.13, -0.01) |
| colorfulness | +0.05 (-0.06, +0.14) | +0.16 (+0.03, +0.29) | +0.06 (+0.01, +0.10) | +0.01 (-0.07, +0.09) |
| blueness | **-0.33 (-0.46, -0.23)** (confirmed) | **+0.30 (+0.24, +0.36)** | **+0.33 (+0.30, +0.36)** | -0.14 (-0.17, -0.11) |
| greenness | +0.12 (-0.02, +0.26) | -0.20 (-0.28, -0.13) | -0.14 (-0.18, -0.10) | +0.01 (-0.04, +0.05) |
| redness | **+0.31 (+0.21, +0.42)** (confirmed) | **-0.25 (-0.32, -0.19)** | **-0.30 (-0.33, -0.28)** | +0.15 (+0.12, +0.17) |
| border_clutter | -0.21 (-0.31, -0.10) | **-0.29 (-0.40, -0.18)** | **-0.46 (-0.50, -0.42)** | +0.03 (-0.02, +0.09) |
| figure_ground | -0.10 (-0.19, +0.01) | -0.14 (-0.23, -0.04) | -0.24 (-0.28, -0.19) | +0.06 (+0.00, +0.13) |
| sharpness | +0.21 (+0.10, +0.32) | -0.10 (-0.23, +0.03) | -0.20 (-0.28, -0.12) | +0.08 (-0.01, +0.18) |

## cat: 69 seeds keep the class, 509 steered images are the class

| meter | SETS (pre-registered) | KEPT | ALL | RANDOM on KEPT seeds |
|---|---|---|---|---|
| brightness | +0.07 (-0.02, +0.18) | -0.00 (-0.03, +0.02) | -0.09 (-0.10, -0.07) | -0.00 (-0.02, +0.02) |
| contrast | -0.01 (-0.11, +0.10) | **+0.25 (+0.15, +0.35)** | +0.15 (+0.11, +0.20) | -0.09 (-0.14, -0.04) |
| saturation | -0.02 (-0.12, +0.08) | +0.00 (-0.09, +0.08) | -0.18 (-0.22, -0.13) | +0.04 (-0.01, +0.08) |
| colorfulness | -0.00 (-0.09, +0.08) | +0.01 (-0.12, +0.13) | -0.15 (-0.20, -0.10) | +0.07 (-0.00, +0.14) |
| blueness | -0.11 (-0.22, -0.01) | -0.16 (-0.25, -0.08) | **-0.29 (-0.34, -0.24)** | +0.01 (-0.02, +0.03) |
| greenness | +0.06 (-0.04, +0.15) | -0.02 (-0.14, +0.11) | **-0.25 (-0.31, -0.19)** | +0.03 (-0.02, +0.09) |
| redness | +0.08 (-0.03, +0.18) | +0.17 (+0.09, +0.24) | **+0.41 (+0.37, +0.45)** | -0.02 (-0.05, +0.01) |
| border_clutter | **+0.25 (+0.15, +0.35)** (confirmed) | **+0.43 (+0.33, +0.54)** | **+0.43 (+0.38, +0.47)** | -0.11 (-0.16, -0.05) |
| figure_ground | -0.13 (-0.23, -0.03) | +0.02 (-0.09, +0.15) | -0.01 (-0.06, +0.05) | +0.00 (-0.05, +0.04) |
| sharpness | **+0.77 (+0.66, +0.89)** (confirmed) | **+0.73 (+0.56, +0.89)** | **+0.64 (+0.56, +0.71)** | -0.01 (-0.12, +0.09) |

## deer: 63 seeds keep the class, 501 steered images are the class

| meter | SETS (pre-registered) | KEPT | ALL | RANDOM on KEPT seeds |
|---|---|---|---|---|
| brightness | +0.23 (+0.11, +0.36) | -0.05 (-0.08, -0.03) | -0.03 (-0.04, -0.02) | -0.01 (-0.02, +0.01) |
| contrast | **+0.32 (+0.20, +0.44)** (confirmed) | -0.00 (-0.09, +0.08) | -0.22 (-0.26, -0.18) | -0.09 (-0.15, -0.03) |
| saturation | -0.12 (-0.23, -0.01) | +0.03 (-0.03, +0.08) | +0.05 (+0.02, +0.09) | -0.09 (-0.17, -0.03) |
| colorfulness | +0.04 (-0.05, +0.15) | +0.01 (-0.09, +0.10) | +0.01 (-0.05, +0.07) | +0.03 (-0.06, +0.14) |
| blueness | **+0.41 (+0.29, +0.53)** (confirmed) | +0.00 (-0.04, +0.04) | -0.10 (-0.12, -0.08) | +0.05 (+0.02, +0.08) |
| greenness | **-0.33 (-0.43, -0.22)** (confirmed) | -0.03 (-0.09, +0.02) | +0.11 (+0.08, +0.13) | -0.10 (-0.15, -0.06) |
| redness | **-0.25 (-0.40, -0.12)** (confirmed) | +0.02 (-0.02, +0.06) | +0.04 (+0.01, +0.07) | +0.02 (-0.01, +0.05) |
| border_clutter | +0.13 (+0.01, +0.23) | -0.01 (-0.10, +0.07) | -0.17 (-0.20, -0.12) | -0.08 (-0.15, -0.02) |
| figure_ground | **+0.49 (+0.35, +0.61)** (confirmed) | -0.07 (-0.20, +0.05) | **-0.25 (-0.30, -0.21)** | -0.07 (-0.17, +0.01) |
| sharpness | **+0.31 (+0.20, +0.43)** (confirmed) | -0.04 (-0.15, +0.08) | -0.04 (-0.12, +0.04) | +0.05 (-0.05, +0.17) |

## ship: 45 seeds keep the class, 467 steered images are the class

| meter | SETS (pre-registered) | KEPT | ALL | RANDOM on KEPT seeds |
|---|---|---|---|---|
| brightness | -0.21 (-0.33, -0.09) | +0.07 (+0.04, +0.10) | +0.02 (+0.00, +0.04) | +0.02 (+0.00, +0.04) |
| contrast | -0.07 (-0.19, +0.06) | **+0.25 (+0.16, +0.36)** | +0.08 (+0.04, +0.13) | +0.00 (-0.08, +0.07) |
| saturation | +0.09 (-0.02, +0.21) | +0.07 (-0.04, +0.18) | **-0.26 (-0.32, -0.20)** | -0.12 (-0.24, -0.01) |
| colorfulness | +0.12 (+0.01, +0.23) | +0.24 (+0.11, +0.39) | +0.01 (-0.05, +0.07) | -0.04 (-0.19, +0.10) |
| blueness | **-0.68 (-0.81, -0.55)** (confirmed) | +0.14 (+0.05, +0.25) | **+0.52 (+0.47, +0.56)** | -0.23 (-0.29, -0.18) |
| greenness | +0.13 (+0.00, +0.25) | -0.11 (-0.19, -0.03) | **-0.25 (-0.33, -0.18)** | -0.05 (-0.14, +0.04) |
| redness | **+0.61 (+0.49, +0.73)** (confirmed) | -0.10 (-0.18, -0.02) | **-0.40 (-0.44, -0.37)** | +0.25 (+0.19, +0.31) |
| border_clutter | -0.09 (-0.22, +0.03) | +0.17 (+0.06, +0.29) | +0.10 (+0.05, +0.15) | +0.07 (-0.01, +0.16) |
| figure_ground | -0.01 (-0.13, +0.10) | -0.03 (-0.11, +0.06) | -0.23 (-0.28, -0.17) | -0.04 (-0.10, +0.02) |
| sharpness | **+0.70 (+0.59, +0.82)** (confirmed) | **+0.50 (+0.33, +0.69)** | **+0.28 (+0.21, +0.35)** | +0.13 (+0.00, +0.26) |

## Summary: confirmed image-meter effects (SETS) on the same seed (KEPT)

| class | meter | SETS | KEPT | same sign | KEPT flagged |
|---|---|---|---|---|---|
| airplane | brightness | -0.42 | +0.05 (+0.03, +0.08) | no | no |
| airplane | contrast | -0.28 | -0.29 (-0.37, -0.20) | yes | yes |
| airplane | blueness | -0.33 | +0.30 (+0.24, +0.36) | no | yes |
| airplane | redness | +0.31 | -0.25 (-0.32, -0.19) | no | yes |
| cat | border_clutter | +0.25 | +0.43 (+0.33, +0.54) | yes | yes |
| cat | sharpness | +0.77 | +0.73 (+0.56, +0.89) | yes | yes |
| deer | contrast | +0.32 | -0.00 (-0.09, +0.08) | no | no |
| deer | blueness | +0.41 | +0.00 (-0.04, +0.04) | yes | no |
| deer | greenness | -0.33 | -0.03 (-0.09, +0.02) | yes | no |
| deer | redness | -0.25 | +0.02 (-0.02, +0.06) | no | no |
| deer | figure_ground | +0.49 | -0.07 (-0.20, +0.05) | no | no |
| deer | sharpness | +0.31 | -0.04 (-0.15, +0.08) | no | no |
| ship | blueness | -0.68 | +0.14 (+0.05, +0.25) | no | no |
| ship | redness | +0.61 | -0.10 (-0.18, -0.02) | no | no |
| ship | sharpness | +0.70 | +0.50 (+0.33, +0.69) | yes | yes |

6 of 15 confirmed image-meter effects have the same sign on the same seed; 6 of 15 are also flagged there.

## Decomposition of the pre-registered effect (point estimates, SDs of real class images)

SETS = STEERING + SEED, exactly: STEERING is steered - unguided on the seeds whose steered image is the class (ALL above); SEED is what those seeds make without steering minus the model's own class images, i.e. what the steered images inherit from the image the seed would have been.

| class | meter | SETS | STEERING | SEED |
|---|---|---|---|---|
| airplane | brightness | -0.42 | +0.05 | -0.47 |
| airplane | contrast | -0.28 | -0.41 | +0.13 |
| airplane | blueness | -0.33 | +0.33 | -0.66 |
| airplane | redness | +0.31 | -0.30 | +0.61 |
| cat | border_clutter | +0.25 | +0.43 | -0.18 |
| cat | sharpness | +0.77 | +0.64 | +0.14 |
| deer | contrast | +0.32 | -0.22 | +0.54 |
| deer | blueness | +0.41 | -0.10 | +0.51 |
| deer | greenness | -0.33 | +0.11 | -0.43 |
| deer | redness | -0.25 | +0.04 | -0.29 |
| deer | figure_ground | +0.49 | -0.25 | +0.74 |
| deer | sharpness | +0.31 | -0.04 | +0.35 |
| ship | blueness | -0.68 | +0.52 | -1.19 |
| ship | redness | +0.61 | -0.40 | +1.02 |
| ship | sharpness | +0.70 | +0.28 | +0.42 |
