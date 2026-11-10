# SD 1.5 reference-bias check: blur direction vs. people and framing

Mid-block edit a x |h| x d during DDIM steps 2-11 of 30. Evaluation: seeds 2-9, 12 object prompts and 4 person prompts. Changes are paired against the unedited image. Mean (95% interval over images).

## Reference sets (meters on the reference images)

| set | group | n | background sharpness | CLIP blur score | person present (OWLv2) | CLIP person score | framing | warmth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biased | blur | 64 | 2.57 | +0.96 | 0.61 | +3.68 | 0.66 | +0.076 |
| biased | sharp | 64 | 4.05 | +0.57 | 0.11 | -0.43 | 0.55 | +0.102 |
| opposed | blur | 64 | 3.29 | +0.93 | 0.09 | -0.67 | 0.46 | +0.093 |
| opposed | sharp | 64 | 3.29 | +0.34 | 0.52 | +3.16 | 0.71 | +0.075 |
| balanced | blur | 64 | 2.85 | +0.94 | 0.31 | +1.44 | 0.57 | +0.088 |
| balanced | sharp | 64 | 4.10 | +0.34 | 0.30 | +1.30 | 0.59 | +0.089 |

## Direction cosines (mean over steps)

| | biased | opposed | balanced | random |
| --- | --- | --- | --- | --- |
| biased | +1.00 | -0.66 | +0.28 | +0.00 |
| opposed | -0.66 | +1.00 | +0.35 | -0.00 |
| balanced | +0.28 | +0.35 | +1.00 | +0.00 |
| random | +0.00 | -0.00 | +0.00 | +1.00 |

Blur meter separation of blur vs. sharp references (standardized): bg_sharp 0.44, clip_blur 0.48. Target meter: **clip_blur**.

## Strength sweep, balanced direction (dev seeds)

| a | blur change (target meter, + = blurrier) | subject score change | person on object prompts |
| --- | --- | --- | --- |
| 0.25 | -0.108 | -0.007 | +0.000 |
| 0.5 | -0.204 | -0.006 | +0.000 |
| 1.0 | +0.016 | -0.025 | +0.000 |

Target: +0.016 (balanced a = 1.0). Calibrated: biased a = 1.000 (dev +0.022, 4 tries), opposed a = 2.000 (dev -0.037, 4 tries); random at a = 1.0.

## Object prompts: change vs. unedited (unedited person rate 0.03)

| direction | a | background sharpness | CLIP blur score | whole-image sharpness | CLIP person score | person present | person score | person box area | framing | subject score | warmth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balanced | 1.000 | -0.050 (-0.132, +0.035) | -0.039 (-0.172, +0.097) | -0.017 (-0.079, +0.048) | +0.028 (-0.165, +0.219) | +0.000 (-0.042, +0.042) | +0.004 (-0.006, +0.014) | +0.004 (-0.010, +0.019) | -0.027 (-0.050, -0.004) | -0.019 (-0.036, -0.002) | -0.001 (-0.003, +0.002) |
| biased | 1.000 | +0.101 (-0.072, +0.282) | -0.029 (-0.210, +0.144) | +0.067 (-0.027, +0.156) | +0.217 (-0.023, +0.471) | -0.010 (-0.042, +0.021) | +0.002 (-0.007, +0.012) | -0.002 (-0.015, +0.010) | -0.011 (-0.048, +0.021) | -0.033 (-0.055, -0.012) | -0.002 (-0.005, +0.001) |
| opposed | 2.000 | +0.142 (-0.005, +0.288) | +0.034 (-0.146, +0.214) | -0.005 (-0.099, +0.088) | +0.028 (-0.210, +0.266) | +0.010 (-0.021, +0.042) | +0.009 (+0.001, +0.018) | +0.012 (-0.001, +0.031) | -0.003 (-0.042, +0.036) | -0.055 (-0.086, -0.028) | +0.001 (-0.003, +0.005) |
| random | 1.000 | +0.000 (-0.083, +0.074) | -0.031 (-0.156, +0.081) | -0.018 (-0.059, +0.027) | +0.009 (-0.115, +0.133) | -0.010 (-0.042, +0.000) | +0.002 (-0.003, +0.008) | -0.000 (-0.001, +0.000) | -0.008 (-0.023, +0.008) | -0.001 (-0.013, +0.012) | +0.001 (-0.000, +0.002) |

## Person prompts: change vs. unedited

| direction | background sharpness | CLIP blur score | framing | subject score | warmth |
| --- | --- | --- | --- | --- | --- |
| balanced | -0.152 (-0.574, +0.185) | +0.125 (-0.057, +0.332) | -0.033 (-0.065, -0.003) | -0.006 (-0.040, +0.031) | -0.003 (-0.008, +0.001) |
| biased | -0.061 (-0.732, +0.547) | +0.084 (-0.184, +0.355) | +0.141 (+0.082, +0.216) | +0.018 (-0.018, +0.057) | +0.001 (-0.003, +0.005) |
| opposed | -0.004 (-0.504, +0.349) | +0.126 (-0.185, +0.475) | -0.030 (-0.118, +0.060) | -0.007 (-0.068, +0.049) | -0.001 (-0.010, +0.006) |
| random | -0.067 (-0.477, +0.210) | -0.003 (-0.186, +0.158) | -0.015 (-0.062, +0.038) | -0.012 (-0.033, +0.010) | -0.002 (-0.004, +0.001) |
