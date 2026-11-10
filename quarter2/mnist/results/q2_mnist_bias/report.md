# MNIST reference-set bias test

Pool of real digits: within-digit corr(thickness, slant) = +0.041. 60 thick + 60 thin per digit per set. Directions from u2 at t = 0.25; edit on t in [0.25, 0.375), strength tuned on dev seed 0 to +-0.3 SD of thickness, evaluated on seed 2 (1000 digits). Thickness SD 0.889 px, slant SD 0.263. Mean (95% interval over images).

## Reference sets

| set | within-digit corr(thickness, slant) | thick - thin: thickness (SD) | thick - thin: slant (SD) | ridge CV accuracy | RFM top-eigenvalue share |
| --- | --- | --- | --- | --- | --- |
| natural | +0.04 | +1.98 | +0.14 | 0.85 | 0.99 |
| balanced | +0.04 | +1.96 | +0.04 | 0.87 | 0.99 |
| aligned | +0.65 | +2.08 | +1.61 | 0.95 | 0.98 |
| opposed | -0.63 | +2.03 | -1.59 | 0.93 | 0.98 |

## Directions: cosine with the balanced meandiff direction / with a slant direction

| set | meandiff | ridge | rfm |
| --- | --- | --- | --- |
| natural | 0.95 / +0.20 | 0.87 / +0.08 | 0.70 / +0.04 |
| balanced | 1.00 / +0.11 | 0.93 / +0.04 | 0.80 / +0.02 |
| aligned | 0.64 / +0.81 | 0.69 / +0.67 | 0.55 / +0.45 |
| opposed | 0.51 / -0.77 | 0.55 / -0.60 | 0.43 / -0.44 |

## At matched thickness: slant change (thicken / thin)

Signed slant change in SD of real slant; a confounded direction should move slant the same way as its reference set (aligned: thicker = higher slant).

| set | learner | strength a (+ / -) | thickness change (+ / -) | signed slant change, thicken | signed slant change, thin | abs. slant change | centroid shift (px) | correct->wrong (+ / -) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| natural | meandiff | 0.42 / 0.50 | +0.31 / -0.30 | +0.018 (+0.009, +0.028) | -0.032 (-0.041, -0.024) | 0.088 | 0.211 | 5 / 5 |
| natural | ridge | 0.41 / 0.51 | +0.31 / -0.30 | -0.004 (-0.013, +0.004) | -0.003 (-0.011, +0.003) | 0.075 | 0.196 | 5 / 2 |
| natural | rfm | 0.52 / 0.61 | +0.30 / -0.29 | -0.022 (-0.031, -0.013) | +0.006 (-0.000, +0.013) | 0.076 | 0.197 | 5 / 2 |
| balanced | meandiff | 0.40 / 0.51 | +0.31 / -0.30 | +0.011 (+0.002, +0.021) | -0.018 (-0.026, -0.010) | 0.085 | 0.209 | 5 / 5 |
| balanced | ridge | 0.40 / 0.52 | +0.31 / -0.30 | +0.002 (-0.006, +0.011) | -0.006 (-0.014, +0.001) | 0.075 | 0.195 | 5 / 3 |
| balanced | rfm | 0.47 / 0.60 | +0.30 / -0.30 | -0.000 (-0.008, +0.008) | -0.009 (-0.016, -0.002) | 0.073 | 0.195 | 5 / 4 |
| aligned | meandiff | 1.62 / 0.62 | +0.30 / -0.30 | +0.329 (+0.307, +0.353) | -0.156 (-0.171, -0.142) | 0.259 | 0.299 | 20 / 2 |
| aligned | ridge | 0.97 / 0.59 | +0.30 / -0.30 | +0.196 (+0.181, +0.213) | -0.134 (-0.147, -0.121) | 0.182 | 0.256 | 9 / 3 |
| aligned | rfm | 1.63 / 0.71 | +0.30 / -0.29 | +0.211 (+0.194, +0.229) | -0.116 (-0.129, -0.104) | 0.182 | 0.257 | 9 / 2 |
| opposed | meandiff | 1.13 / 0.71 | +0.33 / -0.30 | -0.287 (-0.308, -0.267) | +0.160 (+0.146, +0.175) | 0.234 | 0.286 | 13 / 3 |
| opposed | ridge | 0.86 / 0.71 | +0.31 / -0.30 | -0.183 (-0.199, -0.167) | +0.135 (+0.123, +0.148) | 0.172 | 0.249 | 8 / 2 |
| opposed | rfm | 2.04 / 0.92 | +0.30 / -0.29 | -0.269 (-0.291, -0.250) | +0.125 (+0.113, +0.138) | 0.213 | 0.278 | 18 / 4 |

## Signed slant change when thickening, by digit (meandiff)

| set | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| natural | +0.01 | +0.01 | -0.01 | +0.03 | +0.04 | +0.03 | -0.00 | +0.01 | +0.02 | +0.03 |
| balanced | +0.01 | +0.01 | -0.01 | +0.03 | +0.03 | +0.01 | -0.01 | +0.01 | +0.01 | +0.03 |
| aligned | +0.26 | +0.13 | +0.41 | +0.38 | +0.52 | +0.34 | +0.25 | +0.29 | +0.38 | +0.35 |
| opposed | -0.22 | -0.17 | -0.36 | -0.29 | -0.40 | -0.40 | -0.18 | -0.29 | -0.34 | -0.20 |
