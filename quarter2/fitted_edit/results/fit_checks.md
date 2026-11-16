# Round 16: per-image fitted activation edits (3 scenes, privileged feasibility test)

| check | result | detail |
|---|---|---|
| regenerated unedited images match the saved ones (<= 1 level) | PASS | 0, 0, 0 |
| round 13 sampler gives the same unedited image | PASS | 0, 0, 0 |
| no-edit endpoint: bg change 0, subject change 0 | PASS |  |
| target endpoint: reaches the target, subject interior unchanged | PASS |  |
| optimization control | FAIL | 0.480, 0.632 |
| all measures finite | PASS |  |


## Target -0.5

| method | reached | interior MAE | subject MAE | band MAE vs target | bg MAE vs target / no-edit's | layout IoU |
|---|---|---|---|---|---|---|
| none | 0/3 | 0.00 | 0.00 | 1.12 | 1.00 | 1.00 |
| target | 3/3 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| rfm | 3/3 | 33.29 | 33.76 | 34.90 | 16.32 | 0.67 |
| global | 0/3 | 0.87 | 0.95 | 2.21 | 1.55 | 1.00 |
| spatial | 1/3 | 2.07 | 2.17 | 2.81 | 1.67 | 0.99 |

Per scene (bg change / interior MAE; criteria: R reach, S subject <= 3, A background agreement, V spatial >= 50% lower subject error than the best feasible baseline, B boundary no worse):

| scene | rfm (window) | global | spatial | spatial criteria | passes |
|---|---|---|---|---|---|
| 0 person | -0.55 / 11.51 (middle) | -0.25 / 0.78 | -0.29 / 1.16 | -S-VB | no |
| 1 animal | -0.49 / 33.18 (middle) | -0.18 / 1.84 | -0.39 / 2.87 | RS-VB | no |
| 2 animal, cluttered | -0.52 / 55.19 (early) | +0.00 / 0.00 | -0.27 / 2.19 | -S-VB | no |

## Target -2.0

| method | reached | interior MAE | subject MAE | band MAE vs target | bg MAE vs target / no-edit's | layout IoU |
|---|---|---|---|---|---|---|
| none | 0/3 | 0.00 | 0.00 | 4.08 | 1.00 | 1.00 |
| target | 3/3 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| rfm | 0/3 | 39.79 | 39.80 | 37.05 | 4.76 | 0.47 |
| global | 0/3 | 9.81 | 10.67 | 15.70 | 1.98 | 0.96 |
| spatial | 0/3 | 7.91 | 8.64 | 12.51 | 1.77 | 0.96 |

Per scene (bg change / interior MAE; criteria: R reach, S subject <= 3, A background agreement, V spatial >= 50% lower subject error than the best feasible baseline, B boundary no worse):

| scene | rfm (window) | global | spatial | spatial criteria | passes |
|---|---|---|---|---|---|
| 0 person | -0.37 / 11.20 (middle) | -1.81 / 6.51 | -1.46 / 5.04 | ---VB | no |
| 1 animal | -1.09 / 52.25 (full) | -1.59 / 10.13 | -1.60 / 8.13 | ---VB | no |
| 2 animal, cluttered | -1.48 / 55.92 (full) | -1.41 / 12.78 | -1.50 / 10.55 | ---VB | no |

## Decision

- Spatial edit passes all five criteria: 0/3 at -0.5, 0/3 at -2.0 (GO needs >= 2/3 at both).
- Spatial edit passes reach + subject + agreement: 0/3 at -0.5, 0/3 at -2.0
- Global edit passes reach + subject + agreement: 0/3 at -0.5, 0/3 at -2.0
- Failure modes of the spatial edit: {"-0.5": {"subject kept only by missing the blur": [0, 2], "background not like the target": [1]}, "-2.0": {"misses blur and changes subject": [0, 1, 2]}}
- Optimization control (recover a known spatial edit, MSE <= 10% of zero-edit): scene 0 0.480, scene 3 0.632 -> FAIL
- **NO-GO for this architecture** (inconclusive: the optimizer failed its control)

Total time 0 s.
