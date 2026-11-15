# Round 12 idea 1 feasibility: do measured-response corrections beat tuned masking? (6 scenes x seed 42, s 4, target background change -0.5; diagnostic)

| check | result | detail |
|---|---|---|
| all scenes ran | PASS | 6/6 |
| all measures finite | PASS |  |
| tuned masking reaches the target | PASS | -0.56, -0.47, -0.46, -0.49, -0.30, -0.59 |
| responses measurable: the background responds to rfm_bg (gradient < 0) | PASS | -0.03, -0.43, -1.78, -0.65, -0.30, -1.12 |
| contact sheet <= 25 images | PASS | 25 |

Basis: rfm_bg, rfm_fg, eig2_fg, eig2_bg, paired_bg (direction x 16x16 block weight; _bg = non-subject fraction, _fg = the rest). Operating point = masked RFM tuned to the target. Responses of the signed subject pixels (vs the unedited reference) and of the background change, by central differences (+/- 0.5). Correction: minimise |r0 + J dx|^2 + small ridge, background change held at the target, each |dx_k| <= 1.0; generated, then one more step with the same J. Random control: a correction of the same size and predicted background change in a random direction. Nonlinearity = |r(+) + r(-) - 2 r0| / |r(+) - r(-)| (0 = linear). `feasibility_examples.png` columns: subject mask, masked tuned, masked dilated tuned, best correction, copy-back.

| scene | method | achieved | bg | subject interior sharpness | subject RMS | interior RMS | subject SSIM | subject DINOv2 | boundary err | DINOv2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | masked | yes | -0.560 | -0.091 | 0.1643 | 0.1534 | 0.598 | 0.755 | 0.1558 | 0.696 |
| 0 | masked_dilated | yes | -0.584 | -0.247 | 0.1972 | 0.1868 | 0.498 | 0.636 | 0.1938 | 0.710 |
| 0 | global | yes | -0.515 | -0.348 | 0.2053 | 0.1981 | 0.490 | 0.460 | 0.1734 | 0.595 |
| 0 | posthoc | no | -0.787 | +0.000 | 0.0002 | 0.0000 | 1.000 | 0.999 | 0.0134 | 0.999 |
| 0 | copyback | yes | -0.560 | +0.000 | 0.0015 | 0.0000 | 0.998 | 0.965 | 0.0792 | 0.861 |
| 0 | corrected1 | yes | -0.411 | +0.047 | 0.1802 | 0.1698 | 0.544 | 0.756 | 0.1560 | 0.685 |
| 0 | corrected2 | no | -0.212 | +0.012 | 0.1996 | 0.1880 | 0.478 | 0.697 | 0.1790 | 0.642 |
| 0 | random_control | yes | -0.628 | -0.192 | 0.1901 | 0.1797 | 0.506 | 0.653 | 0.1830 | 0.627 |
| 0 | floor_10pct | no | -0.015 | -0.026 | 0.0719 | 0.0634 | 0.867 | 0.937 | 0.0685 | 0.947 |
| 1 | masked | yes | -0.466 | -1.126 | 0.1478 | 0.1391 | 0.550 | 0.160 | 0.1555 | 0.330 |
| 1 | masked_dilated | no | -0.327 | -1.450 | 0.1388 | 0.1255 | 0.588 | 0.206 | 0.1537 | 0.298 |
| 1 | global | no | -0.877 | -0.801 | 0.1623 | 0.1530 | 0.423 | 0.174 | 0.1868 | 0.334 |
| 1 | posthoc | yes | -0.502 | +0.000 | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.0011 | 0.999 |
| 1 | copyback | yes | -0.466 | +0.000 | 0.0014 | 0.0000 | 0.997 | 0.924 | 0.0825 | 0.390 |
| 1 | corrected1 | no | -1.151 | -1.114 | 0.1334 | 0.1215 | 0.522 | 0.140 | 0.1726 | 0.383 |
| 1 | corrected2 | no | -0.755 | -1.532 | 0.1262 | 0.1173 | 0.552 | 0.100 | 0.1601 | 0.394 |
| 1 | random_control | no | -1.076 | -0.908 | 0.1271 | 0.1200 | 0.550 | 0.252 | 0.1540 | 0.316 |
| 1 | floor_10pct | no | -0.327 | -2.104 | 0.0853 | 0.0754 | 0.886 | 0.420 | 0.0525 | 0.708 |
| 3 | masked | yes | -0.461 | +0.218 | 0.0568 | 0.0552 | 0.890 | 0.974 | 0.0475 | 0.976 |
| 3 | masked_dilated | yes | -0.467 | +0.217 | 0.0591 | 0.0577 | 0.884 | 0.974 | 0.0478 | 0.977 |
| 3 | global | yes | -0.509 | +0.119 | 0.0682 | 0.0674 | 0.852 | 0.973 | 0.0521 | 0.967 |
| 3 | posthoc | no | -0.955 | +0.000 | 0.0001 | 0.0000 | 1.000 | 0.997 | 0.0067 | 0.997 |
| 3 | copyback | yes | -0.461 | +0.000 | 0.0006 | 0.0000 | 1.000 | 0.995 | 0.0250 | 0.986 |
| 3 | corrected1 | yes | -0.586 | +0.219 | 0.0667 | 0.0660 | 0.855 | 0.966 | 0.0515 | 0.972 |
| 3 | corrected2 | yes | -0.490 | +0.176 | 0.0613 | 0.0596 | 0.875 | 0.975 | 0.0509 | 0.972 |
| 3 | random_control | yes | -0.468 | +0.204 | 0.0549 | 0.0523 | 0.895 | 0.974 | 0.0504 | 0.973 |
| 3 | floor_10pct | no | -0.019 | +0.001 | 0.0051 | 0.0046 | 0.998 | 1.000 | 0.0065 | 0.999 |
| 5 | masked | yes | -0.487 | +0.264 | 0.2819 | 0.2930 | 0.147 | 0.770 | 0.1637 | 0.785 |
| 5 | masked_dilated | yes | -0.460 | +0.309 | 0.2824 | 0.2933 | 0.147 | 0.764 | 0.1641 | 0.780 |
| 5 | global | yes | -0.516 | +0.258 | 0.2784 | 0.2898 | 0.160 | 0.816 | 0.1628 | 0.802 |
| 5 | posthoc | no | -0.927 | +0.000 | 0.0001 | 0.0000 | 1.000 | 0.999 | 0.0153 | 0.999 |
| 5 | copyback | yes | -0.487 | +0.000 | 0.0013 | 0.0000 | 0.996 | 0.960 | 0.0824 | 0.963 |
| 5 | corrected1 | yes | -0.399 | -0.083 | 0.2269 | 0.2281 | 0.230 | 0.867 | 0.1603 | 0.851 |
| 5 | corrected2 | yes | -0.556 | +0.068 | 0.2619 | 0.2702 | 0.162 | 0.808 | 0.1679 | 0.800 |
| 5 | random_control | no | -0.758 | +0.008 | 0.2599 | 0.2660 | 0.145 | 0.714 | 0.1729 | 0.762 |
| 5 | floor_10pct | no | -0.015 | +0.200 | 0.0714 | 0.0676 | 0.807 | 0.954 | 0.0569 | 0.965 |
| 6 | masked | no | -0.300 | +1.374 | 0.2836 | 0.2899 | 0.250 | 0.386 | 0.1613 | 0.445 |
| 6 | masked_dilated | no | -0.282 | +0.077 | 0.1980 | 0.1894 | 0.407 | 0.567 | 0.1457 | 0.742 |
| 6 | global | yes | -0.435 | +0.305 | 0.2234 | 0.2217 | 0.355 | 0.686 | 0.1439 | 0.760 |
| 6 | posthoc | yes | -0.596 | +0.000 | 0.0001 | 0.0000 | 1.000 | 0.993 | 0.0066 | 0.997 |
| 6 | copyback | no | -0.300 | +0.000 | 0.0029 | 0.0000 | 0.994 | 0.804 | 0.0802 | 0.821 |
| 6 | corrected1 | yes | -0.354 | +0.657 | 0.2335 | 0.2340 | 0.302 | 0.343 | 0.1375 | 0.526 |
| 6 | corrected2 | no | -0.814 | +0.535 | 0.2205 | 0.2224 | 0.267 | 0.231 | 0.1455 | 0.581 |
| 6 | random_control | no | -0.305 | +0.871 | 0.2282 | 0.2303 | 0.351 | 0.414 | 0.1460 | 0.562 |
| 6 | floor_10pct | no | -0.006 | +0.201 | 0.1625 | 0.1654 | 0.537 | 0.794 | 0.0971 | 0.905 |
| 7 | masked | yes | -0.586 | -1.483 | 0.2756 | 0.2683 | 0.184 | 0.331 | 0.2305 | 0.823 |
| 7 | masked_dilated | no | -0.731 | -0.631 | 0.2931 | 0.2741 | 0.233 | 0.719 | 0.2570 | 0.849 |
| 7 | global | no | -0.672 | -2.151 | 0.2879 | 0.2848 | 0.165 | 0.250 | 0.2313 | 0.764 |
| 7 | posthoc | no | +0.000 | +0.000 | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.0003 | 1.000 |
| 7 | copyback | yes | -0.586 | +0.000 | 0.0018 | 0.0000 | 0.986 | 0.901 | 0.1215 | 0.873 |
| 7 | corrected1 | no | -0.037 | +0.154 | 0.2093 | 0.1871 | 0.361 | 0.889 | 0.1881 | 0.843 |
| 7 | corrected2 | yes | -0.537 | -0.118 | 0.2160 | 0.1979 | 0.374 | 0.809 | 0.1735 | 0.834 |
| 7 | random_control | yes | -0.620 | -0.691 | 0.2933 | 0.2748 | 0.222 | 0.704 | 0.2478 | 0.788 |
| 7 | floor_10pct | no | -0.024 | -0.011 | 0.0587 | 0.0591 | 0.896 | 0.978 | 0.0231 | 0.995 |

| scene | start RMS | predicted 1 | realised 1 | predicted bg 1 | realised bg 1 | predicted 2 | realised 2 | nonlinearity (subject, per component) | nonlinearity (bg) | dx1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1643 | 0.1566 | 0.1802 | -0.50 | -0.41 | 0.1761 | 0.1996 | 1.09, 1.08, 1.43, 1.25, 1.28 | 1.66, 0.32, 4.01, 0.41, 0.17 | +0.07, -0.21, +0.17, +0.15, +0.12 |
| 1 | 0.1478 | 0.1411 | 0.1334 | -0.50 | -1.15 | 0.1245 | 0.1262 | 1.22, 0.87, 0.52, 1.26, 2.32 | 0.13, 0.63, 0.11, 0.03, 2.05 | -0.19, +0.00, +1.00, +0.31, +1.00 |
| 3 | 0.0568 | 0.0571 | 0.0667 | -0.50 | -0.59 | 0.0629 | 0.0613 | 1.19, 1.00, 1.03, 1.47, 1.42 | 0.47, 0.01, 0.03, 0.24, 0.20 | +0.05, +0.09, +0.02, +0.02, +0.02 |
| 5 | 0.2819 | 0.2489 | 0.2269 | -0.50 | -0.40 | 0.2218 | 0.2619 | 1.07, 1.09, 1.12, 1.20, 1.71 | 0.74, 0.19, 2.37, 0.24, 0.76 | +0.31, +0.20, +0.46, -0.21, -0.35 |
| 6 | 0.2836 | 0.2167 | 0.2335 | -0.50 | -0.35 | 0.2198 | 0.2205 | 1.78, 1.12, 1.62, 1.71, 1.98 | 0.90, 0.65, 0.26, 0.33, 0.34 | +0.09, -0.90, -0.06, -0.00, +0.37 |
| 7 | 0.2756 | 0.1825 | 0.2093 | -0.50 | -0.04 | 0.2140 | 0.2160 | 1.07, 1.03, 1.04, 1.03, 1.72 | 0.02, 0.04, 0.00, 0.45, 0.31 | -0.37, +0.06, -0.33, -0.19, +0.22 |

## Decision (diagnostic, per-scene probing; not evidence of a reusable method)

- Correction beats the best tuned masking by >= 10% subject RMS, at the target: 2/6 scenes.
- Linear prediction reliable (realised RMS within 25%, background within 0.15): 4/6.
- Correction beats the random control: 3/6.
- Correction has lower boundary error than copy-back: 0/6.
- Noise floor: subject RMS of a 10% edit 0.0758 vs the tuned masked edit 0.2017 (close values mean most subject change comes from any perturbation at all, not from the size of the edit).
- Cost: 22 generations per scene (502 s); a tuned mask needs <= 3.
- **Measured-response corrections do not beat tuned masking: reconsider the basis or mechanism before training.**

Total time 3032 s.
