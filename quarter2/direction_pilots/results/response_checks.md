# Pilot 1: response of the final image to edit components (4 scenes x seed 42, s 4; target background change -0.5)

| check | result | detail |
|---|---|---|
| every run produced finite measures | PASS | 4 scenes |
| masked RFM blurs the background at c = 1 (a < 0) in every scene | PASS | -1.085, -0.178, -1.582, -1.155 |
| linearity in scale: bg at c = 2 vs 2 x bg at c = 1 | info | -0.83 vs -2.17; -0.07 vs -0.36; -1.29 vs -3.16; -1.13 vs -2.31 |
| linearity in scale: subject pixel change, relative error | info | 0.74, 0.63, 0.93, 0.61 |
| additivity: bg of RFM-bg + paired-bg vs sum of the two | info | -0.89 vs -1.39; -0.31 vs -0.10; -1.93 vs -3.11; -2.09 vs -2.79 |
| additivity: subject pixel change, relative error | info | 0.67, 0.67, 0.80, 0.61 |
| contact sheet <= 25 images | PASS | 16 |

Linear model from single components; a relative error near 0 means the response is linear (the model is trustworthy), near 1 or above means it is not. `response_examples.png` columns: none, masked RFM matched, optimised combination, masked RFM at c = 1.

| scene | method | c (rfm_bg, rfm_fg, paired_bg, eig2_bg) | bg (target -0.5) | predicted bg | fg sharpness | subject pixel RMS | predicted | pixel FG / BG | DINOv2 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | masked RFM | +0.46, +0.00, +0.00, +0.00 | -0.851 | -0.500 | -0.236 | 0.1472 | 0.0788 | 0.102 / 0.082 | 0.764 |
| 0 | optimised | +0.26, -0.32, -0.03, +0.01 | -0.244 | -0.500 | -0.091 | 0.0918 | 0.0441 | 0.057 / 0.055 | 0.733 |
| 1 | masked RFM | +2.80, +0.00, +0.00, +0.00 | -0.121 | -0.500 | -0.216 | 0.1144 | 0.2806 | 0.069 / 0.116 | 0.399 |
| 1 | optimised | +0.08, +0.39, -0.18, -0.29 | +2.072 | -0.500 | +0.101 | 0.0961 | 0.0422 | 0.049 / 0.115 | 0.679 |
| 3 | masked RFM | +0.32, +0.00, +0.00, +0.00 | -0.502 | -0.500 | -0.352 | 0.0641 | 0.0447 | 0.043 / 0.059 | 0.969 |
| 3 | optimised | +0.14, -0.09, +0.18, -0.11 | -0.572 | -0.500 | +0.143 | 0.1019 | 0.0293 | 0.070 / 0.126 | 0.868 |
| 5 | masked RFM | +0.43, +0.00, +0.00, +0.00 | -0.331 | -0.500 | +0.011 | 0.1679 | 0.0819 | 0.108 / 0.085 | 0.794 |
| 5 | optimised | +0.03, -0.12, +0.30, -0.11 | -0.509 | -0.500 | +0.197 | 0.1473 | 0.0339 | 0.094 / 0.105 | 0.794 |

Means over the 4 scenes with a baseline: background masked -0.451 vs optimised +0.187; subject pixel RMS masked 0.1234 vs optimised 0.1093; DINOv2 masked 0.732 vs optimised 0.768.

Total time 768 s.
