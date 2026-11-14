# Test: 32 new person/animal (P) + 32 object/scene (O) captions x seeds [42, 123, 256, 789]

Paired with the unsteered image, inside its depth mask; intervals: bootstrap over captions, then seeds.

| condition | setting | background | foreground | log FG/BG ratio | ratio, P | ratio, O | DINOv2 cosine | CLIP caption | pixel change FG / BG | ratio, own mask |
|---|---|---|---|---|---|---|---|---|---|---|
| paper | rfm_eig1_s4 | -0.279 (-0.398, -0.152) | -0.386 (-0.486, -0.282) | -0.106 (-0.223, +0.011) | -0.057 (-0.224, +0.102) | -0.155 (-0.319, -0.000) | +0.675 (+0.637, +0.713) | +0.16 (-0.27, +0.57) | 0.127 / 0.119 | +0.146 (-0.020, +0.326) |
| random | random_s1 | +0.019 (-0.006, +0.047) | -0.003 (-0.027, +0.020) | -0.022 (-0.052, +0.004) | -0.013 (-0.054, +0.018) | -0.032 (-0.073, +0.008) | +0.936 (+0.921, +0.950) | +0.13 (-0.05, +0.32) | 0.036 / 0.031 | -0.009 (-0.042, +0.025) |
| prompt | prompt_s2 | -0.058 (-0.138, +0.025) | -0.046 (-0.118, +0.017) | +0.012 (-0.068, +0.091) | +0.082 (+0.002, +0.172) | -0.059 (-0.184, +0.060) | +0.829 (+0.802, +0.857) | -0.19 (-0.54, +0.17) | 0.086 / 0.074 | +0.100 (+0.008, +0.195) |
| oracle | radius 0.5 | -0.705 (-0.742, -0.664) | -0.051 (-0.057, -0.045) | +0.654 (+0.614, +0.690) | +0.714 (+0.666, +0.761) | +0.594 (+0.536, +0.650) | +0.999 (+0.999, +0.999) | -0.03 (-0.06, -0.01) | 0.000 / 0.006 | n/a |

No eligible challenger (none reached the target background change): the challenger branch of the decision fails.

For the other branches of the decision table (paired differences in log FG/BG ratio change):
- oracle - paper: +0.760 (+0.648, +0.868)
- random - paper: +0.084 (-0.034, +0.192)
