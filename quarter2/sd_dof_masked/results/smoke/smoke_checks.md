# Round 10 smoke test (6 captions x seed 42; pipeline check, not a result)

| check | pass | detail |
|---|---|---|
| identity: unsteered vs itself, every change 0 | PASS | max |change| 0.00e+00 |
| masked hook: layer output unchanged where mask = 0 | PASS | max |change| 0.00e+00 |
| masked hook: layer output changed where mask > 0 | PASS | max |change| 5.093 |
| post-hoc blur: background sharper drop than foreground (|fg| < 0.25 |bg|, bg < 0) | PASS | bg -4.377, fg -0.103 |
| measure separates selective from uniform blur (post-hoc ratio > uniform ratio + 0.3) | PASS | post-hoc +4.274, uniform +0.201 |
| fixed-mask and own-mask ratios finite for every generated condition | PASS |  |
| matching returns a strength or says out of range for each method | PASS | rfm_global: 3.24, rfm_masked: 1.54, posthoc: 0.11 |
| all metrics finite | PASS | 384 values |
| runtime and peak memory recorded | PASS |  |
| contact sheet <= 24 images | PASS | 24 images |

Means over the captions, paired with the unsteered image (log changes; fixed mask unless noted):

| condition | background | foreground | FG/BG ratio | ratio, own mask | pixel change FG / BG | DINOv2 | s per image | peak GB |
|---|---|---|---|---|---|---|---|---|
| none | +0.000 | +0.000 | +0.000 | +0.000 | 0.000 / 0.000 | 1.000 | 14.9 | 6.76 |
| rfm_global_s2 | -0.146 | -0.430 | -0.283 | -0.317 | 0.059 / 0.067 | 0.793 | 21.8 | 6.76 |
| rfm_global_s4 | -0.362 | -0.605 | -0.243 | -0.342 | 0.082 / 0.089 | 0.646 | 21.9 | 6.76 |
| rfm_masked_s2 | -0.364 | -0.143 | +0.221 | +0.251 | 0.052 / 0.059 | 0.865 | 21.9 | 6.76 |
| rfm_masked_s4 | -0.933 | -0.158 | +0.775 | +0.790 | 0.079 / 0.090 | 0.690 | 21.9 | 6.76 |
| posthoc_r1 | -2.439 | -0.085 | +2.354 | +2.359 | 0.001 / 0.031 | 0.968 | nan | nan |
| posthoc_r3 | -4.377 | -0.103 | +4.274 | +4.287 | 0.003 / 0.063 | 0.860 | nan | nan |
| uniform_r2 | -4.624 | -4.422 | +0.201 | +0.201 | 0.027 / 0.052 | 0.980 | nan | nan |

Matched-blur strengths for target background change -0.28: rfm_global 3.24, rfm_masked 1.54, posthoc 0.11

Per caption (ratio, fixed mask):

| caption kind | rfm_global_s2 | rfm_global_s4 | rfm_masked_s2 | rfm_masked_s4 |
|---|---|---|---|---|
| ordinary | -0.564 | -0.643 | +0.272 | +0.887 |
| ordinary | -0.689 | -1.025 | -0.188 | -0.103 |
| no_subject | -0.591 | -0.588 | -0.002 | +0.059 |
| two_depths | -0.400 | -0.428 | +0.184 | +0.710 |
| small_subject | +0.482 | +1.271 | +0.531 | +1.912 |
| sharp_bg | +0.061 | -0.044 | +0.526 | +1.187 |

Total time 647 s.
