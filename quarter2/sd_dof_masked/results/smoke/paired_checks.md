# Smoke test 2: paired data (8 captions x seed 42, levels [2.0, 4.0, 8.0]; pipeline check)

| check | result | detail |
|---|---|---|
| rendered pairs: foreground pixels unchanged (max mean change < 0.005) | PASS | max 0.0000 |
| rendered pairs: background change > 0 and grows with level | PASS | 0.0269 / 0.0445 / 0.0620 |
| paired activations: same shape for original and edit | PASS | 1280 x 16 x 16 |
| paired activations: repeating the original with the same noise gives the same activations | PASS | max relative change 0.00e+00 |
| activation change grows with level | PASS | 261.5 / 389.9 / 511.1 |
| paired direction: cosine with round 9's directions | info | rfm_eig1 +0.128, meandiff +0.159 |
| per-pair changes: mean cosine between pairs (1 = one shared direction) | info | +0.115 (min +0.038, max +0.220) |
| per-pair changes: energy at background / foreground positions (median) | info | 7.13 over 8 images |
| paired direction runs through the sampler; metrics finite | PASS | bg -1.280 fg -0.725; bg -0.546 fg +0.039; 19.5 s per image, peak 6.46 GB |
| contact sheets <= 25 images each | PASS | 16 and 4 |

Rendered pairs: columns original, sigma 2, 4, 8 (`pairs_examples.png`); paired direction at s 4 next to the original (`paired_direction_examples.png`). The paired direction is the mean of per-pair changes, i.e. the difference of the two group means: it only differs from a mean-difference direction through the data.

Total time 184 s.
