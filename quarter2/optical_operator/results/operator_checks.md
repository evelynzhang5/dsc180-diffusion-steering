# Round 14 hypothesis 1: predicting a blur operator on the source pixels (synthetic layered scenes; decisive first test)

| check | result | detail |
|---|---|---|
| identity on the protected region: every method has the same protected error | PASS | 0.143, 0.215, 0.297 |
| classical renderers leave a matched blur (|blur gap| of normconv < 0.5 at every level) | PASS | +0.00, +0.00, -0.00 |
| training loss fell for both models | PASS | image 0.0579 -> 0.0028, operator 0.0495 -> 0.0024 |
| parameter counts within 5% | PASS | image 472,195, operator 481,633 |
| contact sheet <= 25 images | PASS | 24 |

Train 480 / test 120 photos; 4000 steps x batch 16; test 128 scenes per level. Mean absolute error x 255 against the independently rendered target, by region of the true geometry. Blur gap: background log Laplacian variance minus the target's (0 = matched blur, > 0 = not blurred enough). Parameters: image 472,195, operator 481,633. Inference at 512 x 512 (ms): gather 3.2, normconv 4.4, image 9.5, operator 29.3. `operator_examples.png` columns: input, target, gather, normconv, image, operator.

| level (px) | method | protected | thin strands | boundary band | background | all | blur gap |
|---|---|---|---|---|---|---|---|
| 2 | gather | 0.14 | 8.76 | 4.66 | 0.00 | 0.75 | +0.000 |
| 2 | normconv | 0.14 | 11.88 | 3.39 | 0.00 | 0.64 | +0.000 |
| 2 | image | 0.14 | 2.89 | 1.25 | 0.37 | 0.46 | +0.005 |
| 2 | operator | 0.14 | 3.01 | 1.62 | 0.07 | 0.32 | -0.001 |
| 4 | gather | 0.22 | 9.63 | 6.83 | 0.00 | 1.50 | +0.000 |
| 4 | normconv | 0.22 | 15.37 | 3.49 | 0.00 | 0.97 | +0.000 |
| 4 | image | 0.22 | 3.65 | 1.47 | 0.43 | 0.63 | +0.009 |
| 4 | operator | 0.22 | 4.31 | 1.89 | 0.07 | 0.52 | -0.001 |
| 8 | gather | 0.30 | 13.09 | 7.76 | 0.00 | 2.60 | -0.000 |
| 8 | normconv | 0.30 | 20.10 | 3.08 | 0.00 | 1.31 | -0.000 |
| 8 | image | 0.30 | 4.76 | 1.78 | 0.67 | 0.98 | +0.012 |
| 8 | operator | 0.30 | 6.87 | 2.76 | 0.10 | 1.05 | -0.000 |

## Decision (frozen in README)

- Operator boundary error <= 0.9 x the best other method: 0/3 levels.
- Thin strands no worse than the best other (within 5%): 1/3; background within 10%: 0/3.
- **Do not continue as framed: no boundary advantage over the classical renderers and the image model.**

Total time 662 s.
