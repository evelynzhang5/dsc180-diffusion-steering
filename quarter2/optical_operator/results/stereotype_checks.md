# Round 14 hypothesis 2: does a blur request change the kind of scene? (SD 1.5, 4 scenes x 4 seeds; existence check)

| check | result | detail |
|---|---|---|
| all generations measured | PASS | 48 |
| contact sheet <= 25 images | PASS | 12 |

Changes vs the unsteered image of the same seed. area = CLIPSeg area of the named subject (person for the empty street: should stay 0). fg pref = CLIP logit 'close-up object in the foreground' minus 'nothing in the foreground'. `stereotype_examples.png`: rows = scenes, columns = none, prompt, rfm (seed 42).

| scene | condition | bg sharpness change | CLIP blur score | fg pref | subject area (x1000) | DINOv2 |
|---|---|---|---|---|---|---|
| landscape | none | +0.000 | -2.10 | -4.00 | nan | 1.000 |
| landscape | prompt | +0.184 | -1.63 | -4.09 | nan | 0.821 |
| landscape | rfm | -0.168 | -1.18 | -3.11 | nan | 0.521 |
| small_bird | none | +0.000 | -0.45 | -0.80 | 17.3 | 1.000 |
| small_bird | prompt | -1.066 | +0.37 | -1.38 | 17.5 | 0.745 |
| small_bird | rfm | -0.989 | -0.05 | -0.96 | 17.2 | 0.535 |
| sharp_wildlife | none | +0.000 | -1.49 | +0.16 | 124.1 | 1.000 |
| sharp_wildlife | prompt | -0.345 | -1.61 | +0.13 | 126.1 | 0.866 |
| sharp_wildlife | rfm | -0.146 | -1.52 | -0.99 | 132.2 | 0.807 |
| empty_street | none | +0.000 | -0.45 | -3.23 | 0.0 | 1.000 |
| empty_street | prompt | -0.124 | -0.19 | -2.43 | 0.0 | 0.860 |
| empty_street | rfm | -0.218 | -0.11 | -2.41 | 0.0 | 0.569 |

Total time 959 s.
