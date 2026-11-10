# Reference groups: portrait composition x depth of field

1200 photos from svnfs/depth-of-field (label 0 = shallow). Portrait = CLIP ViT-L/14 zero-shot probability on 7 portrait prompts against 12 others >= threshold.

| threshold | shallow, portrait | shallow, not | deep, portrait | deep, not |
|---|---|---|---|---|
| 0.3 | 79 | 521 | 39 | 561 |
| 0.5 | 36 | 564 | 17 | 583 |
| 0.7 | 18 | 582 | 13 | 587 |

Portrait probability, counts per 0.1 bin: 933, 98, 51, 45, 20, 15, 7, 8, 9, 14

Mean log Laplacian variance per cell (threshold 0.5):

- shallow, portrait: n 36, mean 4.57, 10/50/90% 3.63 / 4.55 / 5.67
- shallow, not: n 564, mean 4.35, 10/50/90% 3.18 / 4.37 / 5.46
- deep, portrait: n 17, mean 5.20, 10/50/90% 4.39 / 5.02 / 6.25
- deep, not: n 583, mean 5.19, 10/50/90% 3.82 / 5.32 / 6.33

`audit_blind.png`: 24 photos (6 random per cell), numbered; `audit_key.json` holds their labels.

