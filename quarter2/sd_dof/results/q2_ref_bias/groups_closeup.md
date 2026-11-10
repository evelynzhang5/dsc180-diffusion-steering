# Reference groups: closeup composition x depth of field

1200 photos from svnfs/depth-of-field (label 0 = shallow). closeup = CLIP ViT-L/14 zero-shot probability on 7 closeup prompts against 10 others >= threshold.

| threshold | shallow, closeup | shallow, not | deep, closeup | deep, not |
|---|---|---|---|---|
| 0.3 | 503 | 97 | 146 | 454 |
| 0.5 | 445 | 155 | 104 | 496 |
| 0.7 | 370 | 230 | 75 | 525 |

closeup probability, counts per 0.1 bin: 446, 65, 40, 53, 47, 55, 49, 76, 98, 271

Mean log Laplacian variance per cell (threshold 0.5):

- shallow, closeup: n 445, mean 4.30, 10/50/90% 3.16 / 4.33 / 5.41
- shallow, not: n 155, mean 4.54, 10/50/90% 3.55 / 4.62 / 5.57
- deep, closeup: n 104, mean 5.25, 10/50/90% 3.86 / 5.32 / 6.33
- deep, not: n 496, mean 5.18, 10/50/90% 3.85 / 5.31 / 6.32

`audit_closeup_blind.png`: 24 photos (6 random per cell), numbered; `audit_closeup_key.json` holds their labels.

