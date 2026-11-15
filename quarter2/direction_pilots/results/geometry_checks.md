# Pilot 2: subject protection under depth errors (8 images; ground truth = our renderer with the estimated depth; pipeline check)

| check | result | detail |
|---|---|---|
| all errors finite | PASS | 36 cells |
| no depth error gives zero error for the plain render (errors are real) | PASS |  |
| contact sheet <= 25 images | PASS | 10 |

Mean absolute error against the truth (fraction of 255) by region; boundary = within 2 x sigma of the true subject edge. bg blur gap: log Laplacian variance of the background interior minus the truth's (> 0: not blurred enough). `geometry_examples.png` columns: truth, plain, feather 4, feather L, ensemble (threshold error, sigma 8).

Depth errors: noise = correlated noise of 0.15 depth SD; smoothed = depth blurred (sigma 6 px); threshold = subject cut at the nearest 30% instead of 40%. Ensemble: 8 renders with extra 0.05 SD noise and the cut drawn from the nearest 35-45%.

| depth error | sigma | method | subject interior | boundary | background interior | bg blur gap |
|---|---|---|---|---|---|---|
| noise | 2 | plain | 0.0000 | 0.0027 | 0.0083 | +0.529 |
| noise | 2 | feather4 | 0.0000 | 0.0037 | 0.0083 | +0.533 |
| noise | 2 | feather_L | 0.0000 | 0.0033 | 0.0083 | +0.530 |
| noise | 2 | ensemble8 | 0.0000 | 0.0029 | 0.0088 | +0.550 |
| noise | 4 | plain | 0.0001 | 0.0046 | 0.0107 | +0.771 |
| noise | 4 | feather4 | 0.0000 | 0.0058 | 0.0107 | +0.789 |
| noise | 4 | feather_L | 0.0000 | 0.0058 | 0.0107 | +0.789 |
| noise | 4 | ensemble8 | 0.0001 | 0.0049 | 0.0112 | +0.795 |
| noise | 8 | plain | 0.0001 | 0.0068 | 0.0123 | +1.110 |
| noise | 8 | feather4 | 0.0000 | 0.0077 | 0.0124 | +1.148 |
| noise | 8 | feather_L | 0.0000 | 0.0083 | 0.0124 | +1.161 |
| noise | 8 | ensemble8 | 0.0002 | 0.0071 | 0.0124 | +1.045 |
| smoothed | 2 | plain | 0.0000 | 0.0059 | 0.0060 | -0.373 |
| smoothed | 2 | feather4 | 0.0000 | 0.0060 | 0.0060 | -0.365 |
| smoothed | 2 | feather_L | 0.0000 | 0.0059 | 0.0060 | -0.371 |
| smoothed | 2 | ensemble8 | 0.0000 | 0.0062 | 0.0054 | -0.183 |
| smoothed | 4 | plain | 0.0000 | 0.0073 | 0.0073 | -0.627 |
| smoothed | 4 | feather4 | 0.0000 | 0.0076 | 0.0073 | -0.618 |
| smoothed | 4 | feather_L | 0.0000 | 0.0076 | 0.0073 | -0.618 |
| smoothed | 4 | ensemble8 | 0.0001 | 0.0083 | 0.0062 | -0.275 |
| smoothed | 8 | plain | 0.0000 | 0.0076 | 0.0084 | -0.815 |
| smoothed | 8 | feather4 | 0.0000 | 0.0079 | 0.0084 | -0.813 |
| smoothed | 8 | feather_L | 0.0000 | 0.0084 | 0.0084 | -0.802 |
| smoothed | 8 | ensemble8 | 0.0001 | 0.0090 | 0.0074 | -0.524 |
| threshold | 2 | plain | 0.0006 | 0.0085 | 0.0086 | -1.229 |
| threshold | 2 | feather4 | 0.0006 | 0.0079 | 0.0086 | -1.225 |
| threshold | 2 | feather_L | 0.0006 | 0.0080 | 0.0086 | -1.229 |
| threshold | 2 | ensemble8 | 0.0000 | 0.0009 | 0.0028 | +0.155 |
| threshold | 4 | plain | 0.0015 | 0.0134 | 0.0109 | -1.733 |
| threshold | 4 | feather4 | 0.0015 | 0.0129 | 0.0109 | -1.733 |
| threshold | 4 | feather_L | 0.0015 | 0.0129 | 0.0109 | -1.733 |
| threshold | 4 | ensemble8 | 0.0000 | 0.0020 | 0.0037 | +0.178 |
| threshold | 8 | plain | 0.0017 | 0.0175 | 0.0129 | -1.636 |
| threshold | 8 | feather4 | 0.0017 | 0.0172 | 0.0129 | -1.636 |
| threshold | 8 | feather_L | 0.0016 | 0.0170 | 0.0129 | -1.634 |
| threshold | 8 | ensemble8 | 0.0000 | 0.0039 | 0.0052 | +0.095 |

Total time 254 s.
