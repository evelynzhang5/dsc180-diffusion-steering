# Round 12 idea 2 check: protection under depth errors, with a clean-geometry baseline (8 images; diagnostic)

| check | result | detail |
|---|---|---|
| all finite | PASS |  |
| clean geometry: plain render matches the truth exactly | PASS |  |

The method sees corrupted depth; the truth and the evaluation regions use the uncorrupted depth. 'clean' = the method sees the true depth (the floor each method can reach). soft = sigmoid alpha from depth (width 0.05 depth SD). Truth is our own renderer, so subject-copy methods are exact inside the subject by construction; read the boundary and blur-gap columns.

| depth error | sigma | method | subject interior | boundary | background interior | bg blur gap |
|---|---|---|---|---|---|---|
| clean | 2 | plain | 0.0000 | 0.0000 | 0.0000 | +0.000 |
| clean | 2 | soft | 0.0000 | 0.0000 | 0.0000 | +0.004 |
| clean | 2 | feather_L | 0.0000 | 0.0013 | 0.0000 | +0.000 |
| clean | 2 | ensemble8 | 0.0000 | 0.0012 | 0.0027 | +0.086 |
| clean | 4 | plain | 0.0000 | 0.0000 | 0.0000 | +0.000 |
| clean | 4 | soft | 0.0000 | 0.0002 | 0.0001 | +0.046 |
| clean | 4 | feather_L | 0.0000 | 0.0025 | 0.0000 | +0.001 |
| clean | 4 | ensemble8 | 0.0000 | 0.0017 | 0.0034 | +0.236 |
| clean | 8 | plain | 0.0000 | 0.0000 | 0.0000 | +0.000 |
| clean | 8 | soft | 0.0000 | 0.0004 | 0.0003 | +0.117 |
| clean | 8 | feather_L | 0.0000 | 0.0040 | 0.0000 | +0.004 |
| clean | 8 | ensemble8 | 0.0000 | 0.0033 | 0.0052 | +0.515 |
| noise | 2 | plain | 0.0000 | 0.0026 | 0.0082 | +0.507 |
| noise | 2 | soft | 0.0000 | 0.0026 | 0.0082 | +0.508 |
| noise | 2 | feather_L | 0.0000 | 0.0033 | 0.0082 | +0.508 |
| noise | 2 | ensemble8 | 0.0000 | 0.0028 | 0.0085 | +0.500 |
| noise | 4 | plain | 0.0001 | 0.0045 | 0.0107 | +0.750 |
| noise | 4 | soft | 0.0001 | 0.0044 | 0.0107 | +0.770 |
| noise | 4 | feather_L | 0.0001 | 0.0056 | 0.0107 | +0.770 |
| noise | 4 | ensemble8 | 0.0001 | 0.0050 | 0.0112 | +0.747 |
| noise | 8 | plain | 0.0001 | 0.0067 | 0.0124 | +1.111 |
| noise | 8 | soft | 0.0001 | 0.0067 | 0.0125 | +1.171 |
| noise | 8 | feather_L | 0.0000 | 0.0083 | 0.0125 | +1.166 |
| noise | 8 | ensemble8 | 0.0002 | 0.0072 | 0.0130 | +1.073 |
| smoothed | 2 | plain | 0.0000 | 0.0059 | 0.0060 | -0.373 |
| smoothed | 2 | soft | 0.0000 | 0.0059 | 0.0059 | -0.364 |
| smoothed | 2 | feather_L | 0.0000 | 0.0059 | 0.0060 | -0.371 |
| smoothed | 2 | ensemble8 | 0.0000 | 0.0063 | 0.0048 | -0.099 |
| smoothed | 4 | plain | 0.0000 | 0.0073 | 0.0073 | -0.627 |
| smoothed | 4 | soft | 0.0000 | 0.0072 | 0.0072 | -0.570 |
| smoothed | 4 | feather_L | 0.0000 | 0.0076 | 0.0073 | -0.618 |
| smoothed | 4 | ensemble8 | 0.0000 | 0.0082 | 0.0062 | -0.190 |
| smoothed | 8 | plain | 0.0000 | 0.0076 | 0.0084 | -0.815 |
| smoothed | 8 | soft | 0.0000 | 0.0075 | 0.0081 | -0.697 |
| smoothed | 8 | feather_L | 0.0000 | 0.0084 | 0.0084 | -0.802 |
| smoothed | 8 | ensemble8 | 0.0000 | 0.0088 | 0.0075 | -0.076 |
| threshold | 2 | plain | 0.0006 | 0.0085 | 0.0086 | -1.229 |
| threshold | 2 | soft | 0.0006 | 0.0085 | 0.0086 | -1.229 |
| threshold | 2 | feather_L | 0.0006 | 0.0080 | 0.0086 | -1.229 |
| threshold | 2 | ensemble8 | 0.0007 | 0.0075 | 0.0071 | -1.005 |
| threshold | 4 | plain | 0.0015 | 0.0134 | 0.0109 | -1.733 |
| threshold | 4 | soft | 0.0015 | 0.0134 | 0.0109 | -1.733 |
| threshold | 4 | feather_L | 0.0015 | 0.0129 | 0.0109 | -1.733 |
| threshold | 4 | ensemble8 | 0.0018 | 0.0131 | 0.0100 | -1.628 |
| threshold | 8 | plain | 0.0017 | 0.0175 | 0.0129 | -1.636 |
| threshold | 8 | soft | 0.0016 | 0.0175 | 0.0129 | -1.636 |
| threshold | 8 | feather_L | 0.0016 | 0.0170 | 0.0129 | -1.634 |
| threshold | 8 | ensemble8 | 0.0018 | 0.0159 | 0.0110 | -1.552 |

Total time 285 s.
