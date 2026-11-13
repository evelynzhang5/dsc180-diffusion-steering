# Stage A: NA x RFM window (seeds 0-255)

Settings: lam0_all = RFM alone (paper), lam3_late = NA-RFM (paper); lam0_late = RFM alone but only for sigma <= 11.69; lam3_all = NA plus RFM at every step.

## All classes

Accuracy % / FID per class; last columns: mean over the listed classes.

| setting | NA lambda | RFM window | w | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck | mean acc | mean FID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lam0_all_w1 | 0 | all | 1 | 97 / 106 | 100 / 63 | 93 / 116 | 100 / 118 | 96 / 97 | 98 / 101 | 97 / 111 | 99 / 84 | 93 / 82 | 90 / 70 | 96.3 | 94.9 |
| lam0_late_w1 | 0 | late | 1 | 96 / 108 | 100 / 65 | 92 / 116 | 99 / 118 | 96 / 97 | 98 / 100 | 97 / 110 | 99 / 87 | 93 / 85 | 90 / 71 | 95.9 | 95.6 |
| lam3_all_w1 | 3 | all | 1 | 99 / 99 | 100 / 67 | 95 / 120 | 100 / 110 | 100 / 98 | 98 / 109 | 100 / 103 | 99 / 85 | 84 / 74 | 85 / 81 | 95.9 | 94.7 |
| lam3_late_w1 | 3 | late | 1 | 99 / 98 | 100 / 66 | 96 / 119 | 100 / 109 | 100 / 100 | 96 / 107 | 100 / 102 | 98 / 83 | 84 / 74 | 83 / 83 | 95.7 | 94.1 |

