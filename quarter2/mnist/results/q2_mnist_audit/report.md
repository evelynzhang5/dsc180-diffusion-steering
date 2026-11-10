# MNIST side-effect audit

30 (reference set, split) pairs x 7 attributes = 210 items. Effects per 1 SD of achieved thickness change. Audit on 5 classes, truth on the other 5 (seed 12, 200 per class). Held-out thickness change: +0.30 SD on average (range +0.24 to +0.36).

## How well each audit predicts held-out side effects

| tau | budget per class | audit | true effects | flagged | hits | false alarms | precision | recall | rank corr. with truth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | - | REF | 108 | 129 | 96 | 33 | 0.74 | 0.89 | 0.63 |
| 0.1 | 20 | OUT-1 | 108 | 144 | 105 | 39 | 0.73 | 0.97 | 0.79 |
| 0.1 | 20 | OUT-rep | 108 | 62 | 62 | 0 | 1.00 | 0.57 | 0.72 |
| 0.1 | 20 | BOTH | 108 | 61 | 61 | 0 | 1.00 | 0.56 | 0.72 |
| 0.1 | 100 | OUT-1 | 108 | 134 | 96 | 38 | 0.72 | 0.89 | 0.82 |
| 0.1 | 100 | OUT-rep | 108 | 76 | 73 | 3 | 0.96 | 0.68 | 0.75 |
| 0.1 | 100 | BOTH | 108 | 71 | 69 | 2 | 0.97 | 0.64 | 0.75 |
| 0.15 | - | REF | 89 | 110 | 77 | 33 | 0.70 | 0.87 | 0.63 |
| 0.15 | 20 | OUT-1 | 89 | 129 | 87 | 42 | 0.67 | 0.98 | 0.79 |
| 0.15 | 20 | OUT-rep | 89 | 62 | 62 | 0 | 1.00 | 0.70 | 0.72 |
| 0.15 | 20 | BOTH | 89 | 60 | 60 | 0 | 1.00 | 0.67 | 0.72 |
| 0.15 | 100 | OUT-1 | 89 | 105 | 83 | 22 | 0.79 | 0.93 | 0.82 |
| 0.15 | 100 | OUT-rep | 89 | 75 | 72 | 3 | 0.96 | 0.81 | 0.75 |
| 0.15 | 100 | BOTH | 89 | 67 | 66 | 1 | 0.99 | 0.74 | 0.75 |
| 0.25 | - | REF | 69 | 87 | 58 | 29 | 0.67 | 0.84 | 0.63 |
| 0.25 | 20 | OUT-1 | 69 | 103 | 65 | 38 | 0.63 | 0.94 | 0.79 |
| 0.25 | 20 | OUT-rep | 69 | 57 | 53 | 4 | 0.93 | 0.77 | 0.72 |
| 0.25 | 20 | BOTH | 69 | 50 | 50 | 0 | 1.00 | 0.72 | 0.72 |
| 0.25 | 100 | OUT-1 | 69 | 72 | 67 | 5 | 0.93 | 0.97 | 0.82 |
| 0.25 | 100 | OUT-rep | 69 | 65 | 62 | 3 | 0.95 | 0.90 | 0.75 |
| 0.25 | 100 | BOTH | 69 | 55 | 55 | 0 | 1.00 | 0.80 | 0.75 |

## Planted nuisance: reference difference vs. effect on held-out classes (per SD thickness, mean of splits)

| planted | q | reference difference | OUT-rep on audit classes (budget max) | true held-out effect | true effect on other attributes, largest |
| --- | --- | --- | --- | --- | --- |
| natural | - | - | - | - | darkness +0.90 |
| slant | 1.0 | +0.84 | +1.26 | +1.27 | width +0.63 |
| slant | 0.75 | +0.38 | +0.38 | +0.39 | darkness +0.86 |
| width | 1.0 | +0.92 | +0.67 | +0.72 | darkness +0.87 |
| width | 0.75 | +0.50 | +0.41 | +0.41 | darkness +0.89 |
| height | 1.0 | +0.38 | +0.15 | +0.17 | darkness +0.89 |
| height | 0.75 | +0.30 | +0.13 | +0.15 | darkness +0.91 |
| cx | 1.0 | +0.86 | -0.18 | -0.17 | darkness +0.91 |
| cx | 0.75 | +0.40 | -0.06 | -0.07 | darkness +0.88 |
| cy | 1.0 | +0.83 | -0.07 | -0.05 | darkness +0.91 |
| cy | 0.75 | +0.44 | -0.03 | +0.00 | darkness +0.91 |
| length | 1.0 | +0.74 | +0.56 | +0.68 | darkness +0.87 |
| length | 0.75 | +0.25 | +0.29 | +0.35 | darkness +0.88 |
| darkness | 1.0 | +0.93 | +0.92 | +0.90 | width +0.26 |
| darkness | 0.75 | +0.67 | +0.95 | +0.87 | width +0.27 |

## Natural set: every attribute (per SD thickness, mean of splits)

| attribute | reference difference | OUT-rep (budget max) | true held-out effect |
| --- | --- | --- | --- |
| slant | +0.07 | +0.05 | +0.06 |
| width | +0.32 | +0.32 | +0.32 |
| height | +0.07 | +0.10 | +0.11 |
| cx | +0.01 | +0.03 | +0.04 |
| cy | -0.04 | +0.06 | +0.10 |
| length | -0.25 | +0.07 | +0.09 |
| darkness | +0.82 | +0.92 | +0.90 |

## Does the audit-class effect transfer? Sign agreement, audit vs. held-out classes

Items where the true held-out effect clears tau = 0.15 and its interval excludes 0: share where OUT-rep (budget 100) on the audit classes has the same sign, and median |audit - truth|.

89 items: same sign 98%, median |audit - truth| 0.10.
