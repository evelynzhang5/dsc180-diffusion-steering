# Early check: ../outputs/q2_pilot2/counting/main

## Early previews (144 baselines)

| checkpoint | preview passes | agrees with final | AUC: final pass | AUC: fail is repairable (n) | AUC: pass gets harmed (n) |
| --- | --- | --- | --- | --- | --- |
| 6 | 44% | 76% | 0.83 | 0.67 (12 of 69) | 0.80 (10 of 75) |
| 15 | 56% | 90% | 0.96 | 0.98 (7 of 69) | 0.92 (5 of 75) |
| 24 | 54% | 95% | 0.99 | 0.90 (1 of 69) | 0.94 (4 of 75) |

AUC for 'repairable': does a *higher* preview margin among failures mean a nudge is more likely to fix it (above 0.5) or less likely (below 0.5)? For 'harmed', above 0.5 means images with a higher preview margin are harmed less.

## Gated policies: nudge only when the preview at that checkpoint fails

| policy | applied to | success |
| --- | --- | --- |
| no intervention | 0% | 52.1% |
| step 6, targeted:15: always / gated | 100% / 56% | 54.2% / **53.5%** |
| step 6, global:20: always / gated | 100% / 56% | 52.8% / **52.8%** |
| step 24, targeted:15: always / gated | 100% / 46% | 50.0% / **51.4%** |
| step 24, global:20: always / gated | 100% / 46% | 50.7% / **51.4%** |
| step 15, targeted:15: always / gated | 100% / 44% | 51.4% / **51.4%** |
| step 15, global:20: always / gated | 100% / 44% | 53.5% / **51.4%** |
| per-image hindsight oracle | | 62.5% |

Gating uses only information available at the decision step, so it is a real (if simple) policy. The rows are chosen and evaluated on the same images, so treat the best row as optimistic.
