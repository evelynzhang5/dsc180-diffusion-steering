# Early check: ../outputs/q2_pilot2/two_object/main

## Early previews (192 baselines)

| checkpoint | preview passes | agrees with final | AUC: final pass | AUC: fail is repairable (n) | AUC: pass gets harmed (n) |
| --- | --- | --- | --- | --- | --- |
| 6 | 26% | 51% | 0.79 | 0.67 (13 of 49) | 0.72 (12 of 143) |
| 15 | 70% | 93% | 0.96 | 0.73 (4 of 49) | 0.79 (4 of 143) |
| 24 | 76% | 98% | 0.99 | 0.91 (4 of 49) | 0.91 (1 of 143) |

AUC for 'repairable': does a *higher* preview margin among failures mean a nudge is more likely to fix it (above 0.5) or less likely (below 0.5)? For 'harmed', above 0.5 means images with a higher preview margin are harmed less.

## Gated policies: nudge only when the preview at that checkpoint fails

| policy | applied to | success |
| --- | --- | --- |
| no intervention | 0% | 74.5% |
| step 6, targeted:15: always / gated | 100% / 74% | 75.0% / **75.5%** |
| step 24, targeted:15: always / gated | 100% / 24% | 76.0% / **75.5%** |
| step 24, global:14: always / gated | 100% / 24% | 76.0% / **75.0%** |
| step 15, targeted:15: always / gated | 100% / 30% | 75.0% / **75.0%** |
| step 15, global:14: always / gated | 100% / 30% | 75.0% / **75.0%** |
| step 6, global:14: always / gated | 100% / 74% | 74.5% / **74.5%** |
| per-image hindsight oracle | | 81.8% |

Gating uses only information available at the decision step, so it is a real (if simple) policy. The rows are chosen and evaluated on the same images, so treat the best row as optimistic.
