# Scorer check: ../outputs/q2_pilot2/two_object/main

## A. Margins: counting only outcome changes that clear the cutoff by delta

192 baselines, 1152 real-nudge branches, 51 outcome flips; 24 of them (47%) have the baseline or the branch within 0.1 of the cutoff.

| delta | repairs | harms | no intervention | best fixed | oracle | **oracle - best fixed** (95% interval) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 30 | 21 | 74.5% | 76.0% | 81.8% | **5.7** (2.1 to 7.8) |
| 0.05 | 20 | 18 | 74.5% | 75.0% | 80.2% | **5.2** (1.6 to 6.8) |
| 0.1 | 15 | 12 | 74.5% | 75.5% | 79.2% | **3.6** (1.0 to 5.7) |
| 0.15 | 7 | 10 | 74.5% | 75.0% | 77.6% | **2.6** (0.5 to 4.7) |

delta 0 is the original analysis (the 'best fixed' here is chosen on the same trajectories, so it is the optimistic version). If the gap shrinks quickly as delta grows, most of it was borderline verdicts.

## B. Stability: re-scoring after a mirror flip and a 90% center crop

| images | verdict same under flip | same under crop | same under both |
| --- | --- | --- | --- |
| all baselines (n=192) | 99% | 97% | 97% |
| baselines within 0.1 of the cutoff (n=15) | 93% | 73% | 73% |
| flipped branches (n=51) | 82% | 86% | 80% |

| outcome change | holds in all three views |
| --- | --- |
| harms | 16 of 21 (76%) |
| repairs | 15 of 30 (50%) |
