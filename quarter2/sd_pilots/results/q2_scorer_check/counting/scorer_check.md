# Scorer check: ../outputs/q2_pilot2/counting/main

## A. Margins: counting only outcome changes that clear the cutoff by delta

144 baselines, 864 real-nudge branches, 48 outcome flips; 41 of them (85%) have the baseline or the branch within 0.1 of the cutoff.

| delta | repairs | harms | no intervention | best fixed | oracle | **oracle - best fixed** (95% interval) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 24 | 24 | 52.1% | 54.2% | 62.5% | **8.3** (2.8 to 11.1) |
| 0.05 | 9 | 7 | 52.1% | 52.8% | 55.6% | **2.8** (0.0 to 4.2) |
| 0.1 | 3 | 4 | 52.1% | 52.8% | 53.5% | **0.7** (0.0 to 1.4) |
| 0.15 | 1 | 0 | 52.1% | 52.8% | 52.8% | **0.0** (0.0 to 0.0) |

delta 0 is the original analysis (the 'best fixed' here is chosen on the same trajectories, so it is the optimistic version). If the gap shrinks quickly as delta grows, most of it was borderline verdicts.

## B. Stability: re-scoring after a mirror flip and a 90% center crop

| images | verdict same under flip | same under crop | same under both |
| --- | --- | --- | --- |
| all baselines (n=144) | 92% | 91% | 87% |
| baselines within 0.1 of the cutoff (n=43) | 72% | 77% | 63% |
| flipped branches (n=48) | 67% | 60% | 48% |

| outcome change | holds in all three views |
| --- | --- |
| harms | 4 of 24 (17%) |
| repairs | 9 of 24 (38%) |
