# Scorer check: ../outputs/q2_pilot2/counting/main

## A. Margins: counting only outcome changes that clear the cutoff by delta

288 baselines, 1728 real-nudge branches, 135 outcome flips; 115 of them (85%) have the baseline or the branch within 0.1 of the cutoff.

| delta | repairs | harms | no intervention | best fixed | oracle | **oracle - best fixed** (95% interval) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 79 | 56 | 55.2% | 60.1% | 70.1% | **10.1** (5.6 to 13.9) |
| 0.05 | 22 | 27 | 55.2% | 55.9% | 60.1% | **4.2** (1.0 to 5.9) |
| 0.1 | 8 | 12 | 55.2% | 55.2% | 57.3% | **2.1** (0.3 to 3.1) |
| 0.15 | 4 | 3 | 55.2% | 55.6% | 56.6% | **1.0** (0.0 to 2.1) |

delta 0 is the original analysis (the 'best fixed' here is chosen on the same trajectories, so it is the optimistic version). If the gap shrinks quickly as delta grows, most of it was borderline verdicts.

## Real nudges vs. placebo, counting only outcome changes that clear the cutoff by delta

| delta | real-nudge oracle | placebo oracle | **real minus placebo** (95% interval over prompts) |
| --- | --- | --- | --- |
| 0 | 70.1% | 62.5% | **+7.6** (+4.5 to +11.1) |
| 0.05 | 60.1% | 58.0% | **+2.1** (+0.0 to +4.2) |
| 0.1 | 57.3% | 56.6% | **+0.7** (-1.0 to +2.4) |
| 0.15 | 56.6% | 55.2% | **+1.4** (+0.3 to +2.8) |
