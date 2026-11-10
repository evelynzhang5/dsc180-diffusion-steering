# Scorer check: ../outputs/q2_pilot2/two_object/main

## A. Margins: counting only outcome changes that clear the cutoff by delta

384 baselines, 2304 real-nudge branches, 97 outcome flips; 50 of them (52%) have the baseline or the branch within 0.1 of the cutoff.

| delta | repairs | harms | no intervention | best fixed | oracle | **oracle - best fixed** (95% interval) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 55 | 42 | 74.5% | 76.6% | 82.3% | **5.7** (2.3 to 8.1) |
| 0.05 | 36 | 31 | 74.5% | 75.8% | 80.5% | **4.7** (1.8 to 6.5) |
| 0.1 | 27 | 20 | 74.5% | 75.3% | 79.2% | **3.9** (1.3 to 5.2) |
| 0.15 | 17 | 14 | 74.5% | 75.3% | 78.1% | **2.9** (1.0 to 4.4) |

delta 0 is the original analysis (the 'best fixed' here is chosen on the same trajectories, so it is the optimistic version). If the gap shrinks quickly as delta grows, most of it was borderline verdicts.

## Real nudges vs. placebo, counting only outcome changes that clear the cutoff by delta

| delta | real-nudge oracle | placebo oracle | **real minus placebo** (95% interval over prompts) |
| --- | --- | --- | --- |
| 0 | 82.3% | 80.5% | **+1.8** (-0.8 to +5.2) |
| 0.05 | 80.5% | 78.6% | **+1.8** (-0.3 to +4.4) |
| 0.1 | 79.2% | 77.9% | **+1.3** (-0.3 to +3.1) |
| 0.15 | 78.1% | 77.1% | **+1.0** (-0.3 to +3.1) |
