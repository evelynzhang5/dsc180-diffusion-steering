# Smoke test report

Detection threshold 0.2. 192 baseline trajectories, 1152 branches.

## Baseline checks

| check | pass rate |
| --- | --- |
| obj_1 | 87.0% |
| obj_2 | 87.5% |
| all | 74.5% |

## Each action at each checkpoint (vs. no action)

Rates are over all trajectories; 'repair'/'harm' change the all-requirements outcome; 'collateral' fixes one check but loses another, leaving the outcome unchanged.

| checkpoint | action | success | repair | harm | collateral | no change | pulse size | U-Net evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| – | none (baseline) | 74.5% | | | | | | 60 |
| 6 | global:14 | 74.5% | 3.1% | 3.1% | 0.0% | 93.8% | 0.096 | 48 |
| 6 | targeted:15 | 75.0% | 5.7% | 5.2% | 0.0% | 89.1% | 0.149 | 54 |
| 15 | global:14 | 75.0% | 1.0% | 0.5% | 0.0% | 98.4% | 0.124 | 30 |
| 15 | targeted:15 | 75.0% | 2.1% | 1.6% | 0.0% | 96.4% | 0.168 | 36 |
| 24 | global:14 | 76.0% | 1.6% | 0.0% | 0.0% | 98.4% | 0.136 | 12 |
| 24 | targeted:15 | 76.0% | 2.1% | 0.5% | 0.0% | 97.4% | 0.173 | 18 |

## Oracle headroom

| policy | success |
| --- | --- |
| no intervention | 74.5% |
| best fixed policy, chosen on these same trajectories (optimistic) | 76.0% ((24, 'global:14')) |
| best fixed policy chosen on dev prompts | 75.0% ((15, 'global:14')) |
| per-sample hindsight oracle | 81.8% |

**Oracle minus best fixed policy: 5.7 points** (95% interval over prompts 2.1 to 7.8).

## Where the oracle gain comes from

| per-sample choice | success |
| --- | --- |
| whether to apply the best fixed action | 76.0% |
| when (one action, choose checkpoint) | 81.2% |
| which action (one checkpoint, choose action) | 81.2% |
| everything (oracle) | 81.8% |
