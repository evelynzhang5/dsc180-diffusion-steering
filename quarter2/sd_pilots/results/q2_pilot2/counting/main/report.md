# Smoke test report

Detection threshold 0.2. 192 baseline trajectories, 1152 branches.

## Baseline checks

| check | pass rate |
| --- | --- |
| obj_1 | 100.0% |
| count | 41.1% |
| under | 14.1% |
| over | 44.8% |
| all | 41.1% |

## Each action at each checkpoint (vs. no action)

Rates are over all trajectories; 'repair'/'harm' change the all-requirements outcome; 'collateral' fixes one check but loses another, leaving the outcome unchanged.

| checkpoint | action | success | repair | harm | collateral | no change | pulse size | U-Net evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| – | none (baseline) | 41.1% | | | | | | 60 |
| 6 | global:20 | 42.2% | 5.2% | 4.2% | 0.0% | 90.6% | 0.156 | 48 |
| 6 | targeted:15 | 43.2% | 5.7% | 3.6% | 0.0% | 90.6% | 0.048 | 54 |
| 15 | global:20 | 42.7% | 4.2% | 2.6% | 0.0% | 93.2% | 0.196 | 30 |
| 15 | targeted:15 | 41.7% | 1.6% | 1.0% | 0.0% | 97.4% | 0.052 | 36 |
| 24 | global:20 | 40.6% | 1.6% | 2.1% | 0.0% | 96.4% | 0.227 | 12 |
| 24 | targeted:15 | 40.1% | 0.0% | 1.0% | 0.0% | 99.0% | 0.057 | 18 |

## Oracle headroom

| policy | success |
| --- | --- |
| no intervention | 41.1% |
| best fixed policy, chosen on these same trajectories (optimistic) | 43.2% ((6, 'targeted:15')) |
| best fixed policy chosen on dev prompts | 43.2% ((6, 'targeted:15')) |
| per-sample hindsight oracle | 51.6% |

**Oracle minus best fixed policy: 8.3 points** (95% interval over prompts 3.6 to 10.9).

## Where the oracle gain comes from

| per-sample choice | success |
| --- | --- |
| whether to apply the best fixed action | 46.9% |
| when (one action, choose checkpoint) | 48.4% |
| which action (one checkpoint, choose action) | 49.5% |
| everything (oracle) | 51.6% |
