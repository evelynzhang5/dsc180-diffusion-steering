# Smoke test report

Detection threshold 0.2. 192 baseline trajectories, 1152 branches.

## Baseline checks

| check | pass rate |
| --- | --- |
| obj_1 | 69.8% |
| obj_2 | 81.2% |
| objects | 51.0% |
| color_1 | 47.4% |
| color_2 | 40.1% |
| relation | 24.5% |
| all | 4.2% |

## Each action at each checkpoint (vs. no action)

Rates are over all trajectories; 'repair'/'harm' change the all-requirements outcome; 'collateral' fixes the relation but loses an object or color.

| checkpoint | action | success | repair | harm | collateral | no change | pulse size | U-Net evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| – | none (baseline) | 4.2% | | | | | | 60 |
| 6 | global:14 | 2.6% | 0.5% | 2.1% | 1.6% | 95.8% | 0.079 | 48 |
| 6 | targeted:40 | 3.6% | 2.1% | 2.6% | 4.2% | 91.1% | 0.070 | 51 |
| 15 | global:14 | 5.2% | 1.6% | 0.5% | 1.0% | 96.9% | 0.080 | 30 |
| 15 | targeted:40 | 4.7% | 0.5% | 0.0% | 0.0% | 99.5% | 0.066 | 33 |
| 24 | global:14 | 5.2% | 1.6% | 0.5% | 0.5% | 97.4% | 0.073 | 12 |
| 24 | targeted:40 | 4.7% | 1.0% | 0.5% | 0.0% | 98.4% | 0.064 | 15 |

## Oracle headroom

| policy | success |
| --- | --- |
| no intervention | 4.2% |
| best fixed policy, chosen on these same trajectories (optimistic) | 5.2% ((15, 'global:14')) |
| best fixed policy chosen on dev prompts | 2.6% ((6, 'global:14')) |
| per-sample hindsight oracle | 7.8% |

**Oracle minus best fixed policy: 2.6 points** (95% interval over prompts 0.5 to 4.2).

## Where the oracle gain comes from

| per-sample choice | success |
| --- | --- |
| whether to apply the best fixed action | 5.7% |
| when (one action, choose checkpoint) | 7.3% |
| which action (one checkpoint, choose action) | 6.2% |
| everything (oracle) | 7.8% |
