# Smoke test report

Detection threshold 0.2. 288 baseline trajectories, 3456 branches.

## Baseline checks

| check | pass rate |
| --- | --- |
| obj_1 | 100.0% |
| count | 54.9% |
| under | 18.4% |
| over | 26.7% |
| all | 54.9% |

## Each action at each checkpoint (vs. no action)

Rates are over all trajectories; 'repair'/'harm' change the all-requirements outcome; 'collateral' fixes one check but loses another, leaving the outcome unchanged.

| checkpoint | action | success | repair | harm | collateral | no change | pulse size | U-Net evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| – | none (baseline) | 54.9% | | | | | | 60 |
| 6 | global:20 | 56.9% | 9.0% | 6.9% | 0.0% | 84.0% | 0.153 | 48 |
| 6 | placebo:0.15 | 54.9% | 4.2% | 4.2% | 0.0% | 91.7% | 0.150 | 48 |
| 6 | placebo:0.5 | 55.2% | 5.6% | 5.2% | 0.0% | 89.2% | 0.500 | 48 |
| 6 | targeted:15 | 60.1% | 10.1% | 4.9% | 0.0% | 85.1% | 0.052 | 54 |
| 15 | global:20 | 56.9% | 4.5% | 2.4% | 0.0% | 93.1% | 0.204 | 30 |
| 15 | placebo:0.15 | 55.6% | 1.4% | 0.7% | 0.0% | 97.9% | 0.150 | 30 |
| 15 | placebo:0.5 | 54.2% | 1.7% | 2.4% | 0.0% | 95.8% | 0.500 | 30 |
| 15 | targeted:15 | 56.6% | 2.4% | 0.7% | 0.0% | 96.9% | 0.056 | 36 |
| 24 | global:20 | 54.2% | 1.7% | 2.4% | 0.0% | 95.8% | 0.246 | 12 |
| 24 | placebo:0.15 | 55.6% | 1.4% | 0.7% | 0.0% | 97.9% | 0.150 | 12 |
| 24 | placebo:0.5 | 55.2% | 1.0% | 0.7% | 0.0% | 98.3% | 0.500 | 12 |
| 24 | targeted:15 | 54.5% | 1.0% | 1.4% | 0.0% | 97.6% | 0.064 | 18 |

## Oracle headroom

| policy | success |
| --- | --- |
| no intervention | 54.9% |
| best fixed policy, chosen on these same trajectories (optimistic) | 60.1% ((6, 'targeted:15')) |
| per-sample hindsight oracle | 71.9% |

**Oracle minus best fixed policy: 11.8 points** (95% interval over prompts 6.6 to 16.3).

## Picked nudges vs. control menus (hindsight oracle, same saved states)

The picked nudges cost 198 extra U-Net calls per image in total; re-rolling uses 3 other seeds of the same prompt (180 calls).

| menu | options per image | hindsight oracle | picked minus this menu (95% interval over prompts) |
| --- | --- | --- | --- |
| no intervention | 0 | 54.9% | |
| nudges picked on dev | 6 | 70.1% | |
| placebo (random direction) | 6 | 62.5% | **+7.6** (+4.5 to +11.1) |
| re-roll the seed, best of 4 seeds | 3 | 88.2% | **-18.1** (-24.7 to -11.5) |

Near zero against placebo means the per-image gain is what any re-roll of the image would give. Against wrong sign, near zero means the direction of the push does not matter.

| action | mean image change (0-255 per pixel, vs. no action) |
| --- | --- |
| global:20 | 3.9 |
| placebo:0.15 | 0.8 |
| placebo:0.5 | 2.3 |
| targeted:15 | 2.5 |

## Where the oracle gain comes from

| per-sample choice | success |
| --- | --- |
| whether to apply the best fixed action | 64.9% |
| when (one action, choose checkpoint) | 66.0% |
| which action (one checkpoint, choose action) | 69.4% |
| everything (oracle) | 71.9% |
