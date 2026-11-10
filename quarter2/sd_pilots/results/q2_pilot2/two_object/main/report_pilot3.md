# Smoke test report

Detection threshold 0.2. 384 baseline trajectories, 4608 branches.

## Baseline checks

| check | pass rate |
| --- | --- |
| obj_1 | 87.5% |
| obj_2 | 87.0% |
| all | 74.5% |

## Each action at each checkpoint (vs. no action)

Rates are over all trajectories; 'repair'/'harm' change the all-requirements outcome; 'collateral' fixes one check but loses another, leaving the outcome unchanged.

| checkpoint | action | success | repair | harm | collateral | no change | pulse size | U-Net evals |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| – | none (baseline) | 74.5% | | | | | | 60 |
| 6 | global:14 | 72.9% | 2.6% | 4.2% | 0.0% | 93.2% | 0.095 | 48 |
| 6 | placebo:0.15 | 75.5% | 2.6% | 1.6% | 0.0% | 95.8% | 0.150 | 48 |
| 6 | placebo:0.5 | 76.6% | 5.2% | 3.1% | 0.0% | 91.7% | 0.500 | 48 |
| 6 | targeted:15 | 76.6% | 6.8% | 4.7% | 0.0% | 88.5% | 0.149 | 54 |
| 15 | global:14 | 75.3% | 1.3% | 0.5% | 0.0% | 98.2% | 0.127 | 30 |
| 15 | placebo:0.15 | 75.3% | 1.0% | 0.3% | 0.0% | 98.7% | 0.150 | 30 |
| 15 | placebo:0.5 | 76.3% | 2.1% | 0.3% | 0.0% | 97.7% | 0.500 | 30 |
| 15 | targeted:15 | 75.0% | 1.6% | 1.0% | 0.0% | 97.4% | 0.171 | 36 |
| 24 | global:14 | 75.3% | 1.0% | 0.3% | 0.0% | 98.7% | 0.138 | 12 |
| 24 | placebo:0.15 | 75.0% | 0.8% | 0.3% | 0.0% | 99.0% | 0.150 | 12 |
| 24 | placebo:0.5 | 74.7% | 0.5% | 0.3% | 0.0% | 99.2% | 0.500 | 12 |
| 24 | targeted:15 | 75.3% | 1.0% | 0.3% | 0.0% | 98.7% | 0.174 | 18 |

## Oracle headroom

| policy | success |
| --- | --- |
| no intervention | 74.5% |
| best fixed policy, chosen on these same trajectories (optimistic) | 76.6% ((6, 'placebo:0.5')) |
| per-sample hindsight oracle | 83.9% |

**Oracle minus best fixed policy: 7.3 points** (95% interval over prompts 2.9 to 9.4).

## Picked nudges vs. control menus (hindsight oracle, same saved states)

The picked nudges cost 198 extra U-Net calls per image in total; re-rolling uses 3 other seeds of the same prompt (180 calls).

| menu | options per image | hindsight oracle | picked minus this menu (95% interval over prompts) |
| --- | --- | --- | --- |
| no intervention | 0 | 74.5% | |
| nudges picked on dev | 6 | 82.3% | |
| placebo (random direction) | 6 | 80.5% | **+1.8** (-0.8 to +5.2) |
| re-roll the seed, best of 4 seeds | 3 | 95.8% | **-13.5** (-18.5 to -8.6) |

Near zero against placebo means the per-image gain is what any re-roll of the image would give. Against wrong sign, near zero means the direction of the push does not matter.

| action | mean image change (0-255 per pixel, vs. no action) |
| --- | --- |
| global:14 | 3.0 |
| placebo:0.15 | 1.0 |
| placebo:0.5 | 2.8 |
| targeted:15 | 4.7 |

## Where the oracle gain comes from

| per-sample choice | success |
| --- | --- |
| whether to apply the best fixed action | 79.7% |
| when (one action, choose checkpoint) | 82.0% |
| which action (one checkpoint, choose action) | 83.6% |
| everything (oracle) | 83.9% |
