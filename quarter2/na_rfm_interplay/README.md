# Q2 round 7: when do noise alignment and RFM help or interfere? (CIFAR-10, NA-RFM)

Branch `q2-na-rfm-interplay`. Started Nov 12, 2026. The index of all Q2 rounds is in [../README.md](../README.md).

**Why.** After rounds 5-6 (auditing what steering changes) the project returns to improving steering itself,
starting from a concrete failure in our replication (`cifar10_replication/`, 256 images per class, seeds 0-255):
NA-RFM was below RFM alone on ship (84% vs. 93%) and truck (83% vs. 90%), while it helped on deer (100% vs. 96%) and
bird (96% vs. 93%). NA-RFM is designed so that noise alignment (NA) gives coarse control at high noise and RFM
steering refines at lower noise, so the two should complement each other.

**Question.** Why does adding NA sometimes make RFM steering worse, and does fair tuning remove the problem? If
ordinary tuning fixes it, that is a replication finding and we move on (next candidate: learning directions from
their effect on generated images). If it persists, the next step is to test whether RFM needs to be calibrated on
trajectories that already include NA.

NA-RFM differs from RFM alone in **two** ways: it adds NA (lambda 3 for sigma >= 3.33), and it applies RFM only for
sigma <= 11.69 instead of at every step. The first stage separates the two.

## Design (fixed on Nov 12, 2026, before any image of this round was generated)

Everything uses the paper's code (`run_cifar10_na_rfm.py`, the replication's RFM directions at block 9, CFG-style
scale s = 2, 100 DDIM steps) and the paper's evaluation (ConvNeXt accuracy; per-class FID against the class's 6,000
CIFAR-10 images). FID at 256 images per class is biased upward and noisy, so it is only compared at equal image
counts and used for screening; the confirmation uses 1,024 per class.

| stage | what | classes | seeds |
|---|---|---|---|
| A `decompose` | 2 x 2: NA {off, lambda 3} x RFM window {every step, sigma <= 11.69}, w = 1 | all ten | 0-255 |
| B `tune` | lambda {0, 1, 3} x window {every step, sigma <= 11.69} x RFM strength w {0.5, 1, 2} (18 settings) | ship, truck, deer | 0-255 |
| `select` | best combined (lambda > 0) and best RFM-only (lambda = 0) setting, by the rule below; frozen with a sha256 | | |
| C `confirm` | the two selected settings and the two paper settings | ship, truck, deer | 1000-2023 (new) |

The paper settings on seeds 0-255 are copied from the replication's samples (same code and flags); everything else
is generated here. NA's end point (sigma 3.33) is kept at the paper's value.

**Reading stage A.** If RFM alone with the late window loses accuracy on ship and truck while NA plus RFM at every
step does not, the drop comes from starting RFM late, not from NA: the components are not interfering. If NA plus
RFM at every step loses accuracy, NA itself hurts those classes. Both can contribute.

**Selection rule (stage B).** Within each family, the setting with the highest mean accuracy over ship, truck and
deer among those whose mean FID is at most the paper RFM-only setting's mean FID + 5 (ties: lower FID). The full
accuracy-FID table for both families is reported, not only the picks.

**Decision (stage C, 1,024 images per class).**
- Tuned combined within 2 points of tuned RFM-only (or above) on both ship and truck, with mean FID at most 2
  higher: ordinary tuning fixes it. Record as a replication finding; stop this line and move to learning directions
  from generation effects.
- Tuned combined 3 or more points below on ship or truck: the interference persists after tuning. Next: test
  trajectory-dependent calibration (RFM direction and strength estimated on NA trajectories).
- Otherwise: inconclusive at this sample size; report it, and decide between the two with the stage A result.

Raw vector cosines between NA and RFM are not used as a conflict measure: they act in different spaces (NA on the
noise prediction, RFM on block-9 activations).

## Status

- Nov 12, 2026: design and code written; first pod: smoke test, then stages A and B.
- Nov 12, 2026: stages A and B done (dev seeds 0-255); selection frozen; stage C pod launched.
- Nov 12, 2026: stage C done. **Closed for the tested setup: joint strength tuning fixes it.** Next (round 8,
  branch `q2-direction-choice`): with strength tuned, does a different direction improve the accuracy-FID trade-off?

## Result (stage C: seeds 1000-2023, 1,024 images per class)

Accuracy % (95% CI) / FID against the class's 6,000 CIFAR-10 images:

| setting | ship | truck | deer | mean FID |
|---|---|---|---|---|
| RFM only (paper = tuned RFM-only) | 94.5 (±1.4) / 46.0 | 91.6 (±1.7) / 36.3 | 97.9 (±0.9) / 48.7 | 43.7 |
| NA-RFM (paper, w 1) | 84.8 (±2.2) / 47.1 | 85.7 (±2.1) / 47.4 | 99.7 (±0.3) / 55.0 | 49.8 |
| NA-RFM, w 0.5 (tuned) | 98.0 (±0.8) / 40.3 | 97.9 (±0.9) / 27.6 | 99.0 (±0.6) / 47.7 | 38.5 |

**Decision (rule fixed in advance):** tuned NA-RFM is above tuned RFM-only on both ship (+3.5) and truck (+6.3), and
its mean FID is 5.2 lower (43.7 → 38.5, about 12%), not higher. The frozen choice held on fresh seeds.

What this establishes: the two components **do need joint strength tuning**. The paper's RFM strength (w = 1) is
right for RFM alone but too strong once NA is on; lowering it to 0.5 improves the combination on all three classes
in both accuracy and FID. What they **do not** need is a new RFM direction learned on NA trajectories (the
trajectory-dependent calibration this round would have tested next). Stage A rules out the late RFM window as the
cause. A plausible explanation, not tested directly, is that NA already does part of the steering, so full-strength
RFM on top over-steers ship and truck.

The FID gain is a distribution-level improvement under this metric at equal sample counts (1,024 per class); it does
not mean every image or every class is better.

Recorded as a replication finding; this line stops here for the tested setup.

Caveats: one model (CIFAR-10), one classifier (ConvNeXt) and three classes were tuned; w was tuned on a coarse grid
{0.5, 1, 2}, and the other seven classes were not re-checked at w 0.5. NA's strength and end point were kept at
the paper's values except for lambda in stage B.

## Dev results (seeds 0-255, 256 images per class)

**Stage A: the drop comes from NA, not from the late RFM window.** Accuracy % / FID:

| setting | ship | truck | deer | mean acc (10 classes) | mean FID (10 classes) |
|---|---|---|---|---|---|
| RFM every step (paper RFM only) | 93 / 82 | 90 / 70 | 96 / 97 | 96.3 | 94.9 |
| RFM late only | 93 / 85 | 90 / 71 | 96 / 97 | 95.9 | 95.6 |
| NA + RFM every step | 84 / 74 | 85 / 81 | 100 / 98 | 95.9 | 94.7 |
| NA + RFM late (paper NA-RFM) | 84 / 74 | 83 / 83 | 100 / 100 | 95.7 | 94.1 |

Starting RFM late changes almost nothing. Adding NA (lambda 3) costs ship 9 points and truck 5-7, whichever window is used.

**Stage B: with NA on, RFM at w = 1 is too strong.** Halving the RFM strength fixes ship and truck and also lowers FID:

| setting | ship | truck | deer | mean acc | mean FID |
|---|---|---|---|---|---|
| RFM only, w 1 (paper; best RFM-only) | 93 / 82 | 90 / 70 | 96 / 97 | 93.1 | 83.1 |
| NA lambda 3 + RFM late, w 1 (paper NA-RFM) | 84 / 74 | 83 / 83 | 100 / 100 | 89.1 | 85.7 |
| NA lambda 3 + RFM late, w 0.5 (selected) | 99 / 65 | 98 / 54 | 99 / 90 | 98.7 | 69.7 |
| NA lambda 3 + RFM every step, w 0.5 | 99 / 65 | 98 / 54 | 99 / 90 | 98.6 | 69.7 |
| NA lambda 1 + RFM, w 0.5 | 96 / 60 | 97 / 51 | 89-91 / 86-88 | 94.3-94.7 | 66 |

Full grid: `results/tune.md`. Without NA, w 0.5 is too weak (deer 70-72%) and w 2 breaks every family (ship 19-41%,
FID 117-159), so the paper's w = 1 is the right RFM-only strength. Selected and frozen (`results/selection.json`,
sha256): combined = lambda 3, late window, w 0.5; RFM-only = the paper setting. Reading on dev: together the two
components over-steer at the paper's strength (lower accuracy on ship and truck and, for truck, higher FID); a lower
RFM strength removes this.

## Files

| File | What it does |
|---|---|
| `interplay.py` | Stages `decompose`, `tune`, `select`, `confirm`; `--smoke` for 2 classes, 16 images and 3 grid settings |
| `dsmlp_interplay.sh` | On a DSMLP GPU pod: `dev` (smoke test, then decompose tune select) or `test` (confirm) |
| `results/` | Stage A, B and C tables (`decompose.*`, `tune.*`, `confirm.*`) and the frozen selection (`selection.json`, `selection.sha256`) |

Needs the replication's setup on the pod (`python cifar10_replication/replicate.py setup offline`). Outputs go to
`outputs/q2_na_rfm_interplay/`; samples to `downloads/narfm/samples_interplay/` (neither in git).
