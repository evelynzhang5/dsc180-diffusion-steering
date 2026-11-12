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

## Files

| File | What it does |
|---|---|
| `interplay.py` | Stages `decompose`, `tune`, `select`, `confirm`; `--smoke` for 2 classes, 16 images and 3 grid settings |
| `dsmlp_interplay.sh` | On a DSMLP GPU pod: `dev` (smoke test, then decompose tune select) or `test` (confirm) |

Needs the replication's setup on the pod (`python cifar10_replication/replicate.py setup offline`). Outputs go to
`outputs/q2_na_rfm_interplay/`; samples to `downloads/narfm/samples_interplay/` (neither in git).
