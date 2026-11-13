# Q2 round 8: with strength tuned, is there a better direction to push? (CIFAR-10, NA-RFM)

Branch `q2-direction-choice`. Started Nov 13, 2026. The index of all Q2 rounds is in [../README.md](../README.md).

**Why.** Round 7 (`q2-na-rfm-interplay`) fixed how hard to push: NA-RFM with RFM at strength w = 0.5 instead of the
paper's 1 beat RFM alone on ship, truck and deer on fresh seeds (mean FID 43.7 → 38.5 at 1,024 images per class).
That tuned NA-RFM is now the baseline. This round asks whether a different block-9 direction improves the
accuracy-FID trade-off once strength is tuned fairly for every direction. Ship and truck are already near 98%
accuracy, so the main target is **lower FID at comparable accuracy**.

**Relation to the paper.** This round is not a reproduction. The paper's closest experiment is Appendix E.6, which
compares a mean-difference direction with RFM on one ImageNet class (unconditional ADM, class 222 kuvasz): RFM 30.5%
top-1 accuracy (label confidence 0.24) vs. mean difference 6.2% (0.08), which motivated using RFM. E.6 reports no
steering strength and does not state whether noise alignment was on. Round 7 showed that strength alone can reverse
a comparison, so this round checks whether E.6's conclusion holds on CIFAR-10 when every direction's strength is
tuned, and whether any direction beats the tuned baseline on FID. The w = 1 rows of the dev table give the
untuned comparison in E.6's style; the tuned picks give the fair one.

| | Paper, Appendix E.6 | This round |
|---|---|---|
| model, target | ImageNet (ADM), one class (kuvasz) | CIFAR-10, ship, truck and deer for tuning, all ten for confirmation |
| directions | mean difference vs. RFM | RFM (the paper's), 2nd RFM eigenvector, mean difference, standardized mean difference |
| noise alignment | not stated | on (lambda 3), as in tuned NA-RFM |
| strength | one value, not reported | each direction tuned over w 0.25-1.5 |
| metrics | top-1 accuracy, label confidence | accuracy and per-class FID |

The paper also never compares a single eigenvector with the top-k mix; Section 3.3 only sets k in {1, 3, 5}. The
build's finding that the top-3 mix is effectively the top eigenvector is reported with the per-class eigenvalue
ratio lambda1 / lambda2 (from the replication's RFM files), which explains the collapse.

**Question.** After tuning strength (with NA and the RFM window fixed), does choosing a different activation direction
beat the tuned paper direction? Candidate selection comes before learning combinations: if a short search already
handles it, or nothing beats the baseline, there is no case for a new method.

## Design (fixed on Nov 13, 2026, before any image of this round was generated)

Fixed throughout: the paper's code and model, NA lambda 3 for sigma >= 3.33, the direction added at block 9 for
sigma in [0.01, 11.69] with CFG-style scale s = 2, 100 DDIM steps; evaluation by the paper's ConvNeXt accuracy and
per-class FID against the class's 6,000 CIFAR-10 images. The sampler adds `w * ||activation|| / ||direction|| *
direction`, so only a direction's orientation matters and one w grid is fair to every candidate.

**Candidates** (all from block-9 features of the same 10,000 training images, 1,000 per class, at sigma 0.21, the
ones the paper's RFM was fit on):

| name | direction |
|---|---|
| `rfm_top3` | the paper's: top-3 RFM eigenvectors weighted by eigenvalue (baseline; the replication's files) |
| `rfm_eig1` | the top RFM eigenvector alone (built and reported, **not tuned**: see the amendment below) |
| `rfm_eig2` | the second RFM eigenvector alone |
| `meandiff` | class mean minus the mean of the other nine classes (raw features) |
| `meandiff_std` | the same difference divided by each feature's standard deviation (the RFM fits on standardized features) |

Single eigenvectors and mean differences are signed so the target class projects higher than the rest. Each
candidate's held-out probe AUC (projection, class vs. rest) is reported but **not** used to choose: selection uses
generated images.

| stage | what | classes | seeds |
|---|---|---|---|
| `build` | the candidate directions, their probe AUCs and cosines | all ten | training images |
| `tune` | 4 candidates x w {0.25, 0.5, 0.75, 1, 1.5} = 20 settings | ship, truck, deer | 0-255 |
| `select` | one w per candidate and one challenger by the rule below; frozen with a sha256 | | |
| `confirm` | tuned baseline vs. challenger, 1,024 images per class | **all ten** | 1000-2023 (new) |

The baseline at w 0.5 and 1 on seeds 0-255, and at w 0.5 on seeds 1000-2023 for ship, truck and deer, are copied from
round 7 (same code and flags).

**Selection rule.** Accuracy floor = the baseline's mean dev accuracy at w 0.5 (round 7's choice) minus 1 point.
For each candidate, the w with the lowest mean FID over ship, truck and deer among those at or above the floor (ties:
higher accuracy). The baseline is re-tuned by the same rule on the finer grid. Challenger = the non-baseline candidate
whose pick has the lowest mean FID. The full table is reported, not only the picks.

**Decision (confirm: all ten classes, 1,024 per class, fresh seeds).**

| result | interpretation | next |
|---|---|---|
| Challenger mean FID at least 2 lower, lower FID in at least 8 of 10 classes, mean accuracy at most 1 point lower | Direction choice matters beyond strength | Test whether learning a combination of directions beats the best single one |
| Otherwise | No candidate reliably improves the tuned baseline | Stop this direction rather than add complexity |

The dev table also answers the "scaling only" question: if a candidate beats the baseline at the paper's w = 1 but
not after each is tuned, the earlier gap was strength, not direction.

Caveats fixed in advance: dev FID at 256 images is biased upward and noisy, so it is only compared at equal counts;
one model, one evaluator; three development classes (the confirmation adds the other seven).

## Status

- Nov 13, 2026: design and code written; first pod: smoke test, then build, tune and select.
- Nov 13, 2026, amendment before any dev image: the build found that the paper's direction is the top eigenvector
  (cos(`rfm_top3`, `rfm_eig1`) between 0.9995 and 1.0000 in all ten classes; the top eigenvalue dominates the
  weights), so `rfm_eig1` is dropped from tuning as a duplicate of the baseline. Mean differences are far from it
  (cosine 0.33-0.57). Build table: `results/build.md` (once pulled).

## Files

| File | What it does |
|---|---|
| `directions.py` | Stages `build`, `tune`, `select`, `confirm`; `--smoke` for 2 classes, 16 images, 2 candidates x 2 strengths |
| `dsmlp_directions.sh` | On a DSMLP GPU pod: `dev` (smoke test, then build tune select) or `test` (confirm) |

Needs the replication's setup on the pod (`python cifar10_replication/replicate.py setup offline`) and round 7's
samples in `downloads/narfm/samples_interplay/` (if missing, they are regenerated). Outputs go to
`outputs/q2_direction_choice/`; directions to `downloads/narfm/directions/`; samples to
`downloads/narfm/samples_direction/` (none in git).
