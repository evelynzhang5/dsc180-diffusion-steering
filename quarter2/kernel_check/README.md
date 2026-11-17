# Q2 round 27: is the patch-to-patch blur pattern real, or a measurement effect?

**Question.** Rounds 24–26 found a stable per-patch pattern in delivered blur:
- astronaut and chelsea read about 15 % low;
- coins and night lights read about 10–20 % high.

Steering PISA's attention did not change that pattern. Round 24 already warned that the disk-fit estimator reads known Gaussian blur 3–11 % differently across these same patches. If the pattern is a measurement effect, downstream steering has no target there. If it is real, it does.

**Status.** Plan and rules fixed before running. This round makes no new BokehDiff generations: it re-measures round 25's saved baseline outputs (fp32 PISA + original gate + radius curve; 12 held-out patches × 3 positions × 7 targets, 3–15.5 px).

## Method (`check27.py`)

**A second, shape-agnostic estimator.** Per cell, the blur kernel is a non-negative radial profile, rings of 1 px width up to 34 px, fitted by bounded least squares on luminance with a free offset. Its size is the second moment r2, which is valid for any kernel shape: a disk of radius R has r2 = R/√2, a Gaussian of σ has r2 = σ√2.

| step | what |
|---|---|
| V, validation | known disks (4, 8, 12 px) and Gaussians (σ 2, 4, 6) applied to these exact patches, degraded as before. Pass: median r2 error ≤ 10 % per kernel **and** across-patch CV ≤ 3 % |
| A, real outputs | per-patch pattern (each patch's measure ÷ the median of all patches at that target) for the disk fit and for r2; across-patch CV of each |
| B, measurement only | the median real kernel profile (targets 5, 8, 11.6 px) is applied identically to every patch and read with the disk-fit estimator. Does measurement alone reproduce the real per-patch pattern? |

## Rules (fixed now)

| outcome | rule | meaning for steering |
|---|---|---|
| V | validation fails | no conclusion |
| K1 | median CV with r2 ≤ ½ × with the disk fit, **and** Spearman(real disk pattern, identical-kernel disk pattern) ≥ 0.7 | the pattern is mostly measurement; downstream steering has no target in it |
| K2 | median CV with r2 ≥ 0.8 × with the disk fit, **and** the identical-kernel CV ≤ ½ × the real disk-fit CV | the pattern is a real model effect; downstream steering has a target |
| K3 | otherwise | mixed |

**Also reported:** the radial kernel profiles of the high-contrast patches (coins, night lights) vs typical ones (gravel, chelsea), i.e. whether BokehDiff renders a disk-, Gaussian- or ring-like (bokeh) kernel there.

## Results

Pending.
