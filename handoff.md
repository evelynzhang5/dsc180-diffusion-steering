# Diffusion Steering Capstone — Handoff

**Last updated:** November 1, 2026

Where the project stands and where it is going. The assignment itself is the box at the top of [cifar10_edm_lab.ipynb](cifar10_edm_lab.ipynb) (source of truth); how to reproduce results is in [README.md](README.md); observations are in [notes.md](notes.md); background theory is in [concepts.md](concepts.md).

## Checkpoint deliverables (due Friday, November 6)

Setup
- [x] Public GitHub repo with README
- [x] Notes file
- [ ] Parts 1–3 run on a DSMLP GPU

Generation and sampling (parts 1–5)
- [ ] Sample grids (parts 1–3)
- [ ] Trajectory figures for w = 1 and w = 4, with answers to part 3's questions in notes
- [ ] Part 5 table and plots: accuracy, FID, FID_class vs. w, at 1,000 images

Training
- [ ] Part 6: FID across snapshots for 100 and 1,000 training images
- [ ] Part 6: samples next to their nearest training images
- [ ] Part 6: CFG grids
- [ ] MNIST CFG network (separate MNIST guidance lab, not in this repo): label accuracy and FID per w
- [ ] Encouraged: full 50k-image EDM training run, loss curve and FID across snapshots

Report
- [ ] Title, abstract, introduction (problem, related work, data), repo link
- [ ] Preliminary results: this week's figures and tables, one sentence each, each reproducible from the README

## Capstone direction

**Primary: A — Adaptive CFG across the sampling trajectory.** Optional extension: C. Secondary analysis: B.

### A — Adaptive CFG (recommended)

**Question:** Can guidance that varies with noise level improve the trade-off between class accuracy, sample quality, and diversity compared with fixed CFG?

Make $w$ depend on sampling progress $\tau$ ($\tau = 0$ at high noise, $\tau = 1$ at the end):

- Constant baseline: $w(\tau) = 2$
- Early-heavy: $w(\tau) = 3 - 2\tau$
- Late-heavy: $w(\tau) = 1 + 2\tau$

Both variable schedules average to 2. They are experimental choices, not known optima.

**Why:** direct extension of part 5, no training, one clear sampler-level change, easy to explain.
**Risks:** may not beat fixed CFG; $\tau$ must follow the sampler's σ ordering; check related work before claiming novelty.

### B — CFG and memorization under limited data

**Question:** Does stronger CFG increase memorization or reduce diversity in the part 6 models trained on 100 and 1,000 images?
Measure accuracy, FID, nearest-neighbor distance, and diversity across snapshots and w. Reuses required work, but FID is noisy at small sample counts, and a close nearest neighbor is evidence to investigate, not proof.

### C — Example-based steering of a frozen model

**Question:** Can the frozen unconditional model be steered toward a class from a few examples, comparing RFM (feature-level) with noise alignment (sampler-level), against the conditional/CFG baseline?
Closest to the lecture's theme but highest risk: needs the NA-RFM paper and ideally its code. Don't invent algorithm steps the slides leave out.

### Experimental controls (for A, and any comparison)

Same checkpoints, sampler, noise schedule, and step count; matched seeds; same class balance and sample count; same classifier and FID implementation; record each schedule and the code version.

## Facts from the lab worth remembering

- **Notation:** the course uses $x_1$ = clean image, $x_0$ = noise, $x = x_1 + \sigma x_0$. EDM networks return $D(x;\sigma)$, the clean-image estimate.
- **CFG on CIFAR-10 uses two networks** (separate unconditional and conditional checkpoints), because EDM's released conditional model was trained without label dropout. Part 6 and MNIST use one network with 10% label dropout.
- **Part 5 requests are balanced:** image $i$ requests class $i \bmod 10$, with the same starting noise for each w.
- **FID_class:** FID of each class's generated images vs. that class's 5,000 training images, averaged over 10 classes.
- **The lecture's paper is "NA-RFM".** Its CIFAR-10 FIDs (~40) use a different per-class protocol and aren't comparable to ours. The lab says to switch the accuracy classifier from ResNet-56 to ConvNeXt-Tiny when comparing with it in week 8.

## Open questions

1. Full citation for the NA-RFM paper, and whether official code exists and works with EDM.
2. Details of the MNIST guidance lab (notebook, classifier, FID features).
3. Whether the capstone may use external datasets/models beyond the course materials.

## Notes for the next assistant

The user is learning diffusion models while building this capstone. Explain from first principles and tie equations to the project's code. Keep the project on steering; U-Net vs. DiT and VAE/Stable Diffusion are background, not directions.
