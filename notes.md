# Experiment notes

A dated log of what I ran and what I saw. Reproduction commands live in
[README.md](README.md); the to-do list lives in [handoff.md](handoff.md).

## Run log

One entry per GPU run. Machine and package versions are recorded automatically
in `figures/parts1_3/run_environment.json`; don't copy them here.

### Parts 1–3, first DSMLP run
- Date: November 1, 2026 (batch pod steer-run, GTX 1080 Ti)
- Git commit: e4aff85
- Status and elapsed time: parts 1–3 finished in 1 min; full lab (parts 1–6) in
  135 min; EDM full-data training for 197 min (1,205 kimg). All exit 0 except
  training, which stopped at its time limit as planned.
- Problems and fixes: the DSC180A workspace only schedules small GPUs (H100/A100
  requests are overridden), so the run used a 1080 Ti. GPU quota is 1, and a
  pending pod already holds it. Pods stop after 6 hours by default
  (K8S_TIMEOUT_SECONDS=43200 allows 12).

## Parts 1–3

### Before viewing the CFG grid
Prediction: what will change as guidance weight w increases from 1 to 4?

### Observations
- Unconditional samples:
- Conditional samples:
- CFG grid (same seed across weights):
- Any artifacts or loss of diversity:

### Part 3 trajectory questions
Comparison: cat, three samples, seed 0, w = 1 versus w = 4.

1. At which displayed noise level does the class first appear in the denoiser estimate?
   Does it appear earlier for w = 4?
2. From which displayed noise level onward does the image barely change?
3. At low noise, how do the w = 4 estimates differ from the w = 1 estimates?

Record actual observations and the corresponding sigma values. Do not infer
an exact transition between displayed snapshots without inspecting more steps.

## MNIST CFG network (Nov 1, 2026)

Trained the lab's U-Net (1.9M parameters) from scratch on 60,000 MNIST digits,
10% of labels replaced by 10, 30 epochs (7,050 steps), 22 min on Apple MPS.
Loss fell from 0.22 (epoch 1) to 0.060 (epoch 30), mostly flat after epoch 10.
Evaluated with 5,000 samples per w, 16 DDIM steps, seed 0. Classifier accuracy
on real test digits: 0.993. FID of 5,000 real training digits vs. test: 8.4.

| w | label accuracy | FID | FID_class | pixel spread |
| --- | --- | --- | --- | --- |
| 0 | 0.102 | 127.1 | (no digit requested) | 0.239 |
| 1 | 0.976 | 16.0 | 48.3 | 0.191 |
| 2 | 1.000 | 61.2 | 145.2 | 0.176 |
| 4 | 1.000 | 158.6 | 308.1 | 0.172 |

Observations:
- w = 0 matches chance (10%), as it should: it ignores the request.
- Accuracy saturates at w = 2, but FID gets much worse from w = 1 to 4: the
  classic CFG trade-off. Stronger guidance makes strokes thicker and more
  uniform (see figures/mnist/cfg_grid.png), which moves the features away from
  real digits even though every digit is read correctly.
- w = 1 FID (16.0) is close to the real-data floor (8.4).
- For adaptive CFG on MNIST, accuracy is already ~98% at w = 1, so the room to
  improve is mostly FID/diversity at full accuracy.
- Comparison: the course's pretrained generator.pt, same code at 1,000 samples,
  gave w = 0 FID 57 vs. this model's weaker unconditional mode (FID 127 at
  5,000; different sample counts, so only a rough comparison).

## Questions for the mentor
-
