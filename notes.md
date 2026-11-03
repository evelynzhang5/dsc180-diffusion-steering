# Experiment notes

A dated log of what I ran and what I saw. Reproduction commands live in
[README.md](README.md); the to-do list lives in [plan.md](plan.md).

## Run log

One entry per GPU run. Machine and package versions are recorded automatically
in `figures/parts1_3/run_environment.json`


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

### Full-data EDM training, run 2
- Date: November 1–2, 2026 (batch pod edm-train2, GTX 1080 Ti, 12-hour limit)
- Git commit: 84addec (scripts/dsmlp_train_edm.sh 12 run2)
- Status and elapsed time: started from run 1's 1,004-kimg snapshot with
  `--transfer`; trained 10.6 h (76 ticks, 3,764 kimg, 10.2 s/kimg) and stopped
  at its deadline as planned, then snapshot FID for 8 snapshots in 21 min.
  FID 76.5 → 50.2; loss 0.195 → 0.182. Results committed in 0587a04.
- Problems and fixes: the laptop's SSH session lapsed at the 8-hour Duo limit
  overnight, so the status watcher stopped; the pod was unaffected. Run 2 used
  `--dump=10`, so training-state-003512.pt on DSMLP can be continued exactly
  with `--resume` (not planned).

## Parts 1–3

### Before viewing the CFG grid
Prediction: as w goes from 1 to 4, samples should look more clearly like cats,
with stronger contrast and saturation, and the columns should look more alike
(less variety).

### Observations
- Unconditional samples (figures/parts1_3/01): varied classes and backgrounds,
  mostly recognizable objects at 32x32.
- Conditional samples (03): each row reads as its class; within-class variety is
  still large (pose, color, background).
- CFG grid (04, same noise per column): w = 0 gives unrelated classes (bird,
  plane, car, deer, ship). From w = 1 every column becomes a cat, but layout and
  colors are inherited from the w = 0 image with the same noise (e.g. the ship's
  teal background stays in column 7). From w = 1 to 4 the pose barely changes;
  contrast, saturation and edge sharpness rise, and faces become more frontal.
- Artifacts: at w = 3–4 some samples look oversaturated or posterized (column 8
  turns bright orange; column 1 loses its cat shape at w = 4), consistent with
  FID rising from 31.3 (w = 1) to 39.0 (w = 4) in part 5.

### Part 3 trajectory questions
Comparison: cat, three samples, seed 0, w = 1 versus w = 4 (figures/parts1_3/05
and 06). Displayed noise levels: σ = 80, 28, 8.4, 3.3, 1.1, 0.3, 0.06, 0.002.

1. **When does the class appear?** For w = 1, the denoiser estimate D(x; σ) is a
   flat brown blur at σ = 80 and 28, a centered blob at 8.4, a coarse layout at
   3.3, and a recognizable animal only at σ = 1.1. Even then, two of the three
   are ambiguous (sample 2 looks like a guinea pig or rabbit). For w = 4 the cat
   appears earlier: pointed ears and a cat face are visible at σ = 3.3 in sample 1,
   and all three are clearly cats at σ = 1.1. At σ = 80 the w = 4 estimates
   already have a warmer, less uniform tint than the w = 1 ones.
2. **When does the image stop changing?** For both weights, the estimates at
   σ = 0.3, 0.06 and 0.002 look nearly identical. The coarse layout is fixed by
   about σ = 1.1; the steps below 0.3 only remove the last pixel noise from x.
3. **Low-noise differences:** at σ ≤ 0.3 the w = 4 estimates have higher
   contrast, more saturated orange, darker backgrounds and sharper edges. They
   are more prototypical cats (frontal faces, large ears), but sample 3 has
   garish colors (red nose, green eyes) and sample 2 has exaggerated stripes.
   The w = 1 estimates are more muted and natural but less clearly cats.

Takeaway for direction A: the class is decided between σ ≈ 8 and 1, and very
little changes below σ ≈ 0.3. Guidance probably matters most at middle noise
levels; strong guidance at low noise mostly adds saturation. This is the
motivation for testing noise-dependent w.

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
