# Experiment notes

A dated log of what I ran and what I saw. Reproduction commands live in
[README.md](README.md); the to-do list lives in [handoff.md](handoff.md).

## Run log

One entry per GPU run. Machine and package versions are recorded automatically
in `figures/parts1_3/run_environment.json`; don't copy them here.

### Parts 1–3, first DSMLP run
- Date:
- Git commit:
- Status and elapsed time:
- Problems and fixes:

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

## Questions for the mentor
-
