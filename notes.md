# Experiment notes

## First DSMLP GPU run
- Date:
- Git commit:
- GPU model:
- PyTorch / CUDA versions:
- Run command:
- Status and elapsed time:
- Problems and fixes:

Copy the machine details from figures/parts1_3/run_environment.json after a successful run.

## Before viewing the CFG grid
Prediction: what will change as guidance weight w increases from 1 to 4?

## Observations from parts 1–3
- Unconditional samples:
- Conditional samples:
- CFG grid (same seed across weights):
- Any artifacts or loss of diversity:

## Part 3 trajectory questions
Comparison: cat, three samples, seed 0, w = 1 versus w = 4.

1. At which displayed noise level does the class first appear in the denoiser estimate?
   Does it appear earlier for w = 4?
2. From which displayed noise level onward does the image barely change?
3. At low noise, how do the w = 4 estimates differ from the w = 1 estimates?

Record actual observations and the corresponding sigma values. Do not infer
an exact transition between displayed snapshots without inspecting more steps.

## Next experiments
- Parts 4–5: classifier accuracy, FID, and classwise FID; 1,000 generated images per setting.
- Part 6: training subsets of 100 and 1,000 images; snapshot evaluation and nearest neighbors.
- MNIST CFG experiment.
- Optional longer full-data CIFAR-10 training.

## Questions for the mentor
-
