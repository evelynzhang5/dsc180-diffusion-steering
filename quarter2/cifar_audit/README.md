# Q2 round 5: can a small output audit predict a steering direction's side effects? (CIFAR-10)

Branch `q2-cifar-audit`. Started Nov 11, 2026. The index of all Q2 rounds is in [../README.md](../README.md).

**Question.** Reference examples show what distinguishes the labeled groups; generated outputs show what a learned
steering direction actually changes. Can a small audit of paired steered/unsteered outputs predict the direction's
unintended effects on new generations, better than inspecting the reference examples?

**Why CIFAR-10.** Class steering works there (`cifar10_replication/`: RFM-only reaches 96% accuracy), so effects can be
studied at successful target control. The MNIST audit (`q2-mnist-steering`) favoured paired outputs, but it used a
fixed list of 7 attributes and side effects transferred across digit classes 98% of the time, so it could not test
the hard part.

## Design (fixed on Nov 11, 2026, before any test image was generated)

- **Steering:** the paper's RFM-only setting, one learned direction per class at block 9 (w = 1, s = 2, every step),
  improved-DDPM CIFAR-10 model, 100 DDIM steps, the replication's code and RFM directions.
- **Classes:** airplane, cat, deer, ship. These behaved differently in the replication. Cat has the weakest RFM
  classifier but 100% accuracy; ship is where NA-RFM lost accuracy; airplane and deer have strongly typical
  backgrounds.
- **What counts as a side effect.** Making class c requires changes: an airplane needs wings, and blue sky is common
  behind airplanes, so neither is an error. Steered images are therefore compared with the **model's own images of
  class c**: unsteered (unguided) samples that the paper's ConvNeXt evaluator assigns to class c ("native"). Only
  steered images the evaluator assigns to class c are used. The effect on a meter is
  (steered mean - native mean) / SD of the meter over the real class-c reference images. This removes the change
  needed for the class and the model's general differences from real photos.
- **Meters** (`meter_checks.md` tests each on real images with a known edit): brightness, contrast, saturation,
  colourfulness, blueness, greenness, redness, background clutter (border-ring SD), figure-ground contrast,
  sharpness; and three set meters: feature diversity (ConvNeXt), background-colour diversity, pixel diversity.
- **Random-direction control:** the same hook with a random direction (the hook rescales every direction to
  strength x |h|, so the size is identical), compared with unguided images on the same seeds. A confirmed side effect
  is called generic if the random direction moves the meter the same way by at least half as much.
- **Data split.** Dev: seeds 0-255 (the replication's samples, already seen in grids), plus an unguided dev pool on
  seeds 10000-12047 for native images. Test: new seeds 3000-3511 (steered, unguided, random) and an unguided test pool
  on seeds 20000-25119.

**Prediction sources**, all computed on dev data only and written to `predictions.json` with its sha256 before the
test images are generated (`cifar_audit.py` refuses to generate them otherwise):

| source | information used | prediction |
|---|---|---|
| REF | reference images only | steering makes images like the class references: no side effects |
| OUT_32, OUT_256 | 32 or 256 dev steered images vs. the model's native class images | their measured effects |
| BOTH_32, BOTH_256 | the same steered images vs. the real class-c reference images | their measured effects |
| TRANSFER | OUT_256 of the other three classes, averaged | effects shared across classes |

A source flags a meter if |effect| > 0.25 SD (set meters: |ratio - 1| > 0.10) and its 95% bootstrap interval
excludes 0. On the test seeds an effect is **confirmed** by the same rule.

**Decision (fixed in advance).**
- No confirmed effects beyond the generic ones: nothing to predict on CIFAR; stop.
- OUT_32 finds the confirmed, non-generic effects with few false alarms (most hits, at most about 1 false alarm in
  5 flags): a small paired audit already solves this with fixed meters. That becomes the baseline any method must beat,
  and a harder setting is needed (effects outside a fixed meter list, or contexts where they do not transfer).
- OUT_32 misses many effects or raises many false alarms, while OUT_256 does well: the budget matters, and a cheaper
  way to find effects could be worth studying.
- A blinded visual sample (`blind_<class>.png`, 12 steered and 12 native test images, key in `blind_key.json`)
  checks for effects outside the meter list.

## Status

- Nov 11, 2026: code written; dev pod run (smoke test, meter checks, dev generation); predictions frozen
  (sha256 `3940dc50...`). Test pod run on seeds 3000-3511; the hash still matched at evaluation. Full results in
  [results/report.md](results/report.md).
- Changes after freezing (none touch the predictions): the guard against early test generation was moved so the
  smoke run could reach `predict`; the set-meter check (repeated images) could not detect anything and was replaced
  by one-class vs. mixed-class images before the test stage.
- Nov 12, 2026: **closed.** The same-seed check showed that the pre-registered effects are mostly differences between
  image sets, not changes caused by steering, and the labelling check showed that fine changes (antlers, framing)
  are not labelled reliably at 32x32. Discovery work moves to a labelling feasibility test on saved SD pairs. The
  lasting result is the evaluation lesson below.

**Lesson.** What an audit concludes depends on the comparison population. Steered images vs. the model's own
class images gave 21 repeatable "side effects" that a small audit predicted well; on the same seeds, most of them
vanish or reverse. Three quantities need to be kept apart: target success (did the image become the class?), the
same-seed change (what did switching on steering change?) and the difference from typical class images (how does
the steered distribution differ from the model's usual one?). The same-seed change is the total effect of the
intervention; it does not by itself separate changes needed for the target from collateral ones.

## Result (Nov 11, 2026)

Terminology correction (Nov 12): "side effects" in this section and in `results/report.md` means differences
between steered and native image sets that recur on new seeds, not changes caused by steering (see the same-seed
check below).

21 confirmed side effects on the test seeds, 20 not generic (random directions moved no meter by more than 0.15 SD).
Examples: steered ships are less blue and redder (-0.68 / +0.61 SD), steered cats, deer and ships are sharper
(+0.3 to +0.8), steered deer have bluer backgrounds and stronger figure-ground contrast.

| source | flagged | hits | false alarms | missed | effect-size Spearman |
|---|---|---|---|---|---|
| OUT_32 | 17 | 13 | 4 | 8 | 0.90 |
| OUT_256 | 23 | 19 | 4 | 2 | 0.96 |
| BOTH_32 / BOTH_256 | 17 / 22 | 10 / 15 | 7 / 7 | 11 / 6 | 0.77 / 0.88 |
| TRANSFER | 11 | 10 | 1 | 11 | -0.15 |
| REF | 0 | 0 | 0 | 21 | - |

**Decision.** OUT_256 meets the "few false alarms" rule (19 of 21, 4 false alarms in 23 flags). OUT_32 is just short
of it (13 of 21; 4 of 17 flags false, about 1 in 4): its estimates have the right sign and size (Spearman 0.90), but
its intervals are too wide to flag 8 real effects. All 4 of its false alarms have the right sign and an overestimated
size; 2 are on background diversity, which the meter check could not validate. So with fixed meters an output audit of
32-256 steered images, compared with the model's own class images, predicts which set differences recur on new seeds, and it is the baseline for that question. Comparing with the native class images
beats comparing with real references (BOTH: 7 false alarms). Unlike MNIST, side effects do **not** transfer across
classes here (TRANSFER Spearman -0.15), so each direction needs its own audit.

**Sensitivity analysis (added after the test; the pre-registered result above is unchanged).** The same flags and
confirmations, dropping the meters whose checks failed or were inconclusive:

| meters | confirmed | OUT_32 | OUT_256 | BOTH_256 | TRANSFER |
|---|---|---|---|---|---|
| all 13 (pre-registered) | 21 | 13 found, 4/17 false | 19 found, 4/23 false | 15 found, 7/22 false | 10 found, 1/11 false |
| without feature diversity | 17 | 10 found, 4/14 false | 15 found, 4/19 false | 12 found, 7/19 false | 6 found, 1/7 false |
| 10 image meters only | 15 | 9 found, 2/11 false | 14 found, 3/17 false | 11 found, 3/14 false | 5 found, 1/6 false |

The ranking of sources does not change. On the 10 image meters OUT_32 has 2 false alarms in 11 flags but finds only
9 of 15.

**Scope.** This is prediction across new seeds of the same model, direction and strength, not across models,
strengths or datasets. REF here means one specific rule (reference images predict no side effects); finding nothing
says that rule is too weak, not that reference-based prediction fails in general (Ong et al., 2026, arXiv
2608.11227, forecast LLM steering side effects from unsteered representations). The steered vs. native comparison is
a distribution comparison, not a same-seed intervention effect: if the native "deer" pool contains ambiguous
animals, differences can come from the steered set being more recognisable. The same-seed comparison (steered vs.
unguided on seeds 3000-3511) can be computed from the saved test measures without new images.

**Same-seed check (added after the test; `same_seed.py`, results in
[results/same_seed.md](results/same_seed.md)).** The pre-registered effect splits exactly into two parts:
STEERING (steered minus unguided on the same seeds) and SEED (what those seeds make without steering minus the
model's own class images). For 12 of the 15 confirmed image-meter effects SEED is the larger part, and for 11
STEERING points the other way. Example: steering makes ships bluer (+0.52), but most steered ships start from seeds
that would have been other, less blue images (SEED -1.19), so they end up less blue than the model's own ships
(-0.68). Most colour "side effects" are therefore colour the steered image keeps from the image the seed would have
been: steering moves colour toward the class's typical colours, not all the way. That is not necessarily a defect:
if the request is only "make this a ship", keeping unspecified colour may be desirable; matching the model's typical
ship colours is a different goal. On the 45-82 seeds per class whose unsteered image
was already the class (selected subsets, so these say nothing about other seeds), the effects that hold are cat sharpness (+0.73), cat background clutter (+0.43), ship
sharpness (+0.50) and airplane contrast (-0.29); the others vanish or reverse. Steered-vs-native and same-seed
comparisons answer different questions and should both be reported; any discovery step should look at same-seed
pairs, or it will mostly rediscover what the seed would have produced anyway.

**Labelling check (step 2, dev seeds; `label_sheets.py`, [labeling/agreement.md](labeling/agreement.md)).** Three
raters (Evelyn; a second person; Claude, labelled before opening the key) answered fixed yes/no/unclear questions on
12 same-seed pairs per class whose unsteered image was already the class. Agreement between the two people (Cohen's
kappa): whole deer visible +0.47, water visible +0.39, whole hull visible +0.07, clearly a deer +0.03, clearly a
ship +0.02 (the second rater answered yes to 23 of 24), antlers visible -0.06. Absolute labels at 32x32 are not
reliable. But within pairs all three raters saw the same two changes: more antlers after steering (pairs gaining /
losing antlers: 3 / 0, 5 / 2, 4 / 0) and fewer ships with the whole hull in frame (0 / 3, 0 / 4, 0 / 3). Raters
differ in where they draw the line, not in which image of a pair has more. So a relative question on the pair
("which image shows antlers more clearly?") is likely the usable label format. With 12 pairs, and the two people
disagreeing image by image, these are leads only; per the Nov 12 decision they are not followed up on CIFAR
(an automatic framing measure would itself need validation at 32x32).

**Caveats.** Feature diversity is confirmed for all four classes, but the meter check shows it mostly measures
class spread (one class alone: 0.16-0.21 of mixed-class), so it likely reflects higher evaluator confidence rather
than less varied images. Background and pixel diversity barely react to the one-class check. In the blinded deer
sheet, steered images are mostly full-body antlered stags while native "deer" include ambiguous brown animals; that
change (a more prototypical class) lies outside the meter list and is not separable from the class change.

## Files and how to run

| File | What it does |
|---|---|
| `cifar_audit.py` | Stages `meters`, `gen_dev`, `predict`, `gen_test`, `evaluate`; `--smoke` for 2 classes and a few images |
| `dsmlp_cifar_audit.sh` | On a DSMLP GPU pod: `dev` (smoke test, then meters gen_dev predict) or `test` (meters gen_test evaluate) |
| `same_seed.py` | Same-seed check and decomposition from the saved test measures (no images made) |
| `label_sheets.py`, `labeling/` | Step 2: blinded same-seed labelling sheets (dev seeds, deer and ship, 24 images each), questions fixed in the script; `score` computes agreement |
| `results/` | Frozen predictions and hash, meter checks, test truth, `report.md`, blinded sheets and their key |

Needs the replication's setup on the pod (`python cifar10_replication/replicate.py setup offline generate`). Outputs go
to `outputs/q2_cifar_audit/`; samples to `downloads/narfm/samples_audit/` (neither in git).
