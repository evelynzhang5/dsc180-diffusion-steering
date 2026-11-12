# Q2 round 6: can people label what SD steering changed? (feasibility, development data), closed

**Closed on Nov 12, 2026.** No question met the agreement bar; framing was the only candidate. The project returns
to improving steering itself (round 7, branch `q2-na-rfm-interplay`); the labelling work stays as a supporting
diagnostic.

Started Nov 12, 2026, after round 5 (`q2-cifar-audit`) was closed: there, differences between image sets mostly
vanished on the same seed, and fine changes were not labelled reliably at 32x32. Branch `q2-sd-labeling`. The index of all Q2 rounds is in
[../README.md](../README.md).

**Question.** On same-seed steered / unsteered pairs from Stable Diffusion 1.5, can two people consistently answer
plainly visible questions about what steering changed? Only if they can is it worth freezing those questions and
collecting fresh images to confirm the changes.

**Images.** Round 4's test set (`q2-sd-dof`): RFM depth-of-field steering at the paper's setting vs. unsteered, same
COCO caption and seed. 12 pairs drawn at random (12 different captions, one random seed each), not picked for
striking differences. These images were already inspected in round 4, so this is development, not confirmation.

**Questions** (fixed in `pair_sheets.py` before labelling; steered side randomised, key kept separately):

| column | question | answers |
|---|---|---|
| `subject_A`, `subject_B` | Is the main subject (written under the caption) visible? | yes / no / unclear |
| `larger` | In which image does the main subject fill more of the frame? | A / B / same / unclear |
| `softer` | In which image is the main subject less sharp? | A / B / same / unclear |

Foreground softening is part of the target (the direction was learned for shallow depth of field), not by itself a
side effect. Subject loss and framing are the separate outcomes of interest.

**Decision.** If the two people agree well (kappa about 0.6 or more) on a question and the steered side differs
consistently, freeze that question for a confirmation run on fresh seeds and captions. If not, rewrite or drop it.

## How to label

1. Open `labeling/sheet_1.jpg` and `labeling/sheet_2.jpg` (6 pairs each).
2. Copy `labeling/labels_RATER.csv` to `labeling/labels_<name>.csv` and fill one row per pair.
3. Do not open `labeling/key_DO_NOT_OPEN_BEFORE_LABELLING.json` before finishing.
4. `python quarter2/sd_labeling/pair_sheets.py score` writes `labeling/agreement.md`.

Rebuilding the sheets needs round 4's images in `outputs/q2_sd_dof/images/{rfm,none}/` (copied from DSMLP) and the
captions in `outputs/q2_sd_dof/captions_test.json`.

## Result (Nov 12, 2026; `labeling/agreement.md`)

Three raters: Evelyn, a second person, and Claude (labelled after seeing the other two files, so not independent).
The decision uses the two people.

| question | Evelyn vs. rater 2 (kappa) | steered side, per rater (Evelyn / rater 2 / Claude) |
|---|---|---|
| Main subject visible? | +0.12 | subject missing in steered 0 / 2 / 2 of 12, in unsteered 0 / 1 / 1 |
| Subject fills more of the frame | +0.31 | steered larger 6 / 4 / 5, unsteered larger 3 / 1 / 0 |
| Subject less sharp | +0.00 | steered softer 8 / 12 / 9, unsteered softer 1 / 0 / 1 |

- **Softer:** every rater sees the steered subject as softer in most pairs (rater 2: all 12). Kappa is 0 only
  because rater 2 never varied, so it cannot measure agreement here. The target is visible; not a side effect.
- **Framing:** fair agreement and the same direction for all three (steered subject larger in 4-6 pairs, smaller
  in 0-3), matching the CIFAR ship lead. Below the 0.6 bar: rewrite before confirming, or measure it with an object
  detector at 512 px checked against these labels.
- **Subject loss:** too rare in 12 pairs (0-2 cases) to judge agreement; needs more pairs or a targeted measure.

No question meets the pre-set bar as worded. Framing is the one candidate worth sharpening.
