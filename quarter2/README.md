# Quarter 2: what makes a steering intervention control the final image?

Each round of Quarter 2 experiments lives on its own branch off `main`, with its code, its write-up and copies of its
small result files (`quarter2/<folder>/results/`). `main` keeps Quarter 1, the CIFAR-10 NA-RFM replication
(`cifar10_replication/`) and this index. Quarter 1 results and the earlier MNIST attribute-preservation idea are in
[../notes.md](../notes.md).

```bash
git switch q2-sd-dof        # or any branch below; the round's README is quarter2/<folder>/README.md
```

| # | Round | Branch | Folder | Dates (2026) | Result | Status |
|---|---|---|---|---|---|---|
| 1 | Text-to-image repair pilots, SD 1.5 | `q2-sd-pilots` | `sd_pilots/` | Nov 5-7 | Bounded mid-sampling nudges (CFG and targeted pulses) on color/left-right, two-object and counting prompts showed no robust advantage over random (placebo) pushes | Archived |
| 2 | MNIST steering diagnostics | `q2-mnist-steering` | `mnist/` | Nov 7-8 | Readable is not the same as steerable; edits persist after release but late edits are weaker; a confounded reference set makes a confounded edit, and about 100 slant labels fix moderate bias; in a side-effect audit, paired edited outputs are better evidence of side effects than reference differences, but MNIST lacks the hard parts (open-ended side effects, contexts where effects do not transfer) | Done |
| 3 | Reference-set bias on SD 1.5: blur direction vs. people and framing | `q2-sd-bias` | `sd_bias/` | Nov 8-9 | The mid-block "blur" directions produced no measurable blur, but the portrait-biased one enlarged people in person prompts (framing +0.18 vs. balanced, 4 of 4 prompts) | Closed |
| 4 | SD 1.5 depth-of-field steering (NA-RFM paper's setup) and reference-set bias | `q2-sd-dof` | `sd_dof/` | Nov 9-10 | Background sharpness falls 35.6% (paper: 27.6%) but foreground falls as much; fitted zoom-only controls explain little of the sharpness reduction, which comes with changed scene content; the bias test was stopped before generation because the dataset lacks the needed reference groups (15 confirmed shallow-focus wide scenes, 32 needed) | Closed |
| 5 | Can a small output audit predict steering side effects? CIFAR-10, RFM class steering | `q2-cifar-audit` | `cifar_audit/` | Nov 11-12 | A 256-image audit predicted which steered-vs-native set differences recur on new seeds (19 of 21 found, 4 false alarms), but on the same seed most of them vanish or reverse: they are what the seed would have produced anyway. Lesson: audit conclusions depend on the comparison population. Fine changes (antlers, framing) were not labelled reliably at 32x32 | Closed |
| 6 | Can people label what SD steering changed? (feasibility on round 4's pairs) | `q2-sd-labeling` | `sd_labeling/` | Nov 12 | No question met the agreement bar (people: framing kappa 0.31, subject visible 0.12); all raters saw steered subjects as softer (the target) and more often larger in frame | Closed |
| 7 | When do noise alignment and RFM help or interfere? (CIFAR-10, NA-RFM) | `q2-na-rfm-interplay` | `na_rfm_interplay/` | Nov 12- | | In progress |

Where things stand (Nov 12, 2026): after rounds 5-6 the project returns to improving steering itself, starting
from the replication's concrete failure (NA-RFM below RFM alone on ship and truck). Earlier (Nov 10): reference-set bias does carry over into unintended changes (rounds 2 and 3), but
on Stable Diffusion we have not found a steering target that works cleanly enough for a matched-target comparison
(rounds 3 and 4).
