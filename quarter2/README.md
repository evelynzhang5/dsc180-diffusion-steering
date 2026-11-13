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
| 7 | When do noise alignment and RFM help or interfere? (CIFAR-10, NA-RFM) | `q2-na-rfm-interplay` | `na_rfm_interplay/` | Nov 12 | NA-RFM's ship and truck losses come from NA, not the late RFM window, and joint strength tuning removes them: with RFM at w 0.5 instead of 1, NA-RFM beats RFM alone on ship (98.0% vs. 94.5%) and truck (97.9% vs. 91.6%) on fresh seeds, mean FID 43.7 → 38.5 (1,024 per class). No new direction learned on NA trajectories was needed | Closed |
| 8 | With strength tuned, is there a better direction to push? (CIFAR-10, tuned NA-RFM baseline) | `q2-direction-choice` | `direction_choice/` | Nov 13 | With each direction's strength tuned (NA and window fixed), none of the 2nd RFM eigenvector, the class mean difference or its standardized version beat the paper's RFM direction (dev mean 98.7% / FID 69.7 on ship, truck, deer); none reached the accuracy floor, and all had higher FID. The paper's top-3 direction equals its top eigenvector (cosine >= 0.9995) | Closed |
| 9 | At matched background blur, does another direction blur the foreground less? (SD 1.5 depth of field) | `q2-sd-dof-direction` | `sd_dof_direction/` | Nov 13- | | Design draft |

Where things stand (Nov 13, 2026): after rounds 5-6 the project returned to improving steering itself. Round 7
fixed how hard to push (NA-RFM needs a lower RFM strength); round 8 found that none of a few simple alternative
directions beats the paper's RFM direction once strength is tuned, so learning combinations of them is not pursued. Round 9 (design only) asks the same question on SD 1.5 depth of field, where round 4 found the paper's direction softens the whole image: at matched background blur, is any direction more selective? Earlier (Nov 10): reference-set bias does carry over into unintended changes (rounds 2 and 3), but
on Stable Diffusion we have not found a steering target that works cleanly enough for a matched-target comparison
(rounds 3 and 4).
