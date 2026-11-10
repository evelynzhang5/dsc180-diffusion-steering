#!/bin/bash
# Quarter 2 pilot 4: wrong-sign control for two objects, reusing pilot 2/3's saved states (seeds 0-15).
# Run from the repository root after pilot 3 has finished:
#     bash quarter2/sd_pilots/dsmlp_pilot4.sh
# targeted:-15 pushes toward the single-object prompts instead of away from them; global:1 lowers guidance from
# 7.5 by the same amount global:14 raises it. If these "repair" as often as the picked nudges, the direction of
# the push does not matter. analyze.py report then compares picked nudges vs. placebo, wrong sign and re-rolling
# the seed at about the same cost (the last needs no new images).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
RUN=$PWD/outputs/q2_pilot2/two_object/main
WRONG=targeted:-15,global:1

nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_pilots/requirements.txt || exit 1
cd quarter2/sd_pilots || exit 1
stamp "two_object wrong-sign branches, seeds 0-15"
python sd_branch.py branch --prompts two_object_prompts.json --run "$RUN" --actions $WRONG || exit 1
python evaluate.py --run "$RUN" --prompts two_object_prompts.json || exit 1
python analyze.py report --run "$RUN" --prompts two_object_prompts.json > "$RUN/report_pilot4.md"
cat "$RUN/report_pilot4.md"
stamp "done"
