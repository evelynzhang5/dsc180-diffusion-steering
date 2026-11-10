#!/bin/bash
# Quarter 2 pilot 3: placebo control and more seeds, reusing pilot 2's outputs. Run from the repository root:
#     bash quarter2/sd_pilots/dsmlp_pilot3.sh
# The question: is pilot 2's per-sample gain (hindsight best nudge vs. best fixed rule) targeted repair, or what
# any random re-roll of the image would give? A placebo pulse adds a random direction of fixed relative size
# (placebo:0.15 and placebo:0.5 = 15% and 50% of the normal noise prediction) for 3 steps.
# Real nudges stay as picked on dev in pilot 2 (main/actions.txt). Counting drops books and balloons, which the
# detector cannot count (decided after pilot 2's spot check; counting_countable.json).
# Order, most important first; every stage resumes if the pod restarts:
#   1. placebo branches on seeds 0-7 (real branches already exist)
#   2. baselines for seeds 8-15
#   3. real and placebo branches on seeds 8-15, scoring, report_pilot3.md
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
OUT=$PWD/outputs/q2_pilot2
PLACEBO=placebo:0.15,placebo:0.5

nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_pilots/requirements.txt || exit 1
cd quarter2/sd_pilots || exit 1
prompts() { [ "$1" = counting ] && echo counting_countable.json || echo ${1}_prompts.json; }

for t in two_object counting; do
  stamp "$t placebo, seeds 0-7"
  python sd_branch.py branch --prompts $(prompts $t) --run "$OUT/$t/main" --actions $PLACEBO || exit 1
done
for t in two_object counting; do
  stamp "$t baselines, seeds 8-15"
  python sd_branch.py baseline --prompts $(prompts $t) --seeds 8-15 --out "$OUT/$t/main" || exit 1
done
for t in two_object counting; do
  stamp "$t real and placebo branches, seeds 8-15"
  python sd_branch.py branch --prompts $(prompts $t) --run "$OUT/$t/main" \
      --actions "$(cat "$OUT/$t/main/actions.txt"),$PLACEBO" || exit 1
  python evaluate.py --run "$OUT/$t/main" --prompts $(prompts $t) || exit 1
  python analyze.py report --run "$OUT/$t/main" --prompts $(prompts $t) > "$OUT/$t/main/report_pilot3.md"
  cat "$OUT/$t/main/report_pilot3.md"
done
stamp "done"
