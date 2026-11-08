#!/bin/bash
# Quarter 2 smoke test in one DSMLP GPU pod. Run from the repository root:
#     bash quarter2/sd_pilots/dsmlp_smoke.sh
# 1. 192 baselines (24 prompts x 8 seeds), scored; the success mix is printed (warning only).
# 2. Dev sweep: 4 separate prompts x 4 seeds, every pulse strength in DEV_ACTIONS (plus none:0, which measures
#    numerical noise in resuming) at steps 6, 15, 24.
# 3. Predeclared pick: for each action type, the strength with the highest mean change in all-requirements
#    success on dev (ties keep the weaker strength).
# 4. Main branches with the picked strengths, scoring, and report.md.
# Outputs go to outputs/q2_smoke/{main,dev}; every stage resumes if the pod restarts.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
OUT=$PWD/outputs/q2_smoke
DEV_ACTIONS=${DEV_ACTIONS:-none:0,global:10,global:14,global:20,targeted:5,targeted:15,targeted:40}

nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_pilots/requirements.txt || exit 1
cd quarter2/sd_pilots || exit 1

stamp "baselines"
python sd_branch.py baseline --prompts prompts.json --seeds 0-7 --out "$OUT/main" || exit 1
python evaluate.py --run "$OUT/main" --prompts prompts.json || exit 1
python analyze.py mix --run "$OUT/main" --prompts prompts.json | tee "$OUT/main/mix.md"

stamp "dev sweep: $DEV_ACTIONS"
python sd_branch.py baseline --prompts dev_prompts.json --seeds 100-103 --out "$OUT/dev" || exit 1
python sd_branch.py branch --prompts dev_prompts.json --run "$OUT/dev" --actions "$DEV_ACTIONS" || exit 1
python evaluate.py --run "$OUT/dev" --prompts dev_prompts.json || exit 1
ACTIONS=$(python analyze.py pick --run "$OUT/dev" --prompts dev_prompts.json 2> "$OUT/dev/pick.txt") || exit 1
cat "$OUT/dev/pick.txt"
stamp "picked $ACTIONS"
echo "$ACTIONS" > "$OUT/main/actions.txt"

stamp "main branches"
python sd_branch.py branch --prompts prompts.json --run "$OUT/main" --actions "$ACTIONS" || exit 1
python evaluate.py --run "$OUT/main" --prompts prompts.json || exit 1
python analyze.py report --run "$OUT/main" --prompts prompts.json --dev "$OUT/dev" \
    --dev-prompts dev_prompts.json > "$OUT/main/report.md"
cat "$OUT/main/report.md"
stamp "done"
