#!/bin/bash
# Quarter 2 pilot 2 in one DSMLP GPU pod. Run from the repository root:
#     bash quarter2/sd_pilots/dsmlp_pilot2.sh
# Pilot 1 (left-right relations, dsmlp_smoke.sh) was too hard for SD 1.5: 4.2% baseline success. Pilot 2 uses
# two failure types SD 1.5 gets right about a third of the time (GenEval): two objects in one image, and counting.
# 1. Baselines for both tasks (24 prompts x 8 seeds each), scored; success mix in mix.md.
# 2. For each task with 10-90% baseline success (decided before the run; otherwise it is skipped):
#    dev sweep (6 separate prompts x 4 seeds, every action in DEV_ACTIONS, including longer 8-step pulses),
#    predeclared pick (highest mean change in success per action type; ties keep the weaker/shorter one),
#    main branches with the picked actions, scoring, report.md.
# Outputs go to outputs/q2_pilot2/<task>/{main,dev}; every stage resumes if the pod restarts.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
OUT=$PWD/outputs/q2_pilot2
TASKS=${TASKS:-two_object counting}
DEV_ACTIONS=${DEV_ACTIONS:-none:0,global:10,global:14,global:20,global:14:8,global:20:8,targeted:15,targeted:40,targeted:80,targeted:40:8}

nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_pilots/requirements.txt || exit 1
cd quarter2/sd_pilots || exit 1

for t in $TASKS; do
  stamp "$t baselines"
  python sd_branch.py baseline --prompts ${t}_prompts.json --seeds 0-7 --out "$OUT/$t/main" || exit 1
  python evaluate.py --run "$OUT/$t/main" --prompts ${t}_prompts.json || exit 1
done

for t in $TASKS; do
  python analyze.py mix --strict --run "$OUT/$t/main" --prompts ${t}_prompts.json | tee "$OUT/$t/main/mix.md"
  if [ "${PIPESTATUS[0]}" = 3 ]; then stamp "$t skipped: baseline success outside 10-90%"; continue; fi

  stamp "$t dev sweep: $DEV_ACTIONS"
  python sd_branch.py baseline --prompts ${t}_dev.json --seeds 100-103 --out "$OUT/$t/dev" || exit 1
  python sd_branch.py branch --prompts ${t}_dev.json --run "$OUT/$t/dev" --actions "$DEV_ACTIONS" || exit 1
  python evaluate.py --run "$OUT/$t/dev" --prompts ${t}_dev.json || exit 1
  ACTIONS=$(python analyze.py pick --run "$OUT/$t/dev" --prompts ${t}_dev.json 2> "$OUT/$t/dev/pick.txt") || exit 1
  cat "$OUT/$t/dev/pick.txt"
  stamp "$t picked $ACTIONS"
  echo "$ACTIONS" > "$OUT/$t/main/actions.txt"

  stamp "$t main branches"
  python sd_branch.py branch --prompts ${t}_prompts.json --run "$OUT/$t/main" --actions "$ACTIONS" || exit 1
  python evaluate.py --run "$OUT/$t/main" --prompts ${t}_prompts.json || exit 1
  python analyze.py report --run "$OUT/$t/main" --prompts ${t}_prompts.json --dev "$OUT/$t/dev" \
      --dev-prompts ${t}_dev.json > "$OUT/$t/main/report.md"
  cat "$OUT/$t/main/report.md"
done
stamp "done"
