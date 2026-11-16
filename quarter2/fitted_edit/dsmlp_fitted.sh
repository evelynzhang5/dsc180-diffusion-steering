#!/bin/bash
# Q2 round 16 on a DSMLP GPU pod. From the repo root:
#   bash quarter2/fitted_edit/dsmlp_fitted.sh gen    # scenes, masks, targets, basis, probe (then check masks by eye)
#   bash quarter2/fitted_edit/dsmlp_fitted.sh fit    # baselines, fitted edits, control, report (resumes per scene)
#   bash quarter2/fitted_edit/dsmlp_fitted.sh gen fit  # both, in order
# Needs round 9's directions.pt (outputs/q2_sd_dof_direction) and round 10's outputs (imported by round 11's code).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt || exit 1
for st in "$@"; do
  stamp "round 16: $st"
  python quarter2/fitted_edit/fitted.py "$st" || { stamp "$st FAILED"; exit 1; }
done
stamp "done"
