#!/bin/bash
# Q2 round 12 diagnostics on a DSMLP GPU pod, one after another (a failing stage is recorded; the next still runs).
# From the repo root:  bash quarter2/diagnostics/dsmlp_diag.sh
# Needs round 9's directions.pt and round 10's outputs/q2_sd_dof_masked_smoke/ (images and paired/).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt || exit 1
status=0
for st in evalfix vaecheck geometry2 feasibility; do
  stamp "stage: $st"
  python quarter2/diagnostics/diag.py $st || { stamp "stage $st FAILED"; status=1; }
done
stamp "done (status $status)"
