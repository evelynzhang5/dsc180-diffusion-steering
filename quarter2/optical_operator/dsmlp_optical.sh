#!/bin/bash
# Q2 round 14 on a DSMLP GPU pod (a failing stage is recorded; the next still runs).
# From the repo root:  bash quarter2/optical_operator/dsmlp_optical.sh
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt || exit 1
status=0
for st in operator stereotype; do
  stamp "stage: $st"
  python quarter2/optical_operator/optical.py $st || { stamp "stage $st FAILED"; status=1; }
done
stamp "done (status $status)"
