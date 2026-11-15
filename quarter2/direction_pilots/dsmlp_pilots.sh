#!/bin/bash
# Q2 round 11 pilots on a DSMLP GPU pod, one after another (a failing pilot is recorded; the next still runs).
# From the repo root:  bash quarter2/direction_pilots/dsmlp_pilots.sh
# Needs round 9's outputs/q2_sd_dof_direction/directions.pt and round 10's outputs/q2_sd_dof_masked_smoke/paired/.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt "peft==0.12.0" || exit 1
status=0
for st in response geometry residual; do
  stamp "pilot: $st"
  python quarter2/direction_pilots/pilots.py $st || { stamp "pilot $st FAILED"; status=1; }
done
stamp "done (status $status)"
