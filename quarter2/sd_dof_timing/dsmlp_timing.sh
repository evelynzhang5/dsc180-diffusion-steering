#!/bin/bash
# Q2 round 13 timing test on a DSMLP GPU pod.
# From the repo root:  bash quarter2/sd_dof_timing/dsmlp_timing.sh
# Needs round 9's directions.pt and round 10's paired/ (imported by round 11's code).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt || exit 1
stamp "round 13: timing"
python quarter2/sd_dof_timing/timing.py || { stamp "timing FAILED"; exit 1; }
stamp "done"
