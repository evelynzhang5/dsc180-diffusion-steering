#!/bin/bash
# Q2 round 15 on a DSMLP GPU pod. From the repo root:  bash quarter2/real_bokeh/dsmlp_real_bokeh.sh
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt munch lpips gdown || exit 1
python -c "import cv2" 2>/dev/null || python -m pip install --user -q opencv-python-headless || exit 1
stamp "stage: setup"
python quarter2/real_bokeh/bench.py setup || { stamp "setup FAILED"; exit 1; }
status=0
for st in select subsetA subsetB; do
  stamp "stage: $st"
  python quarter2/real_bokeh/bench.py $st || { stamp "stage $st FAILED"; status=1; }
done
stamp "done (status $status)"
