#!/bin/bash
# SD 1.5 depth-of-field steering (sd_dof.py) on a DSMLP GPU pod. Run from the repo root:
#     bash quarter2/sd_dof/dsmlp_sd_dof.sh
# Needs the paper's code in downloads/narfm/na-rfm (python cifar10_replication/replicate.py setup).
# Stage 1: smoke test; stage 2: reproduce the paper's setup; stage 3: the comparison.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt datasets scikit-learn scipy || exit 1
P=quarter2/sd_dof/sd_dof.py
stamp "smoke test"
rm -rf outputs/q2_sd_dof_smoke
for st in collect train "reproduce compare"; do python $P $st --smoke || exit 1; done
stamp "reproduce"
for st in collect train reproduce; do python $P $st || exit 1; done
stamp "compare"
python $P compare || exit 1
stamp "done"
