#!/bin/bash
# Q2 round 27: shape-agnostic blur-kernel check on round 25's saved outputs (no new BokehDiff generations).
# Run on a GPU pod from the repo root:  bash quarter2/kernel_check/dsmlp_kernel.sh
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 USE_TF=0
source $HOME/bokehdiff-smoke/venv/bin/activate
python -c "import numpy, scipy, skimage, matplotlib; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'skimage', skimage.__version__)" || { stamp "import FAILED"; exit 1; }
stamp "kernel check"
python quarter2/kernel_check/check27.py "$REPO/outputs/q2_pisa_gate" quarter2/kernel_check/results 2>&1 | tee outputs/q2_kernel_check.log
stamp "done"
