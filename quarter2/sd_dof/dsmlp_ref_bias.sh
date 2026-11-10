#!/bin/bash
# Reference-set bias pilot (ref_bias.py) on a DSMLP GPU pod. Run from the repo root:
#     bash quarter2/sd_dof/dsmlp_ref_bias.sh <stage> [<stage> ...]
# Needs outputs/q2_sd_dof/ from sd_dof.py (acts.pt, records, images). Smoke test first, then the stages.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -c "import numpy, scipy.ndimage, torch, transformers, diffusers, datasets; print('numpy', numpy.__version__)" || exit 1
P=quarter2/sd_dof/ref_bias.py
stamp "smoke test"
python $P "$@" --smoke || exit 1
stamp "full"
python $P "$@" || exit 1
stamp "finished"
