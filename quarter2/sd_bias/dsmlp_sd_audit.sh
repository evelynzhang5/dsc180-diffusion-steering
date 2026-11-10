#!/bin/bash
# Follow-up audit of the SD 1.5 blur-direction run, on a DSMLP GPU pod. Run from the repo root:
#     bash quarter2/sd_bias/dsmlp_sd_audit.sh
# Needs outputs/q2_sd_bias/eval from sd_bias.py. Stage 1: smoke test; stage 2: full run.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_bias/requirements.txt || exit 1
stamp "smoke test"
rm -rf outputs/q2_sd_audit_smoke
python quarter2/sd_bias/sd_audit.py --run outputs/q2_sd_bias --out outputs/q2_sd_audit_smoke --smoke || exit 1
stamp "full run"
python quarter2/sd_bias/sd_audit.py --run outputs/q2_sd_bias --out outputs/q2_sd_audit || exit 1
stamp "done"
