#!/bin/bash
# SD 1.5 reference-bias check (blur direction vs. people and framing), on a DSMLP GPU pod. Run from the repo root:
#     bash quarter2/sd_bias/dsmlp_sd_bias.sh
# Stage 1: tiny smoke test; stage 2: full run.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_bias/requirements.txt || exit 1
stamp "smoke test"
rm -rf outputs/q2_sd_bias_smoke
python quarter2/sd_bias/sd_bias.py --out outputs/q2_sd_bias_smoke --smoke || exit 1
stamp "full run"
python quarter2/sd_bias/sd_bias.py --out outputs/q2_sd_bias || exit 1
stamp "done"
