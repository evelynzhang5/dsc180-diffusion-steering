#!/bin/bash
# Q2 round 10 smoke tests on a DSMLP GPU pod, run one after another. From the repo root:
#     bash quarter2/sd_dof_masked/dsmlp_sd_dof_masked.sh smoke
#   1. masked steering and measure checks   (sd_dof_masked.py smoke)
#   2. paired data: rendered defocus pairs, paired activations, paired direction   (smoke_paired.py paired)
#   3. LoRA on the same pairs: 50 steps, save, reload, two slider values           (smoke_paired.py lora)
# Each test writes its own results; a failing test is recorded and the next one still runs.
# Needs round 9's outputs/q2_sd_dof_direction/directions.pt.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt "peft==0.12.0" || exit 1
python -c "import diffusers, transformers, peft; print(diffusers.__version__, transformers.__version__, peft.__version__)" || exit 1
status=0
stamp "smoke 1: masked steering"
python quarter2/sd_dof_masked/sd_dof_masked.py smoke || { stamp "smoke 1 FAILED"; status=1; }
if [ -f quarter2/sd_dof_masked/smoke_paired.py ]; then
  stamp "smoke 2: paired data"
  python quarter2/sd_dof_masked/smoke_paired.py paired || { stamp "smoke 2 FAILED"; status=1; }
  stamp "smoke 3: LoRA"
  python quarter2/sd_dof_masked/smoke_paired.py lora || { stamp "smoke 3 FAILED"; status=1; }
fi
stamp "done (status $status)"
