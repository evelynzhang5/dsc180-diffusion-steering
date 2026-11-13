#!/bin/bash
# Q2 round 9 (SD 1.5 depth-of-field direction choice) on a DSMLP GPU pod. Run from the repo root:
#     bash quarter2/sd_dof_direction/dsmlp_sd_dof_direction.sh dev        # smoke test, then build tune select
#     bash quarter2/sd_dof_direction/dsmlp_sd_dof_direction.sh test       # smoke confirm, then confirm
#     bash quarter2/sd_dof_direction/dsmlp_sd_dof_direction.sh secondary  # text CFG 5 and 10
# Needs round 4's outputs/q2_sd_dof/{acts.pt,directions.pt} and the paper's code in downloads/narfm/na-rfm.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
PHASE=${1:?dev, test or secondary}
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q -r quarter2/sd_dof/requirements.txt datasets scikit-learn scipy || exit 1
python -m pip install --user -q --no-deps "opencv-python-headless==4.10.0.84" || exit 1
python -c "import cv2, diffusers, transformers; print(cv2.__version__, diffusers.__version__, transformers.__version__)" || exit 1
P=quarter2/sd_dof_direction/sd_dof_direction.py
if [ "$PHASE" = dev ]; then
  stamp "smoke test: build tune select confirm"
  rm -rf outputs/q2_sd_dof_direction_smoke
  python $P build tune select confirm --smoke || exit 1
  stamp "dev: build tune select"
  python $P build tune select || exit 1
elif [ "$PHASE" = test ]; then
  stamp "test: confirm"
  python $P confirm || exit 1
else
  stamp "secondary"
  python $P secondary --smoke || exit 1
  python $P secondary || exit 1
fi
stamp "done"
