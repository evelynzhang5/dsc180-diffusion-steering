#!/bin/bash
# Zoom-only control (zoom_control.py) on the saved sd_dof.py images, on a DSMLP pod. Run from the repo root:
#     bash quarter2/sd_dof/dsmlp_zoom_control.sh
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python -m pip install --user -q --no-deps "opencv-python-headless==4.10.0.84" || exit 1
python -c "import numpy, scipy.ndimage, cv2; print(numpy.__version__, cv2.__version__)" || exit 1
P=quarter2/sd_dof/zoom_control.py
stamp "smoke test"
python $P --smoke || exit 1
stamp "full"
python $P || exit 1
stamp "finished"
