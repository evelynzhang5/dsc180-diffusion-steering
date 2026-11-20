#!/bin/bash
# Presentation scene: 8 real background patches (corners filled), 3 generations. Run on a GPU pod from the repo root.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
W17=$HOME/bokehdiff-smoke
W=$HOME/bokehdiff-scene8
export R8_OUT=$REPO/outputs/q2_scene8
export R25_OUT=$REPO/outputs/q2_pisa_gate
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
rm -rf "$W/data" && mkdir -p "$W/data/input" "$W/data/depth" "$W/data/mask" "$R8_OUT"
nvidia-smi -L
source "$W17/venv/bin/activate"
cd "$W17/repo" && git checkout -q $COMMIT && git log -1 --format='bokehdiff commit %H'
python - "$REPO/quarter2/kernel_locate" "$W/data" <<'EOF'
import sys, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); import scenes18 as scenes
s = scenes.make("A")
Image.fromarray(s["image"]).save(f"{sys.argv[2]}/input/sceneA.png")
np.save(f"{sys.argv[2]}/depth/sceneA_pred.npy", s["disparity"].astype(np.float32))
Image.fromarray(((s["depth"] == s["focus"]["near"]) * 255).astype(np.uint8)).save(f"{sys.argv[2]}/mask/sceneA.png")
EOF
stamp "inference"
python "$REPO/quarter2/kernel_locate/render8.py" --test_data_dir "$W/data/input/*" --output_dir "$W/upstream_unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | tee "$R8_OUT/log_inference.txt" | grep -v Warning
[ -f "$R8_OUT/gate.json" ] || { stamp "FAILED"; exit 1; }
stamp "done"
