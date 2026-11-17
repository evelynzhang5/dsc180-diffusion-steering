#!/bin/bash
# Q2 round 25: PISA cell-transfer gate (sub-token blur) vs original gate; sweep, calibration, corrected reruns.
# Run on a GPU pod from the repo root:  bash quarter2/pisa_gate/dsmlp_gate.sh
# Reuses round 17's venv (image's torch 2.2.1+cu121 pinned) and BokehDiff at the same pinned commit.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
W17=$HOME/bokehdiff-smoke
W=$HOME/bokehdiff-gate
export R25_OUT=$REPO/outputs/q2_pisa_gate
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
rm -rf "$W/data" && mkdir -p "$W/data/input" "$W/data/depth" "$W/data/mask" "$R25_OUT"
nvidia-smi -L

stamp "environment (round 17 venv)"
source "$W17/venv/bin/activate"
python -c "import numpy, torch, diffusers, scipy, skimage; print('numpy', numpy.__version__, 'torch', torch.__version__, 'diffusers', diffusers.__version__, 'scipy', scipy.__version__, 'skimage', skimage.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || { stamp "import check FAILED"; exit 1; }
cd "$W17/repo" && git checkout -q $COMMIT && git log -1 --format='bokehdiff commit %H' && git status --short | head

stamp "round 18 scene A in BokehDiff's EBB layout (only used to start BokehDiff's main; never measured)"
python - "$REPO/quarter2/pisa_gate" "$W/data" <<'EOF'
import sys, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); import scenes18 as scenes
s = scenes.make("A")
Image.fromarray(s["image"]).save(f"{sys.argv[2]}/input/sceneA.png")
np.save(f"{sys.argv[2]}/depth/sceneA_pred.npy", s["disparity"].astype(np.float32))
Image.fromarray(((s["depth"] == s["focus"]["near"]) * 255).astype(np.uint8)).save(f"{sys.argv[2]}/mask/sceneA.png")
print("wrote scene A", s["image"].shape)
EOF

stamp "inference grid"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$R25_OUT/gpu_mem.txt" &
GP=$!
python "$REPO/quarter2/pisa_gate/run25.py" --test_data_dir "$W/data/input/*" --output_dir "$W/upstream_unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | tee "$R25_OUT/log_inference.txt" | grep -v Warning
RC=$?
kill $GP 2>/dev/null
stamp "inference exit $RC, peak GPU $(sort -n "$R25_OUT/gpu_mem.txt" | tail -1) MiB"
[ -f "$R25_OUT/conds.json" ] || { stamp "inference FAILED"; exit 1; }

stamp "analysis (estimator validation, gate, measurements, frozen decisions)"
cd "$REPO"
python quarter2/pisa_gate/analyze25.py "$R25_OUT" quarter2/pisa_gate/results 2>&1 | tee "$R25_OUT/log_analysis.txt"
stamp "done"
