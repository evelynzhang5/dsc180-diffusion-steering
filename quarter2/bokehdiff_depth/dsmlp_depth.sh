#!/bin/bash
# Q2 round 18: does BokehDiff blur each depth by the amount it was asked for? 3 rendered scenes with known depth.
# Run on a GPU pod from the repo root:  bash quarter2/bokehdiff_depth/dsmlp_depth.sh
# Reuses round 17's venv (image's torch 2.2.1+cu121 pinned) and BokehDiff at the same pinned commit.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
W17=$HOME/bokehdiff-smoke
W=$HOME/bokehdiff-depth
export R18_OUT=$REPO/outputs/q2_bokehdiff_depth
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
rm -rf "$W/data" && mkdir -p "$W/data/input" "$W/data/depth" "$W/data/mask" "$R18_OUT"
nvidia-smi -L

stamp "environment (round 17 venv)"
source "$W17/venv/bin/activate"
python -c "import numpy, torch, diffusers, scipy, skimage; print('numpy', numpy.__version__, 'torch', torch.__version__, 'diffusers', diffusers.__version__, 'scipy', scipy.__version__, 'skimage', skimage.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || { stamp "import check FAILED"; exit 1; }
cd "$W17/repo" && git checkout -q $COMMIT && git log -1 --format='bokehdiff commit %H' && git status --short | head

stamp "scene A in BokehDiff's EBB layout (for the convention check): image, true disparity 1/z, mask = nearest plane"
python - "$REPO/quarter2/bokehdiff_depth" "$W/data" <<'EOF'
import sys, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); import scenes
s = scenes.make("A")
Image.fromarray(s["image"]).save(f"{sys.argv[2]}/input/sceneA.png")
np.save(f"{sys.argv[2]}/depth/sceneA_pred.npy", s["disparity"].astype(np.float32))
Image.fromarray(((s["depth"] == s["focus"]["near"]) * 255).astype(np.uint8)).save(f"{sys.argv[2]}/mask/sceneA.png")
print("wrote scene A", s["image"].shape)
EOF

stamp "inference grid"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$R18_OUT/gpu_mem.txt" &
GP=$!
python "$REPO/quarter2/bokehdiff_depth/run_depth.py" --test_data_dir "$W/data/input/*" --output_dir "$W/upstream_unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | tee "$R18_OUT/log_inference.txt" | grep -v Warning
RC=$?
kill $GP 2>/dev/null
stamp "inference exit $RC, peak GPU $(sort -n "$R18_OUT/gpu_mem.txt" | tail -1) MiB"
[ -f "$R18_OUT/conds.json" ] || { stamp "inference FAILED"; exit 1; }

stamp "analysis (estimator validation, gate, measurements, decision)"
cd "$REPO"
python quarter2/bokehdiff_depth/analyze_depth.py "$R18_OUT" quarter2/bokehdiff_depth/results 2>&1 | tee "$R18_OUT/log_analysis.txt"
stamp "done"
