#!/bin/bash
# Q2 round 31: per-block LoRA scaling vs request-only correction. Run on a GPU pod from the repo root.
# The gate at the top of run31.py is the smoke stage: it aborts the run if reproduction or the scaling hook fails.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
W17=$HOME/bokehdiff-smoke
W=$HOME/bokehdiff-lora
export R31_OUT=$REPO/outputs/q2_lora_scale
export R25_OUT=$REPO/outputs/q2_pisa_gate
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
rm -rf "$W/data" "$R31_OUT" && mkdir -p "$W/data/input" "$W/data/depth" "$W/data/mask" "$R31_OUT"
nvidia-smi -L
source "$W17/venv/bin/activate"
python -c "import diffusers, peft; print('diffusers', diffusers.__version__, 'peft', peft.__version__)"
cd "$W17/repo" && git checkout -q $COMMIT && git log -1 --format='bokehdiff commit %H'
python - "$REPO/quarter2/lora_scale" "$W/data" <<'EOF'
import sys, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); import scenes18 as scenes
s = scenes.make("A")
Image.fromarray(s["image"]).save(f"{sys.argv[2]}/input/sceneA.png")
np.save(f"{sys.argv[2]}/depth/sceneA_pred.npy", s["disparity"].astype(np.float32))
Image.fromarray(((s["depth"] == s["focus"]["near"]) * 255).astype(np.uint8)).save(f"{sys.argv[2]}/mask/sceneA.png")
EOF
stamp "inference (gate first)"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$R31_OUT/gpu_mem.txt" &
GP=$!
python "$REPO/quarter2/lora_scale/run31.py" --test_data_dir "$W/data/input/*" --output_dir "$W/upstream_unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | tee "$R31_OUT/log_inference.txt" | grep -v Warning
kill $GP 2>/dev/null
stamp "peak GPU $(sort -n "$R31_OUT/gpu_mem.txt" | tail -1) MiB"
[ -f "$R31_OUT/conds.json" ] || { stamp "inference FAILED"; exit 1; }
cd "$REPO"
stamp "analysis (frozen decision)"
python quarter2/lora_scale/analyze31.py "$R31_OUT" quarter2/lora_scale/results 2>&1 | tee "$R31_OUT/log_analysis.txt"
stamp "done"
