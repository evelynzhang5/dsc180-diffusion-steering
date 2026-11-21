#!/bin/bash
# Q2 round 37: frozen downstream GroupNorm statistics with the optimised up_blocks.1 edit. Run on a GPU pod from the repo root.
# The checks at the top of run37.py are the smoke stage: it aborts the run if reproduction or the edit hook fails.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
W17=$HOME/bokehdiff-smoke
W=$HOME/bokehdiff-freeze
export R37_OUT=$REPO/outputs/q2_norm_freeze_v2
export R25_OUT=$REPO/outputs/q2_pisa_gate
export R31_OUT=$REPO/outputs/q2_lora_scale
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
[ "$SMOKE" = 1 ] && R37_OUT=${R37_OUT}_smoke && export R37_OUT
rm -rf "$W/data" "$R37_OUT" && mkdir -p "$W/data/input" "$W/data/depth" "$W/data/mask" "$R37_OUT"
nvidia-smi -L
source "$W17/venv/bin/activate"
python -c "import diffusers, peft; print('diffusers', diffusers.__version__, 'peft', peft.__version__)"
cd "$W17/repo" && git checkout -q $COMMIT && git log -1 --format='bokehdiff commit %H'
python - "$REPO/quarter2/norm_freeze" "$W/data" <<'EOF'
import sys, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); import scenes18 as scenes
s = scenes.make("A")
Image.fromarray(s["image"]).save(f"{sys.argv[2]}/input/sceneA.png")
np.save(f"{sys.argv[2]}/depth/sceneA_pred.npy", s["disparity"].astype(np.float32))
Image.fromarray(((s["depth"] == s["focus"]["near"]) * 255).astype(np.uint8)).save(f"{sys.argv[2]}/mask/sceneA.png")
EOF
stamp "inference (gate first)"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$R37_OUT/gpu_mem.txt" &
GP=$!
python "$REPO/quarter2/norm_freeze/run37.py" --test_data_dir "$W/data/input/*" --output_dir "$W/upstream_unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | tee "$R37_OUT/log_inference.txt" | grep -v Warning
kill $GP 2>/dev/null
stamp "peak GPU $(sort -n "$R37_OUT/gpu_mem.txt" | tail -1) MiB"
[ -f "$R37_OUT/conds.json" ] || { stamp "inference FAILED"; exit 1; }
cd "$REPO"
[ "$SMOKE" = 1 ] && { python quarter2/norm_freeze/analyze37.py "$R37_OUT" "$R37_OUT/res_smoke" 2>&1 | tail -15; stamp "smoke done"; exit 0; }
stamp "analysis (frozen decision)"
python quarter2/norm_freeze/analyze37.py "$R37_OUT" quarter2/norm_freeze/results 2>&1 | tee "$R37_OUT/log_analysis.txt"
stamp "done"
