#!/bin/bash
# Q2 round 39 step 1: boundary errors after request adjustment (BokehDiff), plus Bokehlicious on setting A.
# Run on a GPU pod from the repo root. Stage V checks in run39.py stop the run if the references fail.
# SMOKE=1 runs a reduced plan (1 scene, A/B at L1, 2 global + 1 edge adjustment; Bokehlicious 1 scene).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
W17=$HOME/bokehdiff-smoke
W=$HOME/bokehdiff-boundary
BK=$HOME/bokehlicious
BKV=$HOME/bkl-venv
BK_COMMIT=0d2c45824de076ffc57ce222b60578302fb80923
export R39_OUT=$REPO/outputs/q2_boundary
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
[ "$SMOKE" = 1 ] && R39_OUT=${R39_OUT}_smoke && export R39_OUT
rm -rf "$W/data" "$R39_OUT" && mkdir -p "$W/data/input" "$W/data/depth" "$W/data/mask" "$R39_OUT"
nvidia-smi -L
source "$W17/venv/bin/activate"
python -c "import diffusers, peft, torch; print('diffusers', diffusers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)"
cd "$W17/repo" && git checkout -q $COMMIT && git log -1 --format='bokehdiff commit %H'
python - "$REPO/quarter2/boundary" "$W/data" <<'EOF'
import sys, numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1]); import scenes18 as scenes
s = scenes.make("A")
Image.fromarray(s["image"]).save(f"{sys.argv[2]}/input/sceneA.png")
np.save(f"{sys.argv[2]}/depth/sceneA_pred.npy", s["disparity"].astype(np.float32))
Image.fromarray(((s["depth"] == s["focus"]["near"]) * 255).astype(np.uint8)).save(f"{sys.argv[2]}/mask/sceneA.png")
EOF
stamp "BokehDiff (checks first)"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$R39_OUT/gpu_mem.txt" &
GP=$!
python "$REPO/quarter2/boundary/run39.py" --test_data_dir "$W/data/input/*" --output_dir "$W/upstream_unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | tee "$R39_OUT/log_inference.txt" | grep -v Warning
kill $GP 2>/dev/null
stamp "peak GPU $(sort -n "$R39_OUT/gpu_mem.txt" | tail -1) MiB"
[ -f "$R39_OUT/conds.json" ] || { stamp "inference FAILED"; exit 1; }
python -c "import json,sys; sys.exit(0 if json.load(open('$R39_OUT/conds.json'))['checks_ok'] else 1)" || { stamp "checks FAILED"; exit 1; }
deactivate

stamp "Bokehlicious (setting A)"
if [ ! -d "$BK/.git" ]; then git clone -q https://github.com/TimSeizinger/Bokehlicious "$BK"; fi
cd "$BK" && git fetch -q origin && git checkout -q $BK_COMMIT && git log -1 --format='bokehlicious commit %H'
# GTX 1080 Ti (sm_61): current torch wheels dropped Pascal; torch 2.5.1+cu121 still ships sm_61 kernels.
if ! "$BKV/bin/python" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.5.1') else 1)" 2>/dev/null; then
  rm -rf "$BKV" && python3 -m venv "$BKV" && "$BKV/bin/pip" install -q --upgrade pip && \
  "$BKV/bin/pip" install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 && \
  "$BKV/bin/pip" install -q "timm>=1.0.20" opencv-python-headless "torchmetrics>=1.8.2" "tqdm==4.67.1" scipy scikit-image pillow \
      || { stamp "bokehlicious venv FAILED"; exit 1; }
fi
"$BKV/bin/python" -c "import torch; print('bkl torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list()); print((torch.ones(2, device='cuda') * 2).sum().item())"
"$BKV/bin/python" "$REPO/quarter2/boundary/bkl39.py" "$R39_OUT" 2>&1 | tee "$R39_OUT/log_bkl.txt" | grep -v Warning
[ -f "$R39_OUT/bkl.json" ] || stamp "Bokehlicious FAILED (BokehDiff results still analysed)"

cd "$REPO"
source "$W17/venv/bin/activate"
[ "$SMOKE" = 1 ] && { python quarter2/boundary/analyze39.py "$R39_OUT" "$R39_OUT/res_smoke" 2>&1 | tail -30; stamp "smoke done"; exit 0; }
stamp "analysis (frozen decision)"
python quarter2/boundary/analyze39.py "$R39_OUT" quarter2/boundary/results 2>&1 | tee "$R39_OUT/log_analysis.txt"
stamp "done"
