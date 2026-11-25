#!/bin/bash
# Q2 round 46 (GPU pod, from the repo root): controlled transparency test, feature blending vs output blending.
# compose46 (CPU) builds the composites from round 45's fresh backgrounds; bd46 renders; score46 scores on the pod.
# SMOKE=1: first background, first composite (with all endpoint and zero-intervention checks).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
export R46_DATA=$REPO/outputs/q2_feature46/data R46_OUT=$REPO/outputs/q2_feature46/run
RES=$REPO/quarter2/feature46/results/run
[ "$SMOKE" = 1 ] && { export R46_OUT=${R46_OUT}_smoke; RES=$R46_OUT/res_smoke; }
rm -rf "$R46_OUT" && mkdir -p "$R46_OUT"
nvidia-smi -L
source "$W17/venv/bin/activate"
stamp "compose"
rm -rf "$R46_DATA"
python quarter2/feature46/compose46.py "$REPO/outputs/q2_confirm45/data" "$R46_DATA" 2>&1 | tail -3
[ -f "$R46_DATA/compose46.json" ] || { stamp "compose FAILED"; exit 1; }
stamp "render"
cd "$W17/repo" && git checkout -q $COMMIT
python "$REPO/quarter2/feature46/bd46.py" --test_data_dir "$R46_DATA/input/flat__tinted.png" --output_dir "$R46_OUT/unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | grep -v -i -e warn -e "it/s" | tee "$R46_OUT/log_bd46.txt" | tail -4
cd "$REPO"
[ -f "$R46_OUT/bd46.json" ] || { stamp "render FAILED"; exit 1; }
if [ "$SMOKE" = 1 ]; then deactivate; stamp "round 46 smoke render done"; exit 0; fi
stamp "score"
python quarter2/feature46/score46.py "$R46_DATA" "$R46_OUT" "$RES" 2>&1 | grep -v -i warn | tee "$R46_OUT/log_score.txt" | tail -40
deactivate
[ -f "$RES/score46.json" ] || { stamp "score FAILED"; exit 1; }
stamp "round 46 done"
