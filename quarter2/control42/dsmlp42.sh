#!/bin/bash
# Q2 round 42 (GPU pod, from the repo root): depth-aware control arms for BokehDiff on the 24 held-out round 40
# scenes, then scoring (CPU, same pod). SMOKE=1: one scene, target 8, grid [8, 20, 48].
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
export R40_DATA=$REPO/outputs/q2_compare40/data R40_S3=$REPO/outputs/q2_compare40/s3
export R42_OUT=$REPO/outputs/q2_control42/run
RES=$REPO/quarter2/control42/results/run
[ "$SMOKE" = 1 ] && { export R42_OUT=${R42_OUT}_smoke; RES=$R42_OUT/res_smoke; }
rm -rf "$R42_OUT" && mkdir -p "$R42_OUT"
[ -f "$R40_S3/bd_test.json" ] || { stamp "round 40 test outputs missing"; exit 1; }
nvidia-smi -L
stamp "render"
source "$W17/venv/bin/activate"
cd "$W17/repo" && git checkout -q $COMMIT
python "$REPO/quarter2/control42/bd42.py" --test_data_dir "$R40_DATA/input/hair_0.png" --output_dir "$R42_OUT/unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | grep -v -i -e warn -e "it/s" | tee "$R42_OUT/log_bd42.txt" | tail -4
cd "$REPO"
[ -f "$R42_OUT/bd42.json" ] || { stamp "render FAILED"; exit 1; }
stamp "score"
python quarter2/control42/score42.py "$R42_OUT" "$RES" 2>&1 | grep -v -i warn | tee "$R42_OUT/log_score.txt" | tail -30
deactivate
[ -f "$RES/score42.json" ] || { stamp "score FAILED"; exit 1; }
stamp "round 42 done"
