#!/bin/bash
# Q2 round 44 (GPU pod, from the repo root): fp16 vs fp32 PISA at identical inputs and settings (precision diagnostic),
# then scoring on the same pod. SMOKE=1: 1 dev scene x K 4, 8 and 1 test cell (4 px).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
export R40_DATA=$REPO/outputs/q2_compare40/data R40_S3=$REPO/outputs/q2_compare40/s3 R42_RUN=$REPO/outputs/q2_control42/run
export R44_OUT=$REPO/outputs/q2_precision44/run
RES=$REPO/quarter2/precision44/results/run
[ "$SMOKE" = 1 ] && { export R44_OUT=${R44_OUT}_smoke; RES=$R44_OUT/res_smoke; }
rm -rf "$R44_OUT" && mkdir -p "$R44_OUT"
[ -f "$R42_RUN/bd42.json" ] || { stamp "round 42 outputs missing"; exit 1; }
nvidia-smi -L
stamp "render"
source "$W17/venv/bin/activate"
cd "$W17/repo" && git checkout -q $COMMIT
python "$REPO/quarter2/precision44/bd44.py" --test_data_dir "$R40_DATA/input/hair_0.png" --output_dir "$R44_OUT/unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | grep -v -i -e warn -e "it/s" | tee "$R44_OUT/log_bd44.txt" | tail -4
cd "$REPO"
[ -f "$R44_OUT/bd44.json" ] || { stamp "render FAILED"; exit 1; }
stamp "score"
python quarter2/precision44/score44.py "$R44_OUT" "$R42_RUN" "$RES" 2>&1 | grep -v -i warn | tee "$R44_OUT/log_score.txt" | tail -40
deactivate
[ -f "$RES/score44.json" ] || { stamp "score FAILED"; exit 1; }
stamp "round 44 done"
