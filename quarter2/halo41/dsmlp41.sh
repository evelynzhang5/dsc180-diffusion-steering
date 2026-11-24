#!/bin/bash
# Q2 round 41 step 1 (GPU pod, from the repo root): BokehDiff on the 8 round 40 mapping scenes under three depth
# conditions (original / two_layer / hard_edge) at the three round 40 frozen K. Scoring runs on CPU afterwards
# (measure41.py on the saved outputs). SMOKE=1: one scene, one K, three conditions.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
export R40_DATA=$REPO/outputs/q2_compare40/data R40_MAP=$REPO/outputs/q2_compare40/s3/mapping.json
export R41_OUT=$REPO/outputs/q2_halo41/step1
[ "$SMOKE" = 1 ] && export R41_OUT=${R41_OUT}_smoke
rm -rf "$R41_OUT" && mkdir -p "$R41_OUT"
[ -f "$R40_MAP" ] || { stamp "round 40 mapping.json missing"; exit 1; }
nvidia-smi -L
stamp "render"
source "$W17/venv/bin/activate"
cd "$W17/repo" && git checkout -q $COMMIT
python "$REPO/quarter2/halo41/bd41.py" --test_data_dir "$R40_DATA/input/hair_0.png" --output_dir "$R41_OUT/unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | grep -v -i -e warn -e "it/s" | tee "$R41_OUT/log_bd41.txt" | tail -4
cd "$REPO"; deactivate
[ -f "$R41_OUT/bd41.json" ] || { stamp "render FAILED"; exit 1; }
stamp "step 1 render done"
