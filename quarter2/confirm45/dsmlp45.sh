#!/bin/bash
# Q2 round 45 (GPU pod, from the repo root): fresh-scene confirmation of the monotone controller (fp32 PISA).
# Generate 24 fresh scenes, BokehDiff's own preparation, round 40 regions + contact sheets, plan45 (eligibility, tile
# split, K plan), then render fixed / linear / monotone. Scoring is NOT run here: the input-only review (review45.json)
# is committed first, then score45.py runs on CPU. SMOKE=1: 3 scenes generated, 1 eligible scene x 4 px.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
export R45_DATA=$REPO/outputs/q2_confirm45/data R45_OUT=$REPO/outputs/q2_confirm45/run
[ "$SMOKE" = 1 ] && export R45_DATA=$REPO/outputs/q2_confirm45_smoke/data R45_OUT=$REPO/outputs/q2_confirm45_smoke/run
rm -rf "$R45_DATA" "$R45_OUT" && mkdir -p "$R45_DATA" "$R45_OUT"
nvidia-smi -L
source "$W17/venv/bin/activate"
stamp "generate (RealVisXL V5.0)"
python "$REPO/quarter2/confirm45/gen45.py" "$R45_DATA" 2>&1 | grep -v -i warn || { stamp "generation FAILED"; exit 1; }
stamp "BokehDiff's own preparation (Depth Anything V2 Base + BiRefNet)"
cd "$W17/repo" && git checkout -q $COMMIT && python prepare_data.py --root "$R45_DATA" 2>&1 | grep -v -i -e warn -e "it/s" | tail -5
cd "$REPO"
stamp "inputs $(ls $R45_DATA/input | wc -l) depth $(ls $R45_DATA/depth | wc -l) masks $(ls $R45_DATA/mask | wc -l)"
stamp "regions (round 40 regions40.py, unchanged) + plan"
python "$REPO/quarter2/compare40/regions40.py" "$R45_DATA" 2>&1 | tail -30
python "$REPO/quarter2/confirm45/plan45.py" "$R45_DATA" 2>&1 | tail -30
[ -f "$R45_DATA/plan45.json" ] || { stamp "plan FAILED"; exit 1; }
stamp "render (fp32 PISA)"
cd "$W17/repo" && git checkout -q $COMMIT
FIRST=$(ls "$R45_DATA/input" | head -1)
python "$REPO/quarter2/confirm45/bd45.py" --test_data_dir "$R45_DATA/input/$FIRST" --output_dir "$R45_OUT/unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | grep -v -i -e warn -e "it/s" | tee "$R45_OUT/log_bd45.txt" | tail -4
cd "$REPO"; deactivate
[ -f "$R45_OUT/bd45.json" ] || { stamp "render FAILED"; exit 1; }
stamp "round 45 render done"
