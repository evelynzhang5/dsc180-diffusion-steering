#!/bin/bash
# Q2 round 43 (GPU pod, from the repo root): transparent subjects. Stage 1: generate 16 scenes, fetch 6 RealBokeh
# scenes, BokehDiff's own depth + mask preparation, frozen regions and K plan. Stage 2: BokehDiff under three depth
# conditions. Scoring is NOT run here: the see tiles are chosen by eye on the input sheets first (review43.json),
# then score43.py runs on CPU. SMOKE=1: 1 generated + 1 RealBokeh scene, 12 px only.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
export R43_DATA=$REPO/outputs/q2_transparent43/data R43_OUT=$REPO/outputs/q2_transparent43/run
[ "$SMOKE" = 1 ] && { export R43_DATA=$REPO/outputs/q2_transparent43_smoke/data R43_OUT=$REPO/outputs/q2_transparent43_smoke/run; }
rm -rf "$R43_DATA" "$R43_OUT" && mkdir -p "$R43_DATA" "$R43_OUT"
nvidia-smi -L
source "$W17/venv/bin/activate"
stamp "generate (RealVisXL V5.0)"
python "$REPO/quarter2/transparent43/gen43.py" "$R43_DATA" 2>&1 | grep -v -i warn || { stamp "generation FAILED"; exit 1; }
stamp "RealBokeh transparent scenes"
python "$REPO/quarter2/transparent43/rb43.py" "$R43_DATA" 2>&1 | grep -v -i warn || { stamp "RealBokeh FAILED"; exit 1; }
stamp "BokehDiff's own preparation (Depth Anything V2 Base + BiRefNet)"
cd "$W17/repo" && git checkout -q $COMMIT && python prepare_data.py --root "$R43_DATA" 2>&1 | grep -v -i -e warn -e "it/s" | tail -5
cd "$REPO"
n_in=$(ls "$R43_DATA/input" | wc -l); n_d=$(ls "$R43_DATA/depth" | wc -l); n_m=$(ls "$R43_DATA/mask" | wc -l)
stamp "inputs $n_in depth $n_d masks $n_m"
[ "$n_in" = "$n_d" ] && [ "$n_in" = "$n_m" ] || { stamp "preparation FAILED"; exit 1; }
stamp "regions + K plan"
python "$REPO/quarter2/transparent43/regions43.py" "$R43_DATA" 2>&1 | tail -30
[ -f "$R43_DATA/regions43.json" ] || { stamp "regions FAILED"; exit 1; }
stamp "render"
cd "$W17/repo" && git checkout -q $COMMIT
python "$REPO/quarter2/transparent43/bd43.py" --test_data_dir "$R43_DATA/input/glass_0.png" --output_dir "$R43_OUT/unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | grep -v -i -e warn -e "it/s" | tee "$R43_OUT/log_bd43.txt" | tail -4
cd "$REPO"
[ -f "$R43_OUT/bd43.json" ] || { stamp "render FAILED"; exit 1; }
deactivate
stamp "round 43 done"
