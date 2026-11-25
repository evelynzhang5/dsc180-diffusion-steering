#!/bin/bash
# Q2 round 47 (GPU pod, from the repo root): layer separation -> BokehDiff on the background -> composite.
# Steps: RealBokeh f/22 backgrounds (download), T2 composites, train full + oracle separators, inference (T1, T2, T3,
# no-glass control), BokehDiff renders (pipeline check, SEP, T2 baselines, T3), scoring. SMOKE=1: 30 training steps,
# first background only, no scoring (gate: renders + pipeline check).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache OMP_NUM_THREADS=8 USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
Q=$REPO/quarter2/separate47
export R40_IN=$REPO/outputs/q2_compare40/data/input R45_IN=$REPO/outputs/q2_confirm45/data/input
export R46_DATA=$REPO/outputs/q2_feature46/data R46_RUN=$REPO/outputs/q2_feature46/run
export R43_DATA=$REPO/outputs/q2_transparent43/data R43_RUN=$REPO/outputs/q2_transparent43/run
O=$REPO/outputs/q2_separate47/run
RES=$Q/results/run
[ "$SMOKE" = 1 ] && { O=${O}_smoke; RES=$O/res_smoke; }
export R47_T2=$O/t2 R47_SEP=$O/sep R47_OUT=$O/render
rm -rf "$O" && mkdir -p "$O"
nvidia-smi -L
source "$W17/venv/bin/activate"
stamp "RealBokeh f/22 backgrounds"
export RB_IN=$(python -c "from huggingface_hub import snapshot_download as s; print(s('timseizinger/RealBokeh_3MP', repo_type='dataset', allow_patterns=['test/in/*']))")/test/in
echo "RB_IN $RB_IN $(ls $RB_IN | wc -l) files"
stamp "T2 composites"
python $Q/build47.py "$R46_DATA" "$R47_T2" 2>&1 | tail -2
stamp "train full"
python $Q/train47.py full "$O/full.pt" 2>&1 | grep -v -i warn || { stamp "train FAILED"; exit 1; }
stamp "train oracle"
python $Q/train47.py oracle "$O/oracle.pt" 2>&1 | grep -v -i warn || { stamp "train FAILED"; exit 1; }
stamp "inference"
python $Q/infer47.py "$O/full.pt" "$O/oracle.pt" "$R46_DATA" "$R47_T2" "$R43_DATA" "$R47_SEP" 2>&1 | grep -v -i warn | tail -12
[ -f "$R47_SEP/infer47.json" ] || { stamp "inference FAILED"; exit 1; }
stamp "render"
cd "$W17/repo" && git checkout -q $COMMIT
python $Q/bd47.py --test_data_dir "$R45_IN/f_fur_0.png" --output_dir "$O/unused" \
    --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 | grep -v -i -e warn -e "it/s" | tee "$O/log_bd47.txt" | tail -3
cd "$REPO"
[ -f "$R47_OUT/bd47.json" ] || { stamp "render FAILED"; exit 1; }
if [ "$SMOKE" = 1 ]; then deactivate; stamp "round 47 smoke done"; exit 0; fi
stamp "score"
python $Q/score47.py "$R46_DATA" "$R46_RUN" "$R47_T2" "$R47_SEP" "$R47_OUT" "$R43_DATA" "$R43_RUN" "$RES" 2>&1 \
    | grep -v -i warn | tee "$O/log_score.txt" | tail -60
deactivate
[ -f "$RES/score47.json" ] || { stamp "score FAILED"; exit 1; }
stamp "round 47 done"
