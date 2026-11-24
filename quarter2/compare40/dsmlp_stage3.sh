#!/bin/bash
# Q2 round 40 stage 3 (GPU pod, from the repo root): both methods on the frozen stage 1 regions.
# dev grids (BokehDiff K, Bokehlicious f-number) -> map40 freezes the settings -> test at the frozen settings ->
# preparation latency -> score40 (checks; fails if exclusions, tile sides or frozen settings are violated).
# SMOKE=1: 2 mapping scenes + 1 RealBokeh dev scene on 3 settings each, 2 generated + 1 RealBokeh test scenes.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W17=$HOME/bokehdiff-smoke
BK=$HOME/bokehlicious
BKV=$HOME/bkl-venv
BK_COMMIT=0d2c45824de076ffc57ce222b60578302fb80923
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
export R40_DATA=$REPO/outputs/q2_compare40/data
export R40_OUT=$REPO/outputs/q2_compare40/s3
RES=$REPO/quarter2/compare40/results/stage3
[ "$SMOKE" = 1 ] && { R40_OUT=${R40_OUT}_smoke; RES=$R40_OUT/res_smoke; export R40_OUT; }
rm -rf "$R40_OUT" && mkdir -p "$R40_OUT"
nvidia-smi -L
python3 -c "import json; d=json.load(open('$R40_DATA/regions40.json')); assert all('front_tiles' in v for v in d.values()); print('regions: behind/front split present')" || { stamp "regions40.json is not the frozen version"; exit 1; }

bd() {   # $1 = phase
  source "$W17/venv/bin/activate"
  cd "$W17/repo" && git checkout -q $COMMIT
  PHASE=$1 python "$REPO/quarter2/compare40/bd40.py" --test_data_dir "$R40_DATA/input/hair_0.png" \
      --output_dir "$R40_OUT/upstream_unused" --data_id up --K 20 --mixed_precision fp16 --seed 0 2>&1 \
      | grep -v -i -e warn -e "it/s" | tee -a "$R40_OUT/log_bd.txt" | tail -4
  cd "$REPO"; deactivate
  [ -f "$R40_OUT/bd_$1.json" ] || { stamp "BokehDiff $1 FAILED"; exit 1; }
}
bkl() {
  cd "$BK" && git checkout -q $BK_COMMIT
  PHASE=$1 "$BKV/bin/python" "$REPO/quarter2/compare40/bkl40.py" 2>&1 | grep -v -i warn | tee -a "$R40_OUT/log_bkl.txt" | tail -3
  cd "$REPO"
  [ -f "$R40_OUT/bkl_$1.json" ] || { stamp "Bokehlicious $1 FAILED"; exit 1; }
}
stamp "dev grids"
bd dev
bkl dev
stamp "freeze mapping (dev scenes only)"
source "$W17/venv/bin/activate"
python quarter2/compare40/map40.py "$R40_DATA" "$R40_OUT" 2>&1 | tee "$R40_OUT/log_map.txt" | grep -v -i warn
[ -f "$R40_OUT/mapping.json" ] || { stamp "mapping FAILED"; exit 1; }
deactivate
stamp "test at frozen settings"
bd test
bkl test
stamp "preparation latency"
source "$W17/venv/bin/activate"
python quarter2/compare40/lat40.py "$R40_DATA" "$R40_OUT" 2>&1 | grep -v -i warn | tail -2
stamp "score"
python quarter2/compare40/score40.py "$R40_DATA" "$R40_OUT" "$RES" 2>&1 | tee "$R40_OUT/log_score.txt" | tail -60
[ -f "$RES/score40.json" ] || { stamp "scoring FAILED"; exit 1; }
cp "$R40_OUT/mapping.json" "$RES/" 2>/dev/null
stamp "stage 3 done"
