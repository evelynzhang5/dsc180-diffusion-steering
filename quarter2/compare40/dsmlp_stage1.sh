#!/bin/bash
# Q2 round 40 stage 1 (GPU pod, from the repo root): generate scenes, fetch RealBokeh pairs, BokehDiff's own depth +
# BiRefNet preparation, frozen measurement regions and contact sheets. No method is run. SMOKE=1: 2 + 2 images.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1
W17=$HOME/bokehdiff-smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
D=$REPO/outputs/q2_compare40/data
[ "$SMOKE" = 1 ] && D=$REPO/outputs/q2_compare40_smoke/data
rm -rf "$D" && mkdir -p "$D"
nvidia-smi -L
source "$W17/venv/bin/activate"
python -c "import diffusers, transformers, torch; print('diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
stamp "generate (RealVisXL V5.0)"
python "$REPO/quarter2/compare40/gen40.py" "$D" 2>&1 | grep -v -i warn || { stamp "generation FAILED"; exit 1; }
stamp "RealBokeh selection + download"
python "$REPO/quarter2/compare40/realbokeh40.py" "$D" 2>&1 | grep -v -i warn || { stamp "RealBokeh FAILED"; exit 1; }
stamp "BokehDiff's own preparation (Depth Anything V2 Base + BiRefNet)"
cd "$W17/repo" && git checkout -q $COMMIT && python prepare_data.py --root "$D" 2>&1 | grep -v -i -e warn -e "it/s" | tail -5
cd "$REPO"
n_in=$(ls "$D/input" | wc -l); n_d=$(ls "$D/depth" | wc -l); n_m=$(ls "$D/mask" | wc -l)
stamp "inputs $n_in depth $n_d masks $n_m"
[ "$n_in" = "$n_d" ] && [ "$n_in" = "$n_m" ] || { stamp "preparation FAILED"; exit 1; }
stamp "regions + contact sheets"
python "$REPO/quarter2/compare40/regions40.py" "$D" 2>&1 | tail -70
[ -f "$D/regions40.json" ] || { stamp "regions FAILED"; exit 1; }
stamp "stage 1 done"
