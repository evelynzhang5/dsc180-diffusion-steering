#!/bin/bash
# Q2 round 7 (NA x RFM interplay) on a DSMLP GPU pod. Run from the repo root:
#     bash quarter2/na_rfm_interplay/dsmlp_interplay.sh dev    # smoke test, then decompose tune select
#     bash quarter2/na_rfm_interplay/dsmlp_interplay.sh test   # confirm (needs the frozen selection)
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
PHASE=${1:?dev or test}
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -c "import transformers, datasets, scipy" 2>/dev/null || python -m pip install --user -q datasets scipy transformers || exit 1
S=quarter2/na_rfm_interplay/interplay.py
if [ "$PHASE" = dev ]; then
  stamp "smoke test, all stages"
  rm -rf downloads/narfm/samples_interplay_smoke outputs/q2_na_rfm_interplay_smoke
  python $S decompose tune select confirm --smoke || exit 1
  stamp "dev: decompose tune select"
  python $S decompose tune select || exit 1
else
  stamp "test: confirm"
  python $S confirm || exit 1
fi
stamp "done"
