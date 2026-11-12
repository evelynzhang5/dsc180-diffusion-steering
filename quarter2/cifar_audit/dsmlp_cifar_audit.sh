#!/bin/bash
# Q2 round 5 (CIFAR-10 side-effect audit) on a DSMLP GPU pod. Run from the repo root:
#     bash quarter2/cifar_audit/dsmlp_cifar_audit.sh dev    # smoke test of every stage, then meters gen_dev predict
#     bash quarter2/cifar_audit/dsmlp_cifar_audit.sh test   # after the predictions are frozen: gen_test evaluate
# Needs the replication's setup (downloads/narfm: code, checkpoint, RFM directions, CIFAR-10, samples of seeds 0-255).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
PHASE=${1:?dev or test}
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -c "import transformers, datasets, scipy" 2>/dev/null || python -m pip install --user -q datasets scipy transformers || exit 1
S=quarter2/cifar_audit/cifar_audit.py
if [ "$PHASE" = dev ]; then
  stamp "smoke test, all stages"
  rm -rf downloads/narfm/samples_audit_smoke outputs/q2_cifar_audit_smoke
  python $S meters gen_dev predict gen_test evaluate --smoke || exit 1
  stamp "dev: meters gen_dev predict"
  python $S meters gen_dev predict || exit 1
else
  stamp "test: meters (rerun with the one-class set-meter check), gen_test, evaluate"
  python $S meters --smoke || exit 1
  python $S meters gen_test evaluate || exit 1
fi
stamp "done"
