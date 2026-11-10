#!/bin/bash
# CIFAR-10 replication of the paper's Table 2, on a DSMLP GPU pod. Run from the repo root:
#     bash cifar10_replication/dsmlp_replicate.sh                # everything
#     bash cifar10_replication/dsmlp_replicate.sh report big     # only these stages
# Each run starts with a smoke test of the same stages (2 classes, a few images, to outputs/cifar10_replication_smoke).
# Results go to cifar10_replication/results/.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
STAGES=${*:-all sweep big}
export HF_HOME=$HOME/hf-cache
# The pod gets 8 cores but sees the whole node (40); without this, BLAS starts 40 threads and the PCA crawls.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q datasets scikit-learn scipy transformers gdown || exit 1
stamp "smoke test: $STAGES"
rm -rf downloads/narfm/samples_smoke downloads/narfm/samples_big_smoke outputs/cifar10_replication_smoke
python cifar10_replication/replicate.py all --smoke || exit 1
for s in $STAGES; do
  [ "$s" = all ] || python cifar10_replication/replicate.py "$s" --smoke || exit 1
done
for s in $STAGES; do
  stamp "stage $s"
  python cifar10_replication/replicate.py "$s" || exit 1
done
stamp "done"
