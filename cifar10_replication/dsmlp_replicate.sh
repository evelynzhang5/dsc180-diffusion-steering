#!/bin/bash
# CIFAR-10 replication of the paper's Table 2, on a DSMLP GPU pod. Run from the repo root:
#     bash cifar10_replication/dsmlp_replicate.sh
# Stage 1: smoke test (2 classes, 8 images each); stage 2: the full run; stage 3: the lambda sweep (encouraged).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
export HF_HOME=$HOME/hf-cache
# The pod gets 8 cores but sees the whole node (40); without this, BLAS starts 40 threads and the PCA crawls.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi -L
python -m pip install --user -q datasets scikit-learn scipy transformers gdown || exit 1
stamp "smoke test"
rm -rf downloads/narfm/samples_smoke outputs/cifar10_replication_smoke
python cifar10_replication/replicate.py all --smoke || exit 1
stamp "full run"
python cifar10_replication/replicate.py all || exit 1
stamp "sweep"
python cifar10_replication/replicate.py sweep || exit 1
stamp "done"
