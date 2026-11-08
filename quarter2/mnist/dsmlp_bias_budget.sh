#!/bin/bash
# Reference-set bias under a limited slant-label budget, on a DSMLP GPU pod. Run from the repository root:
#     bash quarter2/mnist/dsmlp_bias_budget.sh
# Resumable: finished directions are kept in outputs/q2_mnist_bias_budget/runs.jsonl.
set -o pipefail
nvidia-smi -L
python -c "import skimage" 2>/dev/null || python -m pip install --user -q scikit-image || exit 1
echo "== [$(date '+%F %T')] start"
python quarter2/mnist/mnist_bias_budget.py --out outputs/q2_mnist_bias_budget || exit 1
echo "== [$(date '+%F %T')] done"
