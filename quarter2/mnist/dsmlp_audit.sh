#!/bin/bash
# Side-effect audit (toy version of "predict unintended steering effects on unseen prompts"), on a DSMLP GPU pod.
# Run from the repository root:
#     bash quarter2/mnist/dsmlp_audit.sh
# Stage 1 is a small smoke test (one planted nuisance, few digits); stage 2 is the full run (resumable).
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
nvidia-smi -L
python -c "import skimage" 2>/dev/null || python -m pip install --user -q scikit-image || exit 1
stamp "smoke test"
rm -rf outputs/q2_mnist_audit_smoke
python quarter2/mnist/mnist_audit.py --out outputs/q2_mnist_audit_smoke --attrs slant --per-class 20 \
    --truth-per-class 20 || exit 1
stamp "full run"
python quarter2/mnist/mnist_audit.py --out outputs/q2_mnist_audit || exit 1
stamp "done"
