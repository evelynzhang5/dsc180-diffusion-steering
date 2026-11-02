#!/bin/bash
# Continue EDM's full-data CIFAR-10 training for as long as one DSMLP pod allows, then measure FID at each
# snapshot (notebook cell 39). Run inside a GPU pod from the repository root:
#     bash scripts/dsmlp_train_edm.sh HOURS TAG
# HOURS is the pod's time limit (12 with K8S_TIMEOUT_SECONDS=43200); TAG names the FID outputs.
# The first call starts from the 1,004-kimg snapshot of the November 1 run with --transfer (that run saved no
# optimizer state); every later call resumes the newest training-state-*.pt exactly with --resume.
HOURS=${1:-12}
TAG=${2:-run2}
DEADLINE=$(( $(date +%s) + HOURS * 3600 - 75 * 60 ))   # leave 75 min for snapshot FID
stamp() { echo "== [$(date '+%F %T')] $*"; }

REPO=$PWD
EDM=$REPO/downloads/edm-008a4e5316c8e3bfe61a62f874bddba254295afb
RUNS=$EDM/training-runs
DATA=$REPO/outputs/cifar10-32x32-n50000.zip
FIRST=$RUNS/00000-cifar10-32x32-n50000-uncond-ddpmpp-edm-gpus1-batch512-fp32/network-snapshot-001004.pkl

nvidia-smi -L
python -m pip install --user -q -r requirements.txt
STATE=$(ls -t "$RUNS"/*/training-state-*.pt 2>/dev/null | head -1)
if [ -n "$STATE" ]; then START="--resume=$STATE"; else START="--transfer=$FIRST"; fi

stamp "train.py $START for $(( (DEADLINE - $(date +%s)) / 60 )) min"
cd "$EDM" && timeout --signal=INT $(( DEADLINE - $(date +%s) )) torchrun --standalone --nproc_per_node=1 train.py \
    --outdir=training-runs --data="$DATA" --cond=0 --arch=ddpmpp --batch=512 --batch-gpu=64 \
    --duration=200 --snap=10 --dump=10 $START
stamp "train.py exit $?"

RUN=$(ls -td "$RUNS"/*/ | head -1)
stamp "snapshot FID for $RUN"
cd "$REPO" && python scripts/edm_snapshot_fid.py "${RUN%/}" "$TAG"
stamp "done"
