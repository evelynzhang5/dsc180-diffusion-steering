#!/bin/bash
# Runs every CIFAR-10 EDM steering experiment from the repo README on one DSMLP GPU pod.
# DSMLP kills pods after 6 h, so EDM training gets whatever time is left.
T0=$(date +%s)
DEADLINE=$((T0 + 6 * 3600 - 25 * 60))     # stop training 25 min before the pod's 6 h limit
RUN=~/runs/$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN"
exec > >(tee -a "$RUN/log.txt") 2>&1
stamp() { echo "== [$(date '+%F %T')] $*"; }

stamp "start on $(hostname)"
nvidia-smi
cd ~ && (test -d dsc180-diffusion-steering || git clone https://github.com/evelynzhang5/dsc180-diffusion-steering.git)
cd ~/dsc180-diffusion-steering && git pull --ff-only
git log --oneline -1
python -m pip install --user -q -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || exit 1

run_nb() {
    local t=$(date +%s)
    stamp "nbconvert $1"
    python -m jupyter nbconvert --to notebook --execute "$1" --output "${1%.ipynb}.executed.ipynb" \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
    stamp "$1 exit $? after $(( ($(date +%s) - t) / 60 )) min"
}
run_nb cifar10_parts1_3.ipynb     # README figures 01-06 in figures/parts1_3/
run_nb cifar10_edm_lab.ipynb      # parts 1-6: accuracy/FID CFG sweep, subset training, dataset export

# Notebook cells 36-38: EDM's own train.py on all 50,000 images, the command the notebook prints
REPO=$PWD
EDM=$REPO/downloads/edm-008a4e5316c8e3bfe61a62f874bddba254295afb
DATA=$REPO/outputs/cifar10-32x32-n50000.zip
LEFT=$((DEADLINE - $(date +%s)))
if [ -f "$DATA" ] && [ -d "$EDM" ] && [ $LEFT -gt 1800 ]; then
    stamp "EDM train.py for $((LEFT / 60)) min"
    cd "$EDM" && timeout --signal=INT $LEFT torchrun --standalone --nproc_per_node=1 train.py \
        --outdir=training-runs --data="$DATA" --cond=0 --arch=ddpmpp --batch=512 --batch-gpu=64 --duration=10 --snap=10
    stamp "train.py exit $?; run folder: $(ls -d "$EDM"/training-runs/* | tail -1)"
else
    stamp "skipped EDM training (data=$DATA edm=$EDM seconds_left=$LEFT)"
fi
stamp "done; log at $RUN/log.txt"
