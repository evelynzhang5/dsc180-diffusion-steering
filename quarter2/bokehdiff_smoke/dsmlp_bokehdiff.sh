#!/bin/bash
# Q2 round 17: can BokehDiff (SDXL-based lens-blur editor) run on a DSMLP GTX 1080 Ti? One 512x512 image.
# Run on a GPU pod from the repo root:  bash quarter2/bokehdiff_smoke/dsmlp_bokehdiff.sh
# Separate venv (system torch reused), upstream code pinned to one commit, nothing in our own code changes.
set -o pipefail
stamp() { echo "== [$(date '+%F %T')] $*"; }
REPO=$PWD
export HF_HOME=$HOME/hf-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export USE_TF=0 USE_TORCH=1 TRANSFORMERS_NO_TF=1   # the image has TensorFlow built for numpy 1; keep transformers off it
W=$HOME/bokehdiff-smoke
OUT=$REPO/outputs/q2_bokehdiff_smoke
COMMIT=38ff9372ab146ab20468832a0424fa0b2888dffd
mkdir -p "$W" "$OUT"
nvidia-smi -L
python -c "import torch; print('system torch', torch.__version__, 'cuda', torch.version.cuda)"

stamp "setup"
# keep the image's CUDA 12.1 PyTorch: newer wheels drop Pascal (sm_61) support; rebuild the venv with pins
python -c "import torch, torchvision, numpy; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}'); print('numpy<2')" > "$W/constraints.txt"
cat "$W/constraints.txt"
rm -rf "$W/venv" && python -m venv --system-site-packages "$W/venv"
source "$W/venv/bin/activate"
pip install -q -c "$W/constraints.txt" diffusers==0.32.1 transformers==4.47.1 peft==0.14.0 accelerate==1.2.1 \
    cupy-cuda12x opencv-python-headless scikit-image piq lpips pandas timm einops kornia "numpy<2" || { stamp "setup FAILED"; exit 1; }
python -c "import numpy, torch, diffusers, transformers, peft, cupy; print('venv numpy', numpy.__version__, 'torch', torch.__version__, diffusers.__version__, transformers.__version__, peft.__version__, cupy.__version__); assert torch.cuda.is_available(), 'CUDA not available'; print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" || { stamp "import check FAILED"; exit 1; }
[ -d "$W/repo" ] || git clone -q https://github.com/FreeButUselessSoul/bokehdiff.git "$W/repo"
cd "$W/repo" && git fetch -q origin && git checkout -q $COMMIT && git log -1 --format='bokehdiff commit %H'

stamp "input: repo test image EBB_4690.jpg, centre crop, 512x512"
mkdir -p "$W/data/input"
python - "$W/repo/test_data/input/EBB_4690.jpg" "$W/data/input/EBB_4690.jpg" <<'EOF'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
s = min(im.size)
l, t = (im.size[0] - s) // 2, (im.size[1] - s) // 2
im.crop((l, t, l + s, t + s)).resize((512, 512), Image.LANCZOS).save(sys.argv[2], quality=100, subsampling=0)
print("input", im.size, "->", (512, 512))
EOF

# peak GPU memory (whole card) and peak resident memory of the process tree, polled every 0.5 s
measure() {  # measure <tag> <cmd...>
  local tag=$1; shift
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$OUT/gpu_$tag.txt" &
  local gp=$!
  local t0=$(date +%s)
  "$@" > "$OUT/log_$tag.txt" 2>&1 &
  local pid=$!
  : > "$OUT/rss_$tag.txt"
  while kill -0 $pid 2>/dev/null; do
    ps -o rss= --ppid $pid -p $pid 2>/dev/null | awk '{s+=$1} END {print s}' >> "$OUT/rss_$tag.txt"; sleep 0.5
  done
  wait $pid; local rc=$?
  kill $gp 2>/dev/null
  local gpu=$(sort -n "$OUT/gpu_$tag.txt" | tail -1) rss=$(sort -n "$OUT/rss_$tag.txt" | tail -1)
  stamp "$tag: exit $rc, $(( $(date +%s) - t0 )) s, peak GPU ${gpu} MiB, peak RSS $(( ${rss:-0} / 1024 )) MiB"
  tail -3 "$OUT/log_$tag.txt"
  return $rc
}

stamp "prepare depth (Depth-Anything-V2 Base) and mask (BiRefNet), separate process"
measure prepare python prepare_data.py --root "$W/data" || { stamp "prepare FAILED"; exit 1; }

run() {  # run <tag> <K>  (fp16 via run_fp16.py's dtype cast + autocast, SDPA attention, no xformers, seed 0; 3 focal planes)
  measure "$1" python "$REPO/quarter2/bokehdiff_smoke/run_fp16.py" --test_data_dir "$W/data/input/*" --output_dir "$OUT" --data_id "$1" \
      --K "$2" --mixed_precision fp16 --seed 0
}
stamp "inference K 20 (fp16)"
run k20a 20 || { stamp "inference FAILED"; exit 1; }
stamp "repeat K 20"
run k20b 20
for k in 10 40; do stamp "inference K $k"; run k$k $k; done

stamp "checks"
python - "$OUT" <<'EOF'
import sys, json, glob, os
import numpy as np
from PIL import Image
out = sys.argv[1]
res = {}
for d in sorted(glob.glob(f"{out}/k*/")):
    for f in sorted(glob.glob(d + "*.jpg")):
        a = np.asarray(Image.open(f)).astype(np.float32)
        res[os.path.relpath(f, out)] = dict(shape=list(a.shape), mean=float(a.mean()), std=float(a.std()),
                                            finite=bool(np.isfinite(a).all()), black=bool(a.max() < 5))
pairs = sorted(glob.glob(f"{out}/k20a/*.jpg"))
rep = {}
for f in pairs:
    g = f.replace("/k20a/", "/k20b/")
    if os.path.exists(g):
        a, b = (np.asarray(Image.open(x)).astype(np.float32) for x in (f, g))
        rep[os.path.basename(f)] = float(np.abs(a - b).max())
src = np.asarray(Image.open(os.path.expanduser("~/bokehdiff-smoke/data/input/EBB_4690.jpg"))).astype(np.float32)
lap = lambda a: float(np.var(np.diff(a.mean(2), 2, axis=0)[:, 1:-1] + np.diff(a.mean(2), 2, axis=1)[1:-1, :]))
sharp = {k: lap(np.asarray(Image.open(os.path.join(out, k))).astype(np.float32)) for k in res}
summary = dict(outputs=res, repeat_max_abs_diff=rep, input_lapvar=lap(src), output_lapvar=sharp)
json.dump(summary, open(f"{out}/smoke.json", "w"), indent=1)
print("repeat max |diff|:", rep)
print("input lapvar %.1f; outputs:" % summary["input_lapvar"], {k: round(v, 1) for k, v in sharp.items()})
print("any non-finite or black:", [k for k, v in res.items() if not v["finite"] or v["black"]])
EOF
stamp "done"
