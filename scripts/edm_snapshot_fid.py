"""FID across the snapshots of the full-data EDM training run (notebook cell 39), without the rest of the lab.

    python scripts/edm_snapshot_fid.py [RUN_DIR [TAG]]

Runs the unchanged cells of cifar10_edm_lab.ipynb that cell 39 needs (settings, downloads, models, sampler,
FID), then cell 39 itself with RUN_DIR set. Writes figures/edm_train/loss_fid{_TAG}.png and
results/edm_train/snapshot_fid{_TAG}.json. Run from the repository root on a GPU.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEEDED = [1, 3, 5, 7, 21]   # settings, downloads, models/data, sampler, FID
RUN_CELL = 39
DEFAULT_RUN = ("downloads/edm-008a4e5316c8e3bfe61a62f874bddba254295afb/training-runs/"
               "00000-cifar10-32x32-n50000-uncond-ddpmpp-edm-gpus1-batch512-fp32")

run_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
suffix = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
cells = json.load(open("cifar10_edm_lab.ipynb"))["cells"]
ns = {"__name__": "__main__"}
for i in NEEDED:
    exec(compile("".join(cells[i]["source"]), f"cell {i}", "exec"), ns)
    plt.close("all")

source = "".join(cells[RUN_CELL]["source"])
assert source.startswith("RUN_DIR = None")
exec(compile(source.replace("RUN_DIR = None", f"RUN_DIR = {run_dir!r}", 1), f"cell {RUN_CELL}", "exec"), ns)

Path("figures/edm_train").mkdir(parents=True, exist_ok=True)
Path("results/edm_train").mkdir(parents=True, exist_ok=True)
ns["fig"].savefig(f"figures/edm_train/loss_fid{suffix}.png", dpi=150, bbox_inches="tight")
Path(f"results/edm_train/snapshot_fid{suffix}.json").write_text(json.dumps(
    {"n_images": 1000, "seed_start": 3000, "run_dir": run_dir,
     "fid_by_million_images": {f"{k:g}": v for k, v in ns["snapshot_fid"].items()}}, indent=2))
print(f"wrote figures/edm_train/loss_fid{suffix}.png and results/edm_train/snapshot_fid{suffix}.json")
