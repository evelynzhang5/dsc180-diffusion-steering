"""Loss and snapshot FID of the two full-data EDM runs on one axis of total training images.

    python scripts/edm_combine_runs.py

Run 2 started from run 1's 1,004-kimg snapshot with --transfer, so its image count restarts at 0; this script shifts
it by the 1.004 million images run 1 had seen when that snapshot was saved. Reads results/edm_train/ (run 1) and
results/edm_train_run2/ plus results/edm_train/snapshot_fid_run2.json; writes figures/edm_train/loss_fid_combined.png.
No GPU needed.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFFSET = 1.004   # million images in run 1 at the snapshot run 2 transferred from


def loss_curve(path, offset=0.0):
    rows = [json.loads(line) for line in open(path)]
    return ([offset + r["Progress/kimg"]["mean"] / 1000 for r in rows], [r["Loss/loss"]["mean"] for r in rows])


def fid_points(path, offset=0.0):
    fid = json.load(open(path))["fid_by_million_images"]
    return [offset + float(k) for k in fid], list(fid.values())


fig, (ax_loss, ax_fid) = plt.subplots(1, 2, figsize=(12, 3.8))
for label, stats, fids, offset in [("run 1", "results/edm_train/stats.jsonl", "results/edm_train/snapshot_fid.json", 0.0),
                                   ("run 2 (--transfer from 1.004M)", "results/edm_train_run2/stats.jsonl",
                                    "results/edm_train/snapshot_fid_run2.json", OFFSET)]:
    ax_loss.plot(*loss_curve(stats, offset), label=label)
    ax_fid.plot(*fid_points(fids, offset), "o-", label=label)
ax_loss.set(title="Training loss, from stats.jsonl", xlabel="million training images", ylabel="loss", ylim=(0.17, 0.25))
ax_fid.set(title="FID across snapshots", xlabel="million training images", ylabel="FID (1,000 samples)", yscale="log")
ax_fid.axhline(31.2, color="gray", ls="--", lw=1, label="pretrained EDM, w = 0 (Table 1)")
for ax in (ax_loss, ax_fid):
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/edm_train/loss_fid_combined.png", dpi=150, bbox_inches="tight")
print("wrote figures/edm_train/loss_fid_combined.png")
