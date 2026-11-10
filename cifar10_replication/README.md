# Replicating Table 2 of the NA-RFM paper on CIFAR-10

Wang, Belkin and Wang, "General and Efficient Steering of Diffusion Models", ICML 2026
([arXiv 2602.11395](https://arxiv.org/abs/2602.11395)). We rerun its CIFAR-10 experiment (Table 2) with its public
code at 256 images per class, and NA-RFM and noise alignment (λ = 3) also at the paper's 2,048 per class.

## What is in this folder

| Path | Contents |
|---|---|
| `replicate.py` | Every step: setup, the offline parts, sampling, accuracy and FID, timing, table and figures |
| `dsmlp_replicate.sh` | Runs `replicate.py` on a DSMLP GPU pod: a smoke test first, then the requested stages |
| `starter_notebook.ipynb` | The course's starter notebook, kept for reference; `replicate.py` does the same steps as a script |
| `results/report.md` | The table next to the paper's Table 2, per-class results, all figures |
| `results/figures/` | Sample grids, PCA figure, λ sweep, FID against the number of images |
| `results/*.json` | Raw numbers: `quality_<setting>.json` (accuracy, FID per class), `timing.json`, `rfm_auc.json`, `sweep.json`, `big.json` |
| `results/logs/` | Logs of the DSMLP runs |

Samples, the checkpoint and the paper's code go to `downloads/narfm/` (about 2 GB with the 2,048-image runs; not
committed).

## What gets installed and downloaded

`replicate.py setup` does all of this:

| Item | Version / source |
|---|---|
| Paper code, [na-rfm](https://github.com/isotrivial/na-rfm) | commit `8e9619bc6735b03d73056bfdc13dd3659b96e127` |
| [Training-Free-Guidance](https://github.com/YWolfeee/Training-Free-Guidance) (FID code, U-Net) | commit `f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77`, in `na-rfm/baselines/tfg` |
| Diffusion model `openai_cifar10.pt` (improved-DDPM U-Net, 210 MB) | TFG's Google Drive, id `1eInndCjx389Tl45kfmntVo23f_-82bHl`, md5 `2754c3e3c778ef6e0ab98e8aacaefaf9` (checked) |
| CIFAR-10 | Hugging Face `uoft-cs/cifar10` (downloaded automatically) |
| Accuracy evaluator | Hugging Face `ahsanjavid/convnext-tiny-finetuned-cifar10` (the paper's) |
| Python packages | `pip install datasets scikit-learn scipy transformers gdown` on top of the DSMLP `scipy-ml` image (PyTorch, torchvision, matplotlib) |

The two code fixes (applied by `setup`):
1. `repro/evaluate_saved_samples.py` loads `load_dataset("cifar10")`; current `datasets` rejects the old name, so it
   becomes `load_dataset("uoft-cs/cifar10")`.
2. TFG's `evaluations/utils/torch_sqrtm.py` imports `scipy.linalg._matfuncs_sqrtm_triu`, which fails in SciPy 1.16
   unless `scipy.linalg._matfuncs_sqrtm` is imported first; the fix adds that import.

If `gdown` fails, download `openai_cifar10.pt` by hand from
[TFG's folder](https://drive.google.com/drive/folders/1fS7dKpO4O-FjaLwuRXuHBxEOlkqMMTGh) into
`downloads/narfm/na-rfm/data/checkpoints/`.

## Commands

From the repo root, on a GPU. On DSMLP, `bash cifar10_replication/dsmlp_replicate.sh` runs a smoke test and then
`all sweep big`. Set `OMP_NUM_THREADS=8` (the script does): a pod sees all of the node's cores, and without it the
PCA step runs for hours instead of three minutes.

| Output in the report | Command |
|---|---|
| Everything below | `python cifar10_replication/replicate.py all sweep big` |
| Quick end-to-end check (2 classes, 8 images; to `outputs/cifar10_replication_smoke/`) | `python cifar10_replication/replicate.py all --smoke` |
| Gaussian denoisers and RFM directions for all ten classes; RFM AUCs (`rfm_auc.json`) | `python cifar10_replication/replicate.py setup offline` |
| Samples for the five settings | `python cifar10_replication/replicate.py generate` |
| Accuracy and FID (`quality_<setting>.json`) | `python cifar10_replication/replicate.py measure` |
| Sampling time (`timing.json`) | `python cifar10_replication/replicate.py time` |
| Table, per-class table, sample grids (`figures/grid_<setting>.png`), PCA figure (`figures/pca_mean_directions.png`), `report.md` | `python cifar10_replication/replicate.py report` |
| λ sweep, noise alignment only (`sweep.json`, `figures/sweep_acc_fid.png`) | `python cifar10_replication/replicate.py sweep` |
| NA-RFM and noise alignment only (λ = 3) at 2,048 images per class, FID at 256 to 2,048 (`big.json`, `figures/fid_vs_images.png`) | `python cifar10_replication/replicate.py big` |

Every stage skips work already on disk, so an interrupted run resumes. On a GTX 1080 Ti: offline parts about 10
minutes, the five settings about 1.5 hours with FID, the sweep about 1 hour, the 2,048-image runs about 3.5 hours.

## Settings

All five settings use seeds 0–255 for every class and 100 DDIM steps (η = 0), at batch size 64.

| Setting | Script and flags |
|---|---|
| Unguided | `run_cifar10_na_rfm.py --gaussian-strength 0 --rfm-strength 0 --rfm-sigma-lo 1e8 --rfm-sigma-hi 1e9` |
| Noise alignment only, λ = 3 / 8 | `run_cifar10_na_only.py --mode diff --strength 3` (or `8`) `--start-step 0 --end-step 50 --num-components 1000 --eval-mode hf_convnext` |
| RFM only (all steps) | `run_cifar10_na_rfm.py --gaussian-strength 0 --rfm-sigma-lo 0 --rfm-sigma-hi 1e4` |
| NA-RFM | `run_cifar10_na_rfm.py` defaults: λ = 3 for σ ≥ 3.33; w_RFM = 1, s = 2 for σ ∈ [0.01, 11.69] |

The unguided setting puts both steering windows above every σ the sampler visits (the largest is about 157). That
gives the same images as `--gaussian-strength 0 --rfm-strength 0`, but with one network call per step instead of two,
so its time is the time of plain DDIM.

Offline parts: Gaussian denoisers from PCA with 1,000 directions of each class's 5,000 training images, and of all
50,000; RFM directions from block-9 (`input_blocks_9`) features of the first 1,000 training images of each class at
σ = 0.21 (step t = 60). The RFM settings are a Laplace kernel, bandwidth 100, regularization 10⁻³, 5 iterations, and the
top 3 eigenvectors; AUC is measured on a held-out 20%.

Measurement: accuracy is the share of images the ConvNeXt-Tiny evaluator assigns to the requested class. FID compares
each class's images with that class's 6,000 CIFAR-10 images (train and test) and is averaged over the ten classes.
Time is the fastest of two batches after a warm-up batch, at batch size 64 and at 16 (Table 2's setting).
