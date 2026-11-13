# Diffusion Steering: Classifier-Free Guidance Across Noise Levels

report: https://drive.google.com/file/d/1g770Y5qUykiv7fV8qXJDeAzsil7pHX60/view?usp=sharing

 We study classifier-free guidance (CFG) in diffusion models,
which steers a sample toward a class with a guidance weight w, and ask whether w
should change across noise levels instead of staying constant. This repository
holds:

- **Replication of the NA-RFM paper's CIFAR-10 experiment** (Wang, Belkin and Wang,
  ICML 2026, Table 2) with its public code: the ten classes' Gaussian denoisers and
  RFM directions, five sampling settings, accuracy, FID and time. See
  [Replicating the NA-RFM paper](#replicating-the-na-rfm-paper-cifar-10-table-2) below.
- **Quarter 1 checkpoint work:**
  - **CIFAR-10 with NVIDIA's pretrained EDM models** (course lab parts 1–6): sampling,
    class conditioning, CFG sweeps, and small-data training.
  - **Full-data EDM training** on CIFAR-10: about 14 GPU hours, FID 412 → 50.2.
  - **MNIST CFG network** trained from scratch, sampled with w = 0, 1, 2, 4.

## Repository layout

| Path | Contents |
| --- | --- |
| `cifar10_parts1_3.ipynb` | Lab parts 1–3, plus figure export and the part 3 trajectories (saved with outputs) |
| `cifar10_edm_lab.ipynb` | Full lab, parts 1–6 (saved with outputs from the Nov 1, 2026 DSMLP run) |
| `train_mnist.py`, `mnist_guidance_lab.py` | MNIST CFG network: training and evaluation; U-Net, sampler, classifier |
| `scripts/` | DSMLP batch jobs, snapshot FID, and the training-curve plot |
| `figures/` | Every figure used in the report |
| `results/` | Small result files: CFG sweep, training stats, FID values, job logs |
| `notes.md` | Quarter 1 findings, part by part, and run log |
| `cifar10_replication/` | Replication of Table 2 of the NA-RFM paper (Wang, Belkin and Wang, 2026) on CIFAR-10: code, README with commands, and all results (`results/report.md`) |
| `quarter2/` | Quarter 2 index: each experiment round (SD pilots, MNIST diagnostics, SD reference bias, SD depth of field) is on its own `q2-*` branch; see `quarter2/README.md` |
| `requirements.txt` | Packages added on top of the DSMLP scipy-ml image |

Downloaded data, model weights and run outputs (`downloads/`, `outputs/`, `*.pt`,
`*.pkl`) are not in Git; the notebooks and scripts fetch or regenerate them.

## Contribution statement (human and AI)

**Evelyn Zhang** chose the project and its research question within the course's
diffusion-steering topic; set up the GitHub repository and DSMLP access; supplied
the course materials (assignment, lab notebooks, LaTeX template); brainstormed
directions for online research; set the timeline and to-do list; decided which GPU
to use and when to stop training; monitored the logs; reviewed the code; verified
the visualizations and results; and wrote the rough draft and design of the report.

**AI (Claude Code, Anthropic, model Claude Opus 5.5)** was used primarily for:
implementing code (the DSMLP batch scripts in `scripts/`, `train_mnist.py`, the
snapshot-FID and plotting scripts); running and monitoring the experiments (the
CIFAR-10 notebooks and EDM training on DSMLP, MNIST training locally); debugging;
supplementing documentation (this README, `notes.md`); literature search and
reference checking; drafting and editing the report and porting it to LaTeX; and
explaining the methods to the author. The report's appendix ("AI Use") lists the
prompts and how the outputs were checked.

## Setup

CIFAR-10 work runs on a DSMLP GPU. From your laptop:

```bash
ssh UCSD_USERNAME@dsmlp-login.ucsd.edu
launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 4 -m 16    # 1 GPU, 4 CPUs, 16 GB RAM
```

Open the printed Jupyter URL (campus network or VPN), open a terminal, and run:

```bash
git clone https://github.com/evelynzhang5/dsc180-diffusion-steering.git
cd dsc180-diffusion-steering
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Keep the image's CUDA-enabled PyTorch (requirements.txt does not replace it).
The first notebook run downloads about 720 MB of data and checkpoints and checks
their SHA-256 hashes.

<details>
<summary>Off-campus access and the downloaded files</summary>

To reach Jupyter without the VPN, open a tunnel in a second laptop terminal
(PORT is the port in the printed URL), then replace
`http://dsmlp-login.ucsd.edu:PORT` with `http://localhost:8889` in that URL:

```bash
ssh -N -L 8889:dsmlp-login.ucsd.edu:PORT UCSD_USERNAME@dsmlp-login.ucsd.edu
```

Never commit the Jupyter token.

| Asset | Source |
| --- | --- |
| CIFAR-10 train/test | Hugging Face uoft-cs/cifar10, revision 0b2714987fa478483af9968de7c934580d0bb9a2 |
| EDM unconditional and conditional checkpoints, FID reference statistics | NVIDIA EDM pretrained CIFAR-10 models |
| Inception feature network (FID) | NVIDIA StyleGAN3 release |
| ResNet-56 classifier (label accuracy) | chenyaofo/pytorch-cifar-models |
| EDM source | NVlabs/edm commit 008a4e5316c8e3bfe61a62f874bddba254295afb |

Exact URLs and checksums are in the notebooks' download cell.
</details>

## Replicating the NA-RFM paper (CIFAR-10, Table 2)

Wang, Belkin and Wang, "General and Efficient Steering of Diffusion Models", ICML 2026
([arXiv 2602.11395](https://arxiv.org/abs/2602.11395); code
[isotrivial/na-rfm](https://github.com/isotrivial/na-rfm)). All code, results and
logs are in [`cifar10_replication/`](cifar10_replication/); its
[README](cifar10_replication/README.md) has the full details. Someone with this
repository and a GPU can rerun every step with the commands below.

### Data, model checkpoint and software

Install the packages, then let `setup` fetch the code, checkpoint and data (about 700 MB)
at fixed versions and apply the two code fixes:

```bash
python -m pip install --user datasets scikit-learn scipy transformers gdown
python cifar10_replication/replicate.py setup
```

| Item | Version / source |
| --- | --- |
| Paper code, [na-rfm](https://github.com/isotrivial/na-rfm) | commit `8e9619bc6735b03d73056bfdc13dd3659b96e127` |
| [Training-Free-Guidance](https://github.com/YWolfeee/Training-Free-Guidance) (TFG: U-Net and FID code) | commit `f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77`, cloned into `na-rfm/baselines/tfg` |
| Diffusion model checkpoint `openai_cifar10.pt` (improved-DDPM U-Net, 210 MB) | TFG's Google Drive, file id `1eInndCjx389Tl45kfmntVo23f_-82bHl`, md5 `2754c3e3c778ef6e0ab98e8aacaefaf9` (checked) |
| CIFAR-10 | Hugging Face `uoft-cs/cifar10` (downloaded automatically) |
| Accuracy evaluator (the paper's) | Hugging Face `ahsanjavid/convnext-tiny-finetuned-cifar10` |
| Python packages | DSMLP `scipy-ml` image (PyTorch, torchvision, matplotlib) plus `pip install datasets scikit-learn scipy transformers gdown` |

If `gdown` fails, download `openai_cifar10.pt` by hand from
[TFG's folder](https://drive.google.com/drive/folders/1fS7dKpO4O-FjaLwuRXuHBxEOlkqMMTGh)
into `downloads/narfm/na-rfm/data/checkpoints/`. Everything downloaded or generated goes
to `downloads/narfm/` (about 2 GB with the 2,048-image runs) and is not in Git.

**The two code fixes** (`setup` applies both):

1. na-rfm's `repro/evaluate_saved_samples.py` calls `load_dataset("cifar10")`, a name current
   `datasets` rejects; it becomes `load_dataset("uoft-cs/cifar10")`.
2. TFG's `evaluations/utils/torch_sqrtm.py` imports `scipy.linalg._matfuncs_sqrtm_triu`,
   which fails in SciPy 1.16 unless `scipy.linalg._matfuncs_sqrtm` is imported first; the
   fix adds that import.

### Run it

On DSMLP, start a pod with 1 GPU, 8 CPUs and 32 GB
(`launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 8 -m 32`), clone the repository, and
from its root run:

```bash
bash cifar10_replication/dsmlp_replicate.sh    # smoke test, then every stage below
```

It sets `OMP_NUM_THREADS=8`; without it a pod sees all of the node's cores and the PCA step
takes hours instead of minutes. Every stage skips work already on disk, so an interrupted
run resumes. On a GTX 1080 Ti: offline parts about 10 minutes, the five settings with FID
about 1.5 hours, the λ sweep about 1 hour, the 2,048-image runs about 3.5 hours.

### Command for each table and figure in the report

All outputs are in `cifar10_replication/results/`.

| Table or figure | Command | Output |
| --- | --- | --- |
| Everything below | `python cifar10_replication/replicate.py all sweep big` | all of `results/` |
| Gaussian denoisers (PCA, 1,000 directions per class) and RFM directions (block 9, σ = 0.21) for all ten classes; RFM AUC per class | `python cifar10_replication/replicate.py setup offline` | `rfm_auc.json` |
| Mean image and first PCA directions of two classes | `python cifar10_replication/replicate.py report` | `figures/pca_mean_directions.png` |
| Samples, five settings, 256 per class (seeds 0–255, 100 DDIM steps) | `python cifar10_replication/replicate.py generate` | `downloads/narfm/samples/` |
| Accuracy (ConvNeXt-Tiny) and per-class FID | `python cifar10_replication/replicate.py measure` | `quality_<setting>.json` |
| Sampling time at batch 64 (and 16) | `python cifar10_replication/replicate.py time` | `timing.json` |
| Table 2 comparison, per-class table, first 16 images of every class per setting | `python cifar10_replication/replicate.py report` | `report.md`, `figures/grid_<setting>.png` |
| Noise alignment only over λ = 1–12: accuracy against FID (paper's Figure 4) | `python cifar10_replication/replicate.py sweep` | `sweep.json`, `figures/sweep_acc_fid.png` |
| NA-RFM and noise alignment (λ = 3) at 2,048 per class; FID at 256 to 2,048 | `python cifar10_replication/replicate.py big` | `big.json`, `figures/fid_vs_images.png` |
| Quick end-to-end check (2 classes, 8 images) | `python cifar10_replication/replicate.py all --smoke` | `outputs/cifar10_replication_smoke/` |

To view the saved tables and figures without a GPU, open
[`cifar10_replication/results_notebook.ipynb`](cifar10_replication/results_notebook.ipynb).

### Results

256 images per class, GTX 1080 Ti. FID is per class against that class's 6,000 CIFAR-10
images, averaged over the ten classes. The paper's FIDs use 2,048 images per class, so they
are lower than ours for that reason alone; compare our rows with each other.

| Setting | Accuracy, ours | Accuracy, paper | FID, ours (256) | FID, paper (2,048) | s per image, batch 64 |
| --- | --- | --- | --- | --- | --- |
| Unguided (not in Table 2) | 10.0% | – | 157.8 | – | 0.199 |
| Noise alignment only, λ = 3 | 62.0% | 62.0% | 91.0 | 99.0 | 0.200 |
| Noise alignment only, λ = 8 | 79.8% | 80.0% | 111.0 | 120.0 | 0.200 |
| RFM only | 96.3% | 94.8% (94.3% at 256) | 94.9 | 40.3 | 0.398 |
| NA-RFM | 95.7% | 96.6% (97.9% at 256) | 94.1 | 41.4 | 0.340 |

At the paper's 2,048 images per class, our NA-RFM has accuracy 95.8% and FID 45.6 (paper
41.4). Noise alignment (λ = 3) has accuracy 62.4% and FID 37.1, against the paper's 99.0.
`results/report.md` has the per-class numbers, all figures and the comparison.

## Reproduce the results

### 1. Lab parts 1–3: sampling and CFG (about 1 minute on a GPU)

Run all cells of `cifar10_parts1_3.ipynb`, or from the terminal:

```bash
python -m jupyter nbconvert --to notebook --execute cifar10_parts1_3.ipynb --output cifar10_parts1_3.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
```

| Figure (`figures/parts1_3/`) | What it shows |
| --- | --- |
| `01_unconditional_samples.png` | 30 unconditional samples (seed 0) |
| `02_unconditional_trajectory.png` | how 3 images form as noise is removed |
| `03_conditional_samples.png` | 8 samples per class (seed 1) |
| `04_cfg_cat_grid.png` | "cat" at w = 0, 1, 1.5, 2, 3, 4 (seed 2) |
| `05_cat_trajectory_w1.png`, `06_cat_trajectory_w4.png` | how a cat forms at w = 1 vs. w = 4 |

All use EDM's 18-step sampler (σ from 80 to 0.002, ρ = 7).

### 2. Lab parts 4–6 and full-data EDM training (one 6-hour DSMLP job)

`scripts/dsmlp_run_all.sh` runs both notebooks and then trains EDM on all of
CIFAR-10 for the rest of the pod's time. Submit it from dsmlp-login:

```bash
launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 8 -m 64 -B -N steer-run -- \
    bash -c "$(cat scripts/dsmlp_run_all.sh)"
```

On a GTX 1080 Ti, parts 4–6 took 135 minutes and training used the remaining
197. To continue training in a 12-hour pod and then measure FID per snapshot:

```bash
bash scripts/dsmlp_train_edm.sh 12 run2          # trains, then runs scripts/edm_snapshot_fid.py
python scripts/edm_combine_runs.py               # joins both runs into one plot
```

| Figure | What it shows |
| --- | --- |
| `figures/lab/07_fid_2d.png` | FID in two dimensions (part 4) |
| `figures/lab/08_cfg_sweep_acc_fid.png` | accuracy, FID and FID_class against w (part 5) |
| `figures/lab/09_part6_loss_fid_snapshots.png` | small-data training: loss and FID (part 6) |
| `figures/lab/10_part6_samples_n100.png`, `11_part6_samples_n1000.png` | samples across training, 100 and 1,000 images |
| `figures/lab/12_part6_nearest_n100.png`, `13_part6_nearest_n1000.png` | samples beside their nearest training images |
| `figures/lab/14_part6_cfg_grid_n100.png`, `15_part6_cfg_grid_n1000.png` | one-network CFG grids, w = 0, 1, 2, 4 |
| `figures/edm_train/loss_fid_combined.png` | full-data training: loss and FID over both runs |

The `figures/lab/` images are the saved outputs of `cifar10_edm_lab.ipynb`,
extracted in cell order.

<details>
<summary>Results tables (parts 5 and 6, full-data training)</summary>

**Part 5, CFG sweep** (two-network CFG, 1,000 images per w, balanced classes;
`results/cfg_sweep_n1000.json`). The classifier scores 94.4% on real test images.
These 1,000-image FIDs can only be compared with each other.

| w | 0 | 1 | 1.5 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| label accuracy | 11.5% | 95.5% | 98.7% | 98.8% | 99.5% | 98.9% |
| FID | 31.19 | 31.33 | 31.44 | 33.05 | 36.03 | 39.03 |
| FID_class (100 per class) | – | 112.1 | 107.6 | 108.9 | 112.4 | 117.5 |

**Part 6, training on 100 and 1,000 images** (7.0M-parameter U-Net, label
dropout 0.1, batch 64, FID on 1,000 samples per snapshot):

| update | 4,000 | 8,000 | 12,000 | 16,000 | 20,000 |
| --- | --- | --- | --- | --- | --- |
| FID, 100 images | 75.8 | 92.2 | 104.8 | 106.6 | 108.6 |
| FID, 1,000 images | 93.9 | 111.3 | 124.9 | 124.9 | 130.7 |

Median pixel distance to the nearest training image: 100-image model 1.3
(real test images 23.7); 1,000-image model 14.0 (real 21.3). One-network CFG
label accuracy at w = 0, 1, 2, 4 (100 samples each): 8%, 93%, 100%, 100%
(100 images); 6%, 49%, 76%, 89% (1,000 images).

**Full-data EDM training** (`--cond=0 --arch=ddpmpp --batch=512`, all 50,000
images; FID on 1,000 samples per snapshot; `results/edm_train/`,
`results/edm_train_run2/`):

| million images seen | 0 | 0.50 | 1.00 | 2.01 | 3.01 | 4.02 | 4.52 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FID | 412.4 | 105.5 | 76.5 | 67.7 | 56.2 | 52.2 | 50.2 |

Run 1 stopped at 1.20 million images (3.3 hours). Run 2 started from its
1.00-million snapshot with `--transfer` (new optimizer) and trained 3.76 million
more (10.6 hours). It saves its optimizer state every 10 ticks (`--dump=10`), so
it can be continued exactly with `--resume`. Snapshots stay on DSMLP.
</details>

### 3. MNIST CFG network (runs on a laptop: CUDA, Apple MPS or CPU)

The lab's U-Net trained from scratch on the 60,000 MNIST training digits, with
the label dropped (set to "no label") for 10% of examples, then sampled at
w = 0, 1, 2, 4:

```bash
python train_mnist.py train                      # about 22 min on an Apple M-series GPU
python train_mnist.py eval --out figures/mnist   # 5,000 samples per w
cp outputs/mnist_cfg/loss.png figures/mnist/
```

| Figure (`figures/mnist/`) | What it shows |
| --- | --- |
| `loss.png` | training loss per step |
| `cfg_grid.png` | one row per w, digits 0–9, same noise in every row |
| `cfg_metrics.png` (and `.csv`, `.json`) | label accuracy, FID, FID_class and pixel spread per w |

<details>
<summary>Training and evaluation settings</summary>

30 epochs, batch 256, AdamW at 2e-4 with warmup and cosine decay, EMA 0.999,
clean-image MSE on the lab's cosine path, seed 0. Sampling uses the lab's
16-step DDIM sampler; image i asks for digit i mod 10, with the same starting
noise for every w. FID uses the course's frozen digit classifier (128 features)
against the 10,000 test digits, so it is not comparable with Inception FIDs;
the JSON also records real digits' FID as a reference value. FID_class at w = 0 is not
meaningful, since w = 0 ignores the label. To score the course's pretrained
generator instead, add `--checkpoint generator.pt --out outputs/mnist_course`.
</details>

## References

- [Course DSMLP lesson](https://dsc-capstone.org/2026-27/lessons/01/) and [Q1 project requirements](https://dsc-capstone.org/2026-27/assignments/projects/q1/)
- [NA-RFM paper](https://arxiv.org/abs/2602.11395) and [code](https://github.com/isotrivial/na-rfm); [Training-Free-Guidance](https://github.com/YWolfeee/Training-Free-Guidance)
- [EDM paper](https://arxiv.org/abs/2206.00364) and [source](https://github.com/NVlabs/edm)
- [CIFAR-10 data](https://huggingface.co/datasets/uoft-cs/cifar10) and [classifiers](https://github.com/chenyaofo/pytorch-cifar-models)
- [UCSD container launch guide](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269)
- [Checkpoint Report](https://drive.google.com/file/d/1g770Y5qUykiv7fV8qXJDeAzsil7pHX60/view?usp=sharing)