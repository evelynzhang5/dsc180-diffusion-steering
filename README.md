# Diffusion steering capstone — CIFAR-10 EDM

Study unconditional generation, class conditioning, and classifier-free guidance
(CFG) with the pretrained NVIDIA EDM CIFAR-10 models. This starter reproduces
parts 1–3 of the supplied capstone lab. Record observations in notes.md.

**Status (November 1, 2026):** parts 1–6 of the CIFAR-10 lab ran end to end on a
DSMLP GTX 1080 Ti (commit e4aff85), followed by 3.3 hours (1.2 million images) of
EDM's full-data training. The MNIST CFG network was trained locally. Not yet done:
FID across the full-data training snapshots (notebook cell 39), and the report.

## Files

- cifar10_edm_lab.ipynb: the full supplied lab, unchanged (parts 1–6).
- cifar10_parts1_3.ipynb: parts 1–3 only, plus a CUDA check, figure export, and the part 3 trajectories.
- guidance_mnist.ipynb: the supplied MNIST guidance lab (pretrained generator; parts 1–4).
- mnist_guidance_lab.py: the lab's helpers (U-Net, CFG sampler, frozen digit classifier), as written by guidance_mnist.ipynb.
- train_mnist.py: the MNIST CFG network deliverable: trains the lab's U-Net from scratch and measures CFG.
- requirements.txt: packages added on top of the DSMLP scipy-ml image.
- notes.md: run log, predictions, and observations.
- report.md: checkpoint report draft (title, abstract, introduction, preliminary results).
- handoff.md: deliverables checklist, project direction, open questions.
- concepts.md: background theory (study notes).
- scripts/dsmlp_run_all.sh: the DSMLP batch job that ran both CIFAR-10 notebooks and EDM training.
- figures/parts1_3/: parts 1–3 figures, exported by cifar10_parts1_3.ipynb.
- figures/lab/: parts 4–6 figures, extracted from the executed full lab notebook.
- figures/edm_train/: loss curve of the full-data EDM training run.
- results/: small result files: the part 5 sweep, EDM training stats and log, and the batch job log.
- downloads/ and outputs/: local working data; excluded from Git.

## DSMLP setup

On your laptop, replace UCSD_USERNAME with your campus username:

```bash
ssh UCSD_USERNAME@dsmlp-login.ucsd.edu
```

On the DSMLP login host, launch the compute container:

```bash
launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 4 -m 16
```

This requests one GPU, four CPU cores, and 16 GB of system RAM. Keep this
terminal open for this foreground session. Run Python only in the launched
container, not on dsmlp-login itself.

Open the printed Jupyter URL while connected to the campus network or UCSD VPN.
Alternatively, in a second terminal on your laptop, replace PORT with the port
from that URL:

```bash
ssh -N -L 8889:dsmlp-login.ucsd.edu:PORT UCSD_USERNAME@dsmlp-login.ucsd.edu
```

Keep that terminal open. In the printed URL, replace only
http://dsmlp-login.ucsd.edu:PORT with http://localhost:8889; retain the path and
token. Do not put a real token into this repository.

In Jupyter, open your private directory, then open a terminal there. Clone your
public repository (replace YOUR_GITHUB_USERNAME and the repository name if needed):

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/dsc180-diffusion-steering.git
cd dsc180-diffusion-steering
python -m pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Use the container's Python 3 kernel. Retain the image's CUDA-enabled PyTorch;
requirements.txt deliberately does not replace it.

## Data downloads

Run the shortened notebook from the repository directory. Its download cell
fetches approximately 720 MB on a first run, checks the provided SHA-256 hashes
for data/model files, and reuses verified files on subsequent runs.

| Asset | Source and purpose |
| --- | --- |
| CIFAR-10 train/test Parquet files | Hugging Face uoft-cs/cifar10, revision 0b2714987fa478483af9968de7c934580d0bb9a2; original CIFAR-10 images |
| EDM unconditional and conditional VP checkpoints | NVIDIA EDM pretrained CIFAR-10 models |
| CIFAR-10 FID reference statistics | NVIDIA EDM; downloaded now for later evaluation |
| Inception feature network | NVIDIA StyleGAN3 release; needed for later FID |
| CIFAR-10 ResNet-56 weights | chenyaofo/pytorch-cifar-models; needed for later label accuracy |
| EDM source archive | NVlabs/edm commit 008a4e5316c8e3bfe61a62f874bddba254295afb |

The exact URLs, checksums, retry behavior, and compatibility fixes are in the
notebook's download cell. An existing original CIFAR-10 tar archive is also
supported by that cell. Raw data and model checkpoints should not be committed.

## Reproduce parts 1–3

Interactive: open cifar10_parts1_3.ipynb in Jupyter, restart its kernel, and run
all cells in order. Its first added cell requires CUDA. The lab's initialization
should print:

```text
device cuda, QUICK = False, 1000 images per setting
```

The 1,000-image setting applies to the later evaluation sweep; these initial
sample grids contain fewer images.

For a fresh, sequential run from the container terminal, use this exact command:

```bash
python -m jupyter nbconvert --to notebook --execute cifar10_parts1_3.ipynb --output cifar10_parts1_3.executed.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
```

Use either interactive execution or this command, not both simultaneously.
Shut down the interactive GPU kernel before starting a separate command-line run.
The command downloads its inputs and produces every figure below. It intentionally
does not execute parts 4–6. Restart and run all if you need to regenerate figures;
the automatic export names assume the original cell order.

| Figure | Notebook source / settings | Reproduction command |
| --- | --- | --- |
| figures/parts1_3/01_unconditional_samples.png | Part 1; 30 images; seed 0 | The parts 1–3 nbconvert command above |
| figures/parts1_3/02_unconditional_trajectory.png | Part 1; 3 images; seed 0 | The parts 1–3 nbconvert command above |
| figures/parts1_3/03_conditional_samples.png | Part 2; 10 classes x 8 images; seed 1 | The parts 1–3 nbconvert command above |
| figures/parts1_3/04_cfg_cat_grid.png | Part 3; cat; w = 0, 1, 1.5, 2, 3, 4; seed 2 | The parts 1–3 nbconvert command above |
| figures/parts1_3/05_cat_trajectory_w1.png | Added part 3 exercise; cat; 3 images; seed 0; w = 1 | The parts 1–3 nbconvert command above |
| figures/parts1_3/06_cat_trajectory_w4.png | Added part 3 exercise; cat; 3 images; seed 0; w = 4 | The parts 1–3 nbconvert command above |

All six use the original 18-step sampler, sigma_min = 0.002,
sigma_max = 80, and rho = 7. Actual Python, package, PyTorch, CUDA, and GPU details
are written to figures/parts1_3/run_environment.json after successful completion.
Seeds are fixed; exact floating-point outputs may differ across software/hardware.

There are no numeric results tables in parts 1–3.

## Reproduce parts 4–6 and the full-data training run

All of these come from one DSMLP batch job, scripts/dsmlp_run_all.sh. It clones
or updates this repository in the pod's home directory, installs requirements.txt,
executes cifar10_parts1_3.ipynb and then the full cifar10_edm_lab.ipynb with
nbconvert, and spends the rest of the pod's 6-hour limit on EDM's train.py (the
command printed by notebook cell 37). Submit it from a clone of this repository
on dsmlp-login (the login host's home directory is not the pod's):

```bash
launch-scipy-ml.sh -W DSC180A_FA26_001 -g 1 -c 8 -m 64 -B -N steer-run -- \
    bash -c "$(cat scripts/dsmlp_run_all.sh)"
```

`-B` queues the pod and returns; follow it with `kubectl get pods` and
`kubectl logs steer-run` (run `source /opt/launch-sh/lib/kubevars.sh` first). The
log is also written to ~/runs/<timestamp>/log.txt. The DSC180A workspace only
schedules small GPUs; the run below used a GTX 1080 Ti (11 GB). Parts 1–3 took
1 minute, parts 4–6 took 135 minutes, and training took the remaining 197 minutes.
To run the lab interactively instead, open cifar10_edm_lab.ipynb and run all cells.

### Part 5: CFG sweep (1,000 images per w, balanced classes, two-network CFG)

Source: notebook cell 26; results/cfg_sweep_n1000.json; plots figures/lab/08_cfg_sweep_acc_fid.png.
Accuracy uses the ResNet-56 classifier (94.4% on 10,000 real test images). FID uses
EDM's reference statistics of the 50,000 training images; FID_class uses 100
images per class against that class's 5,000 training images.

| w | 0 | 1 | 1.5 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| label accuracy | 11.5% | 95.5% | 98.7% | 98.8% | 99.5% | 98.9% |
| FID (1,000 images) | 31.19 | 31.33 | 31.44 | 33.05 | 36.03 | 39.03 |
| FID_class (100 per class) | – | 112.1 | 107.6 | 108.9 | 112.4 | 117.5 |

These FIDs use 1,000 images, so compare them only with each other, not with
EDM's published 50,000-image values.

### Part 6: training on 100 and 1,000 images, with CFG

Source: notebook cells 29–35. Network: EDM's SongUNet reduced to 7.0M parameters,
label dropout 0.1, batch 64, 20,000 updates per training set, snapshot every 4,000
updates, FID on 1,000 unconditional samples per snapshot.

| update | 4,000 | 8,000 | 12,000 | 16,000 | 20,000 |
| --- | --- | --- | --- | --- | --- |
| FID, 100 training images | 75.8 | 92.2 | 104.8 | 106.6 | 108.6 |
| FID, 1,000 training images | 93.9 | 111.3 | 124.9 | 124.9 | 130.7 |

Median pixel distance to the nearest training image at the last snapshot:
100 images: samples 1.3 vs. real test images 23.7; 1,000 images: samples 14.0
vs. real test images 21.3. Label accuracy of one-network CFG (10 samples shown):
100 images w = 0, 1, 2, 4: 8%, 93%, 100%, 100%; 1,000 images: 6%, 49%, 76%, 89%.

| Figure | What it shows |
| --- | --- |
| figures/lab/07_fid_2d.png | FID in two dimensions (part 4) |
| figures/lab/08_cfg_sweep_acc_fid.png | part 5: accuracy, FID, FID_class against w |
| figures/lab/09_part6_loss_fid_snapshots.png | part 6: training loss and FID across snapshots |
| figures/lab/10_part6_samples_n100.png, 11_part6_samples_n1000.png | unconditional samples from the same noise at each snapshot |
| figures/lab/12_part6_nearest_n100.png, 13_part6_nearest_n1000.png | samples next to their nearest training images |
| figures/lab/14_part6_cfg_grid_n100.png, 15_part6_cfg_grid_n1000.png | one-network CFG grids, w = 0, 1, 2, 4 |

The figures/lab images are the embedded outputs of the executed notebook,
extracted in cell order (the executed notebook itself is not committed).

### Full-data EDM training (encouraged)

EDM's train.py with `--cond=0 --arch=ddpmpp --batch=512 --batch-gpu=64 --duration=10 --snap=10`
on all 50,000 images, stopped by the time limit after 1,205 kimg (24 ticks,
3.3 hours, about 9.7 s per kimg). Loss fell from 1.02 at the start to 0.196.
Snapshots were saved at 0, 502 and 1,004 kimg (kept on DSMLP; not in Git).
Training stats: results/edm_train/stats.jsonl, training_options.json, log.txt;
loss curve: figures/edm_train/loss.png. To continue, rerun train.py with
`--resume` pointing at the newest training-state-*.pt; for FID across snapshots,
set RUN_DIR in notebook cell 39 to the run folder
`downloads/edm-008a4e5316c8e3bfe61a62f874bddba254295afb/training-runs/00000-cifar10-32x32-n50000-uncond-ddpmpp-edm-gpus1-batch512-fp32`
and run that cell on a GPU.

## MNIST CFG network

The checkpoint asks for the lab's U-Net trained from scratch on the 60,000 MNIST
training digits, with the label replaced by 10 ("no label") for a random 10% of
examples, then sampled with w = 0, 1, 2, 4. train_mnist.py does both steps. It
needs PyTorch, torchvision, SciPy and matplotlib, and runs on CUDA, Apple MPS or
CPU (`--device auto` picks the first available).

```bash
python train_mnist.py train                          # about 30 min on an Apple M-series GPU; minutes on a DSMLP GPU
python train_mnist.py eval --out figures/mnist       # 5,000 samples per w
```

Training settings: 30 epochs, batch 256, AdamW at 2e-4 with warmup and cosine
decay, EMA 0.999, clean-image (x1) MSE on the lab's cosine path, seed 0. The
checkpoint is saved in the same format as the course's generator.pt.
Evaluation: 16 DDIM steps (the lab's sampler), seed 0, requests balanced over
the digits (image i asks for digit i mod 10, same starting noise for every w).
MNIST is downloaded by torchvision into downloads/mnist; the frozen classifier
evaluator.pt is fetched from the course link and checked against its SHA-256.

| Output | What it shows |
| --- | --- |
| outputs/mnist_cfg/loss.png, loss.csv | training loss per step |
| figures/mnist/cfg_grid.png | one row per w, digits 0–9, same noise in every row |
| figures/mnist/cfg_metrics.csv, .json, .png | label accuracy, FID, FID_class and pixel spread per w |

FID here uses the frozen classifier's 128 features against the 10,000 MNIST test
digits, so it cannot be compared with Inception-based FIDs. The JSON also records
the FID of 5,000 real training digits against the test set, which is the
lowest value to expect. FID_class at w = 0 is not meaningful because w = 0
ignores the requested digit. To score the course's pretrained generator with the
same code, add `--checkpoint generator.pt --out outputs/mnist_course`.

## Save your work to GitHub

Commit the source files, README, notes, and small figures after checking git status.
Keep downloaded data and large generated arrays out of Git with .gitignore.
Never commit passwords, access tokens, or private keys.

If GitHub authentication is already configured in the container:

```bash
git add README.md notes.md handoff.md concepts.md .gitignore requirements.txt cifar10_edm_lab.ipynb cifar10_parts1_3.ipynb guidance_mnist.ipynb mnist_guidance_lab.py train_mnist.py figures/
git commit -m "Run CIFAR-10 EDM parts 1-3 on DSMLP"
git push
```

If it is not configured, download the changed notebook, notes, and figures from
Jupyter and upload them to your repository through GitHub's website for this first
run. HTTPS cloning a public repository does not require authentication.

## Troubleshooting

- CUDA unavailable: verify that you launched with -g 1 and selected the container's
  Python kernel. Do not continue with CPU fallback for the GPU assignment.
- Pending container: wait for GPU scheduling; avoid launching duplicate GPU pods.
- Jupyter URL fails off campus: use the UCSD VPN or the SSH tunnel above.
- Workspace not found: run workspace --list on the login host and use your enrolled
  course workspace.
- Missing pyarrow: rerun the requirements install in the container Python environment.
  In a notebook, use %pip install pyarrow and restart the kernel.
- A download fails: rerun from the download cell; it retries/resumes downloads.
- Closing the launch SSH session stops this foreground container. Save your work
  first; release the container when finished.

## References

- [Course DSMLP lesson](https://dsc-capstone.org/2026-27/lessons/01/)
- [Q1 project requirements](https://dsc-capstone.org/2026-27/assignments/projects/q1/)
- [UCSD container launch guide](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269)
- [EDM paper](https://arxiv.org/abs/2206.00364)
- [EDM source](https://github.com/NVlabs/edm)
- [CIFAR-10 data mirror](https://huggingface.co/datasets/uoft-cs/cifar10)
- [CIFAR-10 classifiers](https://github.com/chenyaofo/pytorch-cifar-models)
