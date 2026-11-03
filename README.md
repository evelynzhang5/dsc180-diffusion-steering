# Diffusion steering capstone — CIFAR-10 EDM

Study unconditional generation, class conditioning, and classifier-free guidance
(CFG) with the pretrained NVIDIA EDM CIFAR-10 models. This starter reproduces
parts 1–3 of the supplied capstone lab. Record observations in notes.md.


## Files

- cifar10_edm_lab.ipynb: the full supplied lab (parts 1–6), saved with its outputs from the DSMLP run on November 1, 2026 (GTX 1080 Ti).
- cifar10_parts1_3.ipynb: parts 1–3 only, plus a CUDA check, figure export, and the part 3 trajectories, saved with its outputs from the same run.
- mnist_guidance_lab.py: helpers from the course's MNIST guidance lab (U-Net, CFG sampler, frozen digit classifier), used by train_mnist.py.
- train_mnist.py: the MNIST CFG network deliverable: trains the lab's U-Net from scratch and measures CFG.
- requirements.txt: packages added on top of the DSMLP scipy-ml image.
- notes.md: run log, predictions, and observations.
- report.md: checkpoint report draft (title, abstract, introduction, preliminary results).
- report_tex/: the checkpoint report in the course's LaTeX template (report.tex, reference.bib, figure/, compiled report.pdf). Build with `cd report_tex && latexmk -xelatex report.tex`, or upload the folder to Overleaf and compile with XeLaTeX.
- plan.md: where the project stands, next steps, open questions.
- scripts/dsmlp_run_all.sh: the DSMLP batch job that ran both CIFAR-10 notebooks and EDM training.
- scripts/dsmlp_train_edm.sh: 12-hour DSMLP job that continues EDM training, then measures snapshot FID.
- scripts/edm_snapshot_fid.py, scripts/edm_combine_runs.py: snapshot FID (GPU) and the two-run loss/FID plot.
- figures/parts1_3/: parts 1–3 figures, exported by cifar10_parts1_3.ipynb.
- figures/lab/: parts 4–6 figures, extracted from the outputs saved in cifar10_edm_lab.ipynb.
- figures/edm_train/: loss and snapshot FID of the full-data EDM training runs.
- results/: small result files: the part 5 sweep, both EDM training runs' stats, logs and snapshot FID, and the batch job logs.


## Contribution statement (human and AI)

**Evelyn Zhang** chose the project and its research question within the course's
diffusion-steering topic, set up the GitHub repository and DSMLP access, supplied
the course materials (assignment, lab notebooks, LaTeX template), brainstormed directions for online research, set the timeline/to-do list, decided which GPU to use and when to stop training, monitored the logs, reviewed the code, verified the visualizations and results, and wrote rough draft and design for report.

**AI (Claude Code, Anthropic, model Claude Opus 5.5)** was used primarily for:
implementing the coding (the DSMLP batch scripts in scripts/, train_mnist.py, the snapshot-FID and plotting scripts); running and monitoring the experiments (the CIFAR-10 notebooks and EDM training on DSMLP, MNIST training locally); debugging; supplementing documentation (this README, notes.md, plan.md);
literature search and reference checking; drafting and editing the report and
porting it to LaTeX; and explaining the methods to the author. 

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
python -m jupyter nbconvert --to notebook --execute cifar10_parts1_3.ipynb --output cifar10_parts1_3.ipynb --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
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
nbconvert (writing the results to *.executed.ipynb copies next to the originals;
these copies are not kept in Git, and the committed notebooks already hold the
same outputs), and spends the rest of the pod's 6-hour limit on EDM's train.py (the
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
vs. real test images 21.3. Label accuracy of one-network CFG (100 samples per w; the grids show 10):
100 images w = 0, 1, 2, 4: 8%, 93%, 100%, 100%; 1,000 images: 6%, 49%, 76%, 89%.

| Figure | What it shows |
| --- | --- |
| figures/lab/07_fid_2d.png | FID in two dimensions (part 4) |
| figures/lab/08_cfg_sweep_acc_fid.png | part 5: accuracy, FID, FID_class against w |
| figures/lab/09_part6_loss_fid_snapshots.png | part 6: training loss and FID across snapshots |
| figures/lab/10_part6_samples_n100.png, 11_part6_samples_n1000.png | unconditional samples from the same noise at each snapshot |
| figures/lab/12_part6_nearest_n100.png, 13_part6_nearest_n1000.png | samples next to their nearest training images |
| figures/lab/14_part6_cfg_grid_n100.png, 15_part6_cfg_grid_n1000.png | one-network CFG grids, w = 0, 1, 2, 4 |

The figures/lab images are the embedded outputs of the full lab notebook,
extracted in cell order from cifar10_edm_lab.ipynb.

### Full-data EDM training (encouraged/optional)

EDM's train.py with `--cond=0 --arch=ddpmpp --batch=512 --batch-gpu=64 --duration=10 --snap=10`
on all 50,000 images, stopped by the time limit after 1,205 kimg (24 ticks,
3.3 hours, about 9.7 s per kimg). Loss fell from 1.02 at the start to 0.196.
Snapshots were saved at 0, 502 and 1,004 kimg (kept on DSMLP; not in Git).
Training stats: results/edm_train/stats.jsonl, training_options.json, log.txt;
loss curve: figures/edm_train/loss.png. This run cannot be resumed exactly:
train.py saves its optimizer state (training-state-*.pt) only every `--dump`
ticks, 500 by default, and the run stopped at tick 24. To keep training from
the last weights, start a new run with `--transfer=<run folder>/network-snapshot-001004.pkl`
(fresh optimizer, image count restarts at 0), and add `--dump=10` so that later
runs can be continued with `--resume`.

FID across the snapshots (notebook cell 39, 1,000 unconditional samples each,
seeds from 3000) comes from scripts/edm_snapshot_fid.py. It runs the lab cells
that cell 39 needs (1, 3, 5, 7, 21), then cell 39 with RUN_DIR set to the run
folder above, and writes figures/edm_train/loss_fid.png and
results/edm_train/snapshot_fid.json. Run it on a GPU from the repository root:

```bash
python scripts/edm_snapshot_fid.py
```

| million training images | 0 | 0.502 | 1.004 |
| --- | --- | --- | --- |
| FID (1,000 images) | 412.4 | 105.5 | 76.5 |

Run 2 continued from the 1,004-kimg snapshot in a 12-hour batch pod
(`K8S_TIMEOUT_SECONDS=43200`) with scripts/dsmlp_train_edm.sh. The script uses
`--transfer` the first time and `--resume` on the newest training-state-*.pt
after that; it stops training 75 minutes before the pod limit and then runs
edm_snapshot_fid.py with the tag run2:

```bash
bash scripts/dsmlp_train_edm.sh 12 run2
```

It trained for 10.6 hours (76 ticks, 3,764 kimg, about 10.2 s per kimg) with
`--dump=10`, so training-state-003512.pt on DSMLP can be continued exactly.
Training stats: results/edm_train_run2/; job log:
results/logs/dsmlp_train_run2_20261102.log; FID:
results/edm_train/snapshot_fid_run2.json and figures/edm_train/loss_fid_run2.png
(run 2's own image count). scripts/edm_combine_runs.py puts both runs on one
axis, shifting run 2 by the 1.004 million images of its starting snapshot, and
writes figures/edm_train/loss_fid_combined.png (report Figure 16):

| million training images (both runs) | 1.004 | 1.506 | 2.008 | 2.509 | 3.011 | 3.513 | 4.015 | 4.516 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FID (1,000 images) | 76.5 | 73.1 | 67.7 | 59.9 | 56.2 | 54.2 | 52.2 | 50.2 |

## MNIST CFG network

The checkpoint asks for the lab's U-Net trained from scratch on the 60,000 MNIST
training digits, with the label replaced by 10 ("no label") for a random 10% of
examples, then sampled with w = 0, 1, 2, 4. train_mnist.py does both steps. It
needs PyTorch, torchvision, SciPy and matplotlib, and runs on CUDA, Apple MPS or
CPU (`--device auto` picks the first available).

```bash
python train_mnist.py train                          # about 22 min on an Apple M-series GPU; minutes on a DSMLP GPU
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
| figures/mnist/loss.png | training loss per step (`train` writes it, with loss.csv, to outputs/mnist_cfg/; copy it with `cp outputs/mnist_cfg/loss.png figures/mnist/`) |
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
git add README.md notes.md plan.md report.md .gitignore requirements.txt cifar10_edm_lab.ipynb cifar10_parts1_3.ipynb mnist_guidance_lab.py train_mnist.py scripts/ figures/ results/ report_tex/
git commit -m "Run CIFAR-10 EDM parts 1-3 on DSMLP"
git push
```

If it is not configured, download the changed notebook, notes, and figures from
Jupyter and upload them to your repository through GitHub's website for this first
run. HTTPS cloning a public repository does not require authentication.


## References

- [Course DSMLP lesson](https://dsc-capstone.org/2026-27/lessons/01/)
- [Q1 project requirements](https://dsc-capstone.org/2026-27/assignments/projects/q1/)
- [UCSD container launch guide](https://support.ucsd.edu/services?id=kb_article_view&sysparm_article=KB0032269)
- [EDM paper](https://arxiv.org/abs/2206.00364)
- [EDM source](https://github.com/NVlabs/edm)
- [CIFAR-10 data mirror](https://huggingface.co/datasets/uoft-cs/cifar10)
- [CIFAR-10 classifiers](https://github.com/chenyaofo/pytorch-cifar-models)
