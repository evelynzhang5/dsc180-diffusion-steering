# Diffusion steering capstone — CIFAR-10 EDM

Study unconditional generation, class conditioning, and classifier-free guidance
(CFG) with the pretrained NVIDIA EDM CIFAR-10 models. This starter reproduces
parts 1–3 of the supplied capstone lab. Record observations in notes.md.

**Status:** starter files prepared; the GPU experiment has not yet been run.
Replace this sentence with your actual run status once verified. Later
evaluation tables and training experiments are not implemented in this starter.

## Files

- cifar10_edm_lab.ipynb: the full supplied lab, unchanged (parts 1–6).
- cifar10_parts1_3.ipynb: parts 1–3 only, plus a CUDA check, figure export, and the part 3 trajectories.
- requirements.txt: packages added on top of the DSMLP scipy-ml image.
- notes.md: run log, predictions, and observations.
- handoff.md: deliverables checklist, project direction, open questions.
- concepts.md: background theory (study notes).
- figures/parts1_3/: created by a successful run.
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

There are no numeric results tables in parts 1–3. When implementing parts 4–6,
add each table's path, settings, and executable reproduction command here.
Do not describe unrun experiments as completed results.

## Save your work to GitHub

Commit the source files, README, notes, and small figures after checking git status.
Keep downloaded data and large generated arrays out of Git with .gitignore.
Never commit passwords, access tokens, or private keys.

If GitHub authentication is already configured in the container:

```bash
git add README.md notes.md handoff.md concepts.md .gitignore requirements.txt cifar10_edm_lab.ipynb cifar10_parts1_3.ipynb figures/
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
