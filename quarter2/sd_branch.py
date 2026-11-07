"""Stable Diffusion 1.5 baselines and branched continuations for the Quarter 2 smoke test.

    python sd_branch.py baseline --prompts prompts.json --seeds 0-7 --out RUN
    python sd_branch.py branch   --prompts prompts.json --run RUN --actions global:12,targeted:5

baseline: 30-step deterministic DDIM with CFG 7.5. Saves the final image and the latents after steps 6, 15
          and 24 (step k = after k of the 30 updates).
branch:   resumes from each saved latent, applies one 3-step pulse, and finishes the trajectory.
          none:0       continue unchanged (measures numerical noise in resuming)
          global:W     CFG weight W instead of 7.5 for 3 steps
          targeted:L   add L * (eps(prompt) - eps(counterfactual prompt)) for 3 steps; the counterfactual
                       swaps "left" and "right"
Every branch starts from the baseline's saved latent, never from another branch. Records are appended to
RUN/baseline.jsonl and RUN/branches.jsonl, so an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

MODEL = os.environ.get("SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")  # override only for tests
STEPS, CFG, PULSE = 30, 7.5, 3
CHECKPOINTS = (6, 15, 24)


def parse_seeds(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def counterfactual(prompt):
    swap = {"left": "right", "right": "left"}
    return " ".join(swap.get(w, w) for w in prompt.split(" "))


def load_pipe(device):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(STEPS)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


@torch.no_grad()
def embed(pipe, texts):
    ids = pipe.tokenizer(texts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids
    return pipe.text_encoder(ids.to(pipe.device))[0]


@torch.no_grad()
def denoise(pipe, z, start, cond, uncond, cf=None, action=("none", 0.0)):
    """Run updates start+1 .. STEPS on latents z (n, 4, 64, 64 for 512x512 images).

    Returns final latents, latents saved at CHECKPOINTS, the pulse's size relative to the normal CFG noise
    prediction (mean over pulse steps, per image), and U-Net evaluations per image."""
    kind, strength = action
    n, saved, rel, evals = len(z), {}, [], 0
    for i in range(start, STEPS):
        t = pipe.scheduler.timesteps[i]
        pulse = kind != "none" and i < start + PULSE
        embs = [uncond, cond] + ([cf] if pulse and kind == "targeted" else [])
        out = pipe.unet(torch.cat([z] * len(embs)), t,
                        encoder_hidden_states=torch.cat(embs).repeat_interleave(n, 0)).sample.chunk(len(embs))
        eps = out[0] + CFG * (out[1] - out[0])
        if pulse:
            extra = (strength - CFG) * (out[1] - out[0]) if kind == "global" else strength * (out[1] - out[2])
            rel.append(extra.flatten(1).norm(dim=1) / eps.flatten(1).norm(dim=1))
            eps = eps + extra
        evals += len(embs)
        z = pipe.scheduler.step(eps, t, z).prev_sample
        if i + 1 in CHECKPOINTS:
            saved[i + 1] = z.clone()
    rel = torch.stack(rel).mean(0).tolist() if rel else [0.0] * n
    return z, saved, rel, evals


@torch.no_grad()
def save_image(pipe, z, path):
    x = pipe.vae.decode(z[None] / pipe.vae.config.scaling_factor).sample[0]
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(x).save(path, quality=95)


def done_keys(path):
    return {json.loads(l)["key"] for l in open(path)} if path.exists() else set()


def chunks(xs, k):
    return [xs[i:i + k] for i in range(0, len(xs), k)]


def write_run_info(out, pipe, args):
    info = {"model": MODEL, "steps": STEPS, "cfg": CFG, "pulse_steps": PULSE, "checkpoints": CHECKPOINTS,
            "scheduler": "DDIM, eta 0", "dtype": "float32", "args": vars(args),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__}
    try:
        import diffusers
        import huggingface_hub
        info["diffusers"] = diffusers.__version__
        info["model_revision"] = huggingface_hub.model_info(MODEL).sha
    except Exception as e:  # offline: the revision is optional
        info["model_revision_error"] = str(e)
    (out / f"run_info_{args.cmd}.json").write_text(json.dumps(info, indent=2))


def cmd_baseline(args):
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "latents").mkdir(exist_ok=True)
    pipe = load_pipe(args.device)
    write_run_info(out, pipe, args)
    uncond, done = embed(pipe, [""]), done_keys(out / "baseline.jsonl")
    h = pipe.unet.config.sample_size  # 64 latent pixels = 512 image pixels
    with open(out / "baseline.jsonl", "a") as log:
        for p in json.load(open(args.prompts)):
            cond = embed(pipe, [p["prompt"]])
            todo = [s for s in parse_seeds(args.seeds) if f"{p['id']}_s{s}" not in done]
            for group in chunks(todo, args.chunk):
                z0 = torch.cat([torch.randn(1, 4, h, h, generator=torch.Generator().manual_seed(s))
                                for s in group]).to(args.device) * pipe.scheduler.init_noise_sigma
                t0 = time.time()
                z, saved, _, evals = denoise(pipe, z0, 0, cond, uncond)
                for j, s in enumerate(group):
                    key = f"{p['id']}_s{s}"
                    save_image(pipe, z[j], out / "images" / f"{key}.jpg")
                    torch.save({k: v[j].cpu() for k, v in saved.items()}, out / "latents" / f"{key}.pt")
                    log.write(json.dumps({"key": key, "prompt_id": p["id"], "seed": s,
                                          "image": f"images/{key}.jpg", "unet_evals": evals,
                                          "seconds": round((time.time() - t0) / len(group), 2)}) + "\n")
                log.flush()
                print(f"baseline {p['id']} seeds {group}: {(time.time() - t0) / len(group):.1f} s/image", flush=True)


def cmd_branch(args):
    run = Path(args.run)
    (run / "branches").mkdir(exist_ok=True)
    actions = [(k, float(v)) for k, v in (a.split(":") for a in args.actions.split(","))]
    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    base = [json.loads(l) for l in open(run / "baseline.jsonl")]
    pipe = load_pipe(args.device)
    write_run_info(run, pipe, args)
    uncond, done = embed(pipe, [""]), done_keys(run / "branches.jsonl")
    with open(run / "branches.jsonl", "a") as log:
        for p in json.load(open(args.prompts)):
            seeds = sorted(r["seed"] for r in base if r["prompt_id"] == p["id"])
            if not seeds:
                continue
            cond, cf = embed(pipe, [p["prompt"]]), embed(pipe, [counterfactual(p["prompt"])])
            latents = {s: torch.load(run / "latents" / f"{p['id']}_s{s}.pt") for s in seeds}
            for k in checkpoints:
                for kind, strength in actions:
                    name = f"{kind}{strength:g}"
                    todo = [s for s in seeds if f"{p['id']}_s{s}_k{k}_{name}" not in done]
                    for group in chunks(todo, args.chunk):
                        z = torch.stack([latents[s][k] for s in group]).to(args.device)
                        t0 = time.time()
                        zf, _, rel, evals = denoise(pipe, z, k, cond, uncond, cf, (kind, strength))
                        for j, s in enumerate(group):
                            key = f"{p['id']}_s{s}_k{k}_{name}"
                            save_image(pipe, zf[j], run / "branches" / f"{key}.jpg")
                            log.write(json.dumps({"key": key, "base_key": f"{p['id']}_s{s}", "prompt_id": p["id"],
                                                  "seed": s, "checkpoint": k, "action": kind,
                                                  "strength": strength, "rel_size": round(rel[j], 4),
                                                  "unet_evals": evals, "image": f"branches/{key}.jpg",
                                                  "seconds": round((time.time() - t0) / len(group), 2)}) + "\n")
                        log.flush()
            print(f"branch {p['id']} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--prompts", required=True)
    b.add_argument("--seeds", default="0-7")
    b.add_argument("--out", required=True)
    r = sub.add_parser("branch")
    r.add_argument("--prompts", required=True)
    r.add_argument("--run", required=True)
    r.add_argument("--actions", required=True)
    r.add_argument("--checkpoints", default=",".join(map(str, CHECKPOINTS)))
    for p in (b, r):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--chunk", type=int, default=4, help="images per U-Net batch")
    args = ap.parse_args()
    {"baseline": cmd_baseline, "branch": cmd_branch}[args.cmd](args)


if __name__ == "__main__":
    main()
