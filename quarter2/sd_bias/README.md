# Q2 round 3: reference-set bias on SD 1.5 (blur direction vs. people and framing), closed

Branch `q2-sd-bias`. Nov 8-9, 2026. Question: does a "background blur" direction learned from a portrait-biased
reference set add people and close-up framing to other prompts, at the same blur? The index of all Q2 rounds is in
[../README.md](../README.md).

**Results in git.** The scripts write to `outputs/` (not in git). Copies of the small result files (reports, figures,
logs, small tables; files over 1 MB left out, large PNGs converted to JPEG) are in `results/`, under the same folder
names: `outputs/q2_sd_bias/` -> `results/q2_sd_bias/`, `outputs/q2_sd_audit/` -> `results/q2_sd_audit/`. Reference
and evaluation images are on DSMLP only, under
`/dsmlp/workspaces-fs05/DSC180A_FA26_001/home/yuz271/dsc180-diffusion-steering/outputs/`.

## Setup (`sd_bias.py`)

- References: SD 1.5 images of 8 person and 8 object subjects, prompted for shallow depth of field ("blur") or deep
  focus ("sharp"), 8 seeds each. Three reference sets of 64 blur vs. 64 sharp images: **biased** (blur images 7/8
  people, sharp images 7/8 objects), **opposed** (the reverse) and **balanced** (half and half).
- Direction: mean mid-block activation (1280 x 8 x 8) of blur minus sharp references at each steering step, unit
  norm; edit h <- h + a |h| d during DDIM steps 2-11 of 30. Control: a random direction.
- Strength: balanced direction swept on dev seeds; biased and opposed tuned to the same blur; test on 12 held-out
  object prompts and 4 person prompts x seeds 2-9, paired against the unedited image.

## Results

**The edits made no measurable blur.** On the target meter (CLIP blur score) the balanced direction's dev sweep
reached +0.016 at a = 1 (a = 0.25 and 0.5 went the wrong way), and on the test prompts no direction changed CLIP blur
or background sharpness beyond its interval (`results/q2_sd_bias/report.md`). The matched-blur comparison therefore
compares directions at essentially zero blur.

**The biased direction enlarged people anyway.** On person prompts, framing (subject box share) rose under the
biased direction against the balanced one by +0.175 (95% interval +0.087 to +0.293; same prompt and seed), in 4 of 4
person prompts, with no change in subject detection score, so it is not a detector artefact
(`results/q2_sd_audit/report.md`, part A). A blinded AI audit of 12 biased/balanced pairs (`blind_pairs.png`, key in
`blind_key.json`) chose the biased image as showing the person larger in 9 of 11 decided pairs.

**The meters work** (part C): background-only blur lowers background sharpness and raises CLIP blur with the radius;
zoom without blur moves framing. CLIP blur also moves somewhat under zoom (-0.10 to -0.16), so it is not
zoom-proof. **The model can make the blur** (part D): prompting "shallow depth of field, bokeh" vs. "deep focus" on
the same seeds raises CLIP blur (+0.59) and lowers background sharpness, but also changes framing (-0.11).

**Conclusion.** The reference-set bias propagated to framing, but the direction did not deliver its intended
effect, so this is not a matched-target comparison. The next round (`q2-sd-dof`) reproduced a documented working
blur setup first.

## Files and how to run

| File | What it does |
| --- | --- |
| `sd_bias.py` | References, three reference sets, mid-block directions, strength calibration, test-set report |
| `sd_audit.py` | Follow-up: paired re-analysis, blinded pairs, meter validation on controlled edits, prompt positive control |
| `dsmlp_sd_bias.sh`, `dsmlp_sd_audit.sh` | Run each on a DSMLP GPU pod (smoke test first) |
| `requirements.txt` | Python packages on top of the DSMLP scipy-ml image |

From the repository root, on a GPU pod: `bash quarter2/sd_bias/dsmlp_sd_bias.sh`, then
`bash quarter2/sd_bias/dsmlp_sd_audit.sh`.
