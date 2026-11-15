# Q2 round 11: smoke pilots for three candidate directions (SD 1.5 depth of field)

Branch `q2-direction-pilots`. Started Nov 14, 2026. The index of all Q2 rounds is in [../README.md](../README.md).
Pipeline checks with a first look at the signal (4-8 images each); not results. Each pilot, if it shows signal, gets
its own pre-registered round.

**Why.** Round 10's smoke test (`q2-sd-dof-masked`) showed that zeroing the RFM edit on the subject at one layer does
not protect the subject. Pixel change was about equal inside and outside it, and DINOv2 similarity was 0.69: later
layers spread the edit. Three candidate directions respond to this:

1. **Model how an edit spreads, then choose an edit that protects the subject.** Measure each edit component's
   effect on the *final* image, then pick a combination that reaches the requested background blur while keeping
   the subject's change small. The first decisive experiment is the expensive, direct version (measured responses,
   no learned predictor): can it beat tuned masking at equal blur?
2. **Request-dependent protection under uncertain geometry.** When depth is wrong, which protection rule keeps the
   subject without giving up the background blur?
3. **Carry source detail through the optical edit.** Restore detail lost by the VAE by passing the residual
   r = I - VAE(I) through the same blur as the edit.

Idea 4 (choosing calibration examples by informativeness) is deferred: it selects training data for idea 1's
predictor, so it has nothing to select for until idea 1 shows a signal.

**Related work that bounds novelty** (from the reading notes, not yet checked against the papers):
- LOCO Edit: Jacobian-based local editing.
- AcFlow: learned, spatially varying activation updates.
- BokehDiff: optical footprints and occlusion in attention; reports VAE detail loss.
- FlashReFocus: lightweight re-rendering, with source-detail reinjection.
- AnyBokeh: estimates the current blur state.
- Vid2Bokeh: light-field supervision.
- Asymmetric VQGAN: a conditional decoder.

## Pilots (`pilots.py`, one pod: `dsmlp_pilots.sh`)

| pilot | setup | pass/fail checks | signal to look for |
|---|---|---|---|
| `response` (idea 1) | 4 scenes, seed 42, s 4. Components: RFM x background mask, RFM x foreground mask, round 10's paired direction x background, 2nd eigenvector x background. Each at c = 1; plus scale and additivity checks. Linear model: background sharpness a.c; subject pixel change sum c_k Delta_k. Choose min subject change s.t. a.c = -0.5, generate it, compare with masked RFM at the c predicted to reach -0.5 | finite measures; masked RFM blurs the background at c = 1 | linearity errors near 0 (else a linear predictor is hopeless); at matched background change, lower subject pixel change and higher DINOv2 than masked RFM |
| `geometry` (idea 2) | round 10's 8 images; truth = our depth renderer with the estimated depth; depth errors: correlated noise, smoothed depth, a shifted subject cut. Methods: plain render, 4 px feather, feather growing with blur, 8-sample depth ensemble | finite; errors are non-zero | lower boundary error without a background blur gap (protecting by not blurring is a failure) |
| `residual` (idea 3) | same images; truth = rendered blur; simulated perfect latent edit = VAE round trip of the truth. Methods: VAE alone, + residual, + residual carried through the blur, subject copy, subject copy + carried residual | VAE round trip 20-40 dB; residual restores the unedited subject | carried residual beats direct residual in the background and boundary, and beats a plain subject copy |

**Known limits of the pilots:**
- **Pilots 2 and 3 use our renderer as the truth.** It keeps the subject exactly and ignores occlusion, so any
  subject-copy method is perfect inside the subject by construction. Only the background and boundary columns test
  anything.
- **Pilot 1's components are hand-picked, and its linear model is the simplest possible one.**

## Status

- Nov 14, 2026: code written; first launch had an empty pod command (the login node's home differs from the pod's),
  deleted within a minute; relaunched as `q2-direction-pilots` (waiting for a GPU).
