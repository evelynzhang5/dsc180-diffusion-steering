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

## Pilot results (Nov 14, 2026; all checks passed; files in [results/](results/))

These are pipeline checks on 4-8 images with a first look at the signal. None of them is a result.

**Pilot 1 (response): the linear model failed, so the comparison is not valid.**
- **Scaling.** Doubling the masked edit gave a background change far from twice the original: -0.83 vs a predicted
  -2.17, and -1.29 vs -3.16. The relative error in subject pixel change was 0.61-0.93.
- **Adding.** Combining two components was just as far off (relative error 0.61-0.80).
- **Prediction.** Predicted subject RMS was off by a factor of 2-4.
- **The target was missed.** The "optimised" combination hit the background target in only 2 of 4 scenes. In scene 1
  it sharpened the background instead (+2.07). Its lower mean subject change (0.109 vs 0.123) therefore comes at a
  different background effect and is not a like-for-like comparison.
- **Masking was not tuned either.** The masked baseline missed the target in 2 of 4 scenes (-0.85 and -0.12 against
  -0.5).
- **Probably a large noise floor.** Masked RFM at c 0.3-0.5 already moved the subject by 0.06-0.17 RMS. The sheet
  shows why: the dog and the deer change pose and position under almost any edit. A pixel measure of "subject
  change" mostly measures the subject moving, and copying the subject back cannot fix a subject that moved.

**Pilot 2 (geometry): no protection rule helped, and the one apparent win was a confound.**
- Feathering at 4 px or at sigma gave the same errors as plain rendering under every depth error.
- The ensemble looked far better under the threshold error: boundary 0.0039 vs 0.0175 at sigma 8. But its subject cut
  was drawn around the *true* threshold (35-45%), not the method's wrong one, so it never saw that error. Round 12's
  `geometry2` fixes this.
- Under noisy depth every method under-blurs the background (gap +0.5 to +1.1). Under a shifted cut every method
  except the ensemble over-blurs it (-1.2 to -1.7).

**Pilot 3 (residual): the signal is real, but only for editing a real photo through the VAE.**
- The VAE round trip of our images: 30.3 dB.
- Passing the lost detail through the same blur ("carried") beat adding it directly in the background, at every
  sigma: 39.0 vs 29.1 dB at sigma 8. Adding it directly puts sharp detail back into regions that should be blurred.
- Carried residual plus a subject copy was best overall: 40.1 dB at sigma 8.
- Caveats:
  - The truth is our own renderer.
  - Our generated images are D(z) and were never encoded, so this does not address the steering failure.
  - Source-conditioned decoders already exist (Asymmetric VQGAN).
  - Round 12's `vaecheck` measures how large the VAE loss is next to the steering damage.

**Next:** round 12 (`q2-diagnostics`) fixes the evaluation and runs idea 1's feasibility test properly:
- tuned baselines at matched background blur;
- central-difference responses around the operating point;
- generated corrections;
- a random control and a noise-floor measurement.

## Status

- Nov 14, 2026: code written; first launch had an empty pod command (the login node's home differs from the pod's),
  deleted within a minute; relaunched as `q2-direction-pilots`.
- Nov 14, 2026: ran on a GTX 1080 Ti (22:00-22:18, 18 min); all checks passed; results above. Closed.
