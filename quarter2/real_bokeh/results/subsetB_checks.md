# Round 15 subset B: additive glass reflections (10 synthetic scenes from EBB! photos; diagnostic)

| check | result | detail |
|---|---|---|
| all scenes ran | PASS | 10 |
| targets differ (focus actually moves) | PASS |  |
| contact sheet <= 25 images | PASS | 10 |

I = transmitted scene (dimmed 30% inside the window) + reflection (strength 0.25-0.45) inside a window. Exact targets: focus on the scene (reflection blurred, K 24) and focus on the reflection (scene blurred). gather / hard: focus and K fitted to each target on the window (oracle settings); mpib: hard's settings. The additive oracle renders T and R separately with their own blur, so it is exact by construction: it shows what a two-depth additive representation could reach, not a method. MAE x 255 inside the window.

| target | input | gather | hard | mpib | additive_oracle |
|---|---|---|---|---|---|
| focus_scene | 10.66 | 11.57 | 11.50 | 11.71 | 0.00 |
| focus_reflection | 21.02 | 11.86 | 12.53 | 12.20 | 0.00 |

## Decision (frozen in README)

- Best single-image renderer as a fraction of doing nothing (input): focus_scene 1.07, focus_reflection 0.56
- **No single-depth or layered renderer can produce the two focus targets from one image: they barely beat the unedited input (>= 0.7 of its error) on at least one target.**
- Real data with a focus shift between reflection and scene is still needed: EBB! has one fixed focus per pair. Its glass scenes are scored in subset A.

`subsetB_examples.png` rows: focus on scene, focus on reflection; columns: input, target, gather, hard, mpib.
