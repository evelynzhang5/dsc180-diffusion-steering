# Round 12 idea 3 check: VAE round-trip loss vs steering damage (4 generated scenes; diagnostic)

| check | result | detail |
|---|---|---|
| round trip finite and sane (15-60 dB) | PASS | 28.2, 35.4, 25.3, 25.5 |
| real photos measured | PASS | 12 |

Our images are generated as D(z): they never passed through the encoder. Residual reinjection can only repair what an encode-decode cycle loses, so its ceiling is the round-trip error. Steering damage = subject RMS between the steered image (round 10, s 4) and the unsteered one.

| scene | round-trip PSNR (dB) | round-trip subject RMS | masked-steering subject RMS | global-steering subject RMS |
|---|---|---|---|---|
| 0 | 28.2 | 0.0287 | 0.2018 | 0.1830 |
| 1 | 35.4 | 0.0133 | 0.1158 | 0.0665 |
| 3 | 25.3 | 0.0246 | 0.1635 | 0.1545 |
| 5 | 25.5 | 0.0431 | 0.2747 | 0.2784 |

Steering damage is 6.9x (masked) and 6.2x (global) the round-trip loss in the subject.

Real photos (round 4's depth-of-field set, 12): round-trip PSNR 30.4 dB vs 28.6 dB for generated images.

Total time 14 s.
