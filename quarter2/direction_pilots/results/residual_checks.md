# Pilot 3: carrying VAE-lost detail through the optical edit (8 images; pipeline check)

| check | result | detail |
|---|---|---|
| all PSNRs finite | PASS |  |
| VAE round trip loses detail (PSNR < 40 dB) but is sane (> 20 dB) | info | 30.3 dB |
| direct residual restores the unedited subject (subject interior PSNR up vs vae alone) | PASS |  |
| contact sheet <= 25 images | PASS | 12 |

PSNR (dB) against the rendered truth. The truth keeps the subject exactly, so `subject_copy` (with the true mask) is perfect inside the subject by construction; the background and boundary columns are the test. VAE round trip of the originals: 30.3 dB. `residual_examples.png` columns: truth, vae, direct, carried, subject_copy, subject_copy+carried (sigma 8).

| sigma | method | all | subject interior | boundary | background interior |
|---|---|---|---|---|---|
| 2 | vae | 32.31 | 33.96 | 31.58 | 31.98 |
| 2 | direct | 32.61 | 42.62 | 36.22 | 30.61 |
| 2 | carried | 36.01 | 42.62 | 36.84 | 34.49 |
| 2 | subject_copy | 34.15 | 87.86 | 35.69 | 31.98 |
| 2 | subject_copy+carried | 36.68 | 93.94 | 40.05 | 34.49 |
| 4 | vae | 33.51 | 33.73 | 31.93 | 34.82 |
| 4 | direct | 31.29 | 39.72 | 33.47 | 29.46 |
| 4 | carried | 36.51 | 39.72 | 34.74 | 36.35 |
| 4 | subject_copy | 36.60 | 144.69 | 35.82 | 34.82 |
| 4 | subject_copy+carried | 38.22 | 145.87 | 37.88 | 36.35 |
| 8 | vae | 34.09 | 33.52 | 32.51 | 37.41 |
| 8 | direct | 30.76 | 38.14 | 32.07 | 29.12 |
| 8 | carried | 36.84 | 38.14 | 34.22 | 39.03 |
| 8 | subject_copy | 38.50 | 148.13 | 36.97 | 37.41 |
| 8 | subject_copy+carried | 40.11 | 148.13 | 37.91 | 39.03 |

Total time 42 s.
