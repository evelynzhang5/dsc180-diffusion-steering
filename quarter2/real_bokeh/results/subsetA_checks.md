# Round 15 subset A: single-depth vs layered rendering on real EBB! pairs (diagnostic)

| check | result | detail |
|---|---|---|
| every scene ran | PASS | 60/60 |
| measures finite (outer band, all methods) | PASS |  |
| renderers differ from the input (blur applied) | info |  |
| contact sheet <= 25 images | PASS | 20 |

Error against the real f/1.8 photo (MAE x 255; LPIPS AlexNet). outer = defocused side of the in-focus boundary (where foreground colour bleeds: the layer-leakage measure); inner = in-focus side; thin = thin near structures (disparity top-hat); interior = defocused background far from boundaries. floor = the target against itself shifted 1 px (how much a 1 px misalignment alone costs). Focus and K fitted per scene with `hard`, then shared.

## thin (hair, branches, fence, foliage) (32 scenes)

| method | mae_outer | mae_inner | mae_thin | mae_interior_bg | lpips_band | lpips_all | psnr |
|---|---|---|---|---|---|---|---|
| input | 17.60 | 15.62 | 25.70 | 18.90 | 0.250 | 0.418 | 20.22 |
| gather | 17.58 | 15.16 | 22.29 | 12.99 | 0.234 | 0.229 | 23.00 |
| hard | 14.79 | 15.60 | 23.41 | 13.09 | 0.212 | 0.223 | 23.10 |
| soft | 17.92 | 15.16 | 22.48 | 13.25 | 0.252 | 0.241 | 22.79 |
| mpib | 15.51 | 15.68 | 23.11 | 13.18 | 0.220 | 0.229 | 22.97 |
| floor_shift1 | 5.24 | 8.44 | 12.30 | 2.20 | 0.028 | 0.024 | 31.32 |

Per scene, outer-band MAE at least 10% lower: mpib vs hard 0/32, mpib vs gather 13/32, soft vs hard 0/32, mpib vs soft (inpainting) 13/32; thin-structure MAE mpib vs hard 8/32.

## glass (8 scenes)

| method | mae_outer | mae_inner | mae_thin | mae_interior_bg | lpips_band | lpips_all | psnr |
|---|---|---|---|---|---|---|---|
| input | 14.74 | 12.70 | 21.82 | 13.02 | 0.164 | 0.232 | 23.79 |
| gather | 13.64 | 12.55 | 17.65 | 10.42 | 0.156 | 0.189 | 25.57 |
| hard | 13.19 | 13.17 | 18.24 | 10.45 | 0.157 | 0.187 | 25.56 |
| soft | 13.49 | 12.85 | 16.51 | 10.44 | 0.166 | 0.193 | 25.59 |
| mpib | 13.39 | 13.12 | 17.44 | 10.45 | 0.171 | 0.194 | 25.59 |
| floor_shift1 | 5.27 | 7.23 | 11.86 | 2.91 | 0.025 | 0.025 | 29.61 |

Per scene, outer-band MAE at least 10% lower: mpib vs hard 0/8, mpib vs gather 3/8, soft vs hard 0/8, mpib vs soft (inpainting) 3/8; thin-structure MAE mpib vs hard 3/8.

## control (20 scenes)

| method | mae_outer | mae_inner | mae_thin | mae_interior_bg | lpips_band | lpips_all | psnr |
|---|---|---|---|---|---|---|---|
| input | 17.96 | 12.66 | 31.84 | 19.97 | 0.183 | 0.275 | 21.09 |
| gather | 17.03 | 13.07 | 26.68 | 15.43 | 0.193 | 0.219 | 22.66 |
| hard | 15.34 | 13.66 | 26.57 | 15.49 | 0.186 | 0.223 | 22.71 |
| soft | 17.20 | 13.18 | 26.96 | 15.58 | 0.207 | 0.231 | 22.53 |
| mpib | 16.57 | 14.71 | 28.02 | 15.54 | 0.193 | 0.226 | 22.59 |
| floor_shift1 | 6.05 | 8.21 | 13.51 | 3.38 | 0.026 | 0.026 | 28.60 |

Per scene, outer-band MAE at least 10% lower: mpib vs hard 0/20, mpib vs gather 3/20, soft vs hard 0/20, mpib vs soft (inpainting) 4/20; thin-structure MAE mpib vs hard 1/20.

## Decision (frozen in README)

- Thin-structure scenes where MPIB's outer-band error is >= 10% below the better single-depth renderer: 0/32.
- Mean outer-band gain of MPIB over the better single-depth renderer: -0.79 (MAE x 255); 1 px shift floor 5.24.
- MPIB residual: outer band / background interior = 1.32.
- **Layered rendering does not clearly beat single-depth rendering at boundaries on these real pairs: the 2.5D failure is not the dominant error here (depth error and misalignment are candidates).**

`subsetA_examples.png` columns: input, real target, gather, hard, mpib (one hair, branches, fence, glass scene).

Total time 230 s.
