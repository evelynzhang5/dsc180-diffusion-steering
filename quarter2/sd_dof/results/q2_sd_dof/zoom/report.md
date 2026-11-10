# Zoom-only control for the SD 1.5 depth-of-field steering

Pairs: 25 test captions x seeds [42, 123, 256, 789] per condition, from `outputs/q2_sd_dof/`. Comparable: low-passed NCC >= 0.7 after alignment and overlap >= 50% of the frame. Zoom-only control: the unsteered image warped (bicubic) by the estimated scale and shift onto the steered image's grid; scales within 1.5% of 1 count as no zoom and become a whole-pixel shift (bicubic resampling at a scale of 1.004 alone costs about 0.19 in log Laplacian variance). Changes are in log Laplacian variance; intervals bootstrap over captions, then pairs within caption.

## Comparable pairs

| condition | NCC >= 0.5 | NCC >= 0.6 | NCC >= 0.7 | NCC >= 0.8 | median scale (all) | median scale (comparable) | comparable with zoom > 1.5% |
|---|---|---|---|---|---|---|---|
| rfm | 92 / 100 | 87 / 100 | 78 / 100 | 55 / 100 | 1.00 | 1.00 | 7 / 78 |
| meandiff | 95 / 100 | 95 / 100 | 87 / 100 | 72 / 100 | 1.00 | 1.00 | 9 / 87 |

## Sharpness changes (log Laplacian variance)

Rows 1-2: the original measure (whole image, each image's own depth mask). Rows 3-5: comparable pairs, same regions and resolution; row 3 = row 4 + row 5.

| condition | measure | pairs | background | foreground | FG/BG ratio |
|---|---|---|---|---|---|
| rfm | original, all pairs | 100 | -0.390 (-0.617, -0.149) | -0.370 (-0.551, -0.197) | +0.020 (-0.216, +0.245) |
| rfm | original, comparable | 78 | -0.335 (-0.572, -0.099) | -0.356 (-0.539, -0.182) | -0.021 (-0.255, +0.207) |
| rfm | steered vs unsteered, same region | 78 | -0.267 (-0.490, -0.037) | -0.342 (-0.486, -0.201) | -0.075 (-0.260, +0.111) |
| rfm | zoom-only control vs unsteered | 78 | -0.021 (-0.045, -0.003) | -0.023 (-0.046, -0.003) | -0.001 (-0.008, +0.004) |
| rfm | **steered vs zoom-only (beyond zoom)** | 78 | -0.246 (-0.473, -0.014) | -0.319 (-0.467, -0.177) | -0.074 (-0.258, +0.113) |
| rfm | **steered vs zoom-only (beyond zoom)** (NCC >= 0.5) | 92 | -0.270 (-0.477, -0.051) | -0.315 (-0.452, -0.180) | -0.045 (-0.219, +0.118) |
| meandiff | original, all pairs | 100 | -0.368 (-0.612, -0.141) | -0.503 (-0.663, -0.343) | -0.136 (-0.416, +0.192) |
| meandiff | original, comparable | 87 | -0.276 (-0.452, -0.075) | -0.537 (-0.712, -0.372) | -0.261 (-0.499, -0.061) |
| meandiff | steered vs unsteered, same region | 87 | -0.273 (-0.437, -0.096) | -0.549 (-0.696, -0.404) | -0.277 (-0.434, -0.126) |
| meandiff | zoom-only control vs unsteered | 87 | -0.025 (-0.053, -0.004) | -0.027 (-0.056, -0.005) | -0.002 (-0.009, +0.004) |
| meandiff | **steered vs zoom-only (beyond zoom)** | 87 | -0.247 (-0.416, -0.063) | -0.523 (-0.664, -0.375) | -0.275 (-0.433, -0.126) |
| meandiff | **steered vs zoom-only (beyond zoom)** (NCC >= 0.5) | 95 | -0.248 (-0.410, -0.072) | -0.500 (-0.647, -0.363) | -0.252 (-0.405, -0.099) |

## Decision rule (from the beyond-zoom row, NCC >= 0.7)

Fewer than 25 comparable pairs: inconclusive. Background interval below 0 and foreground interval not below 0 and ratio interval above 0: background blur. Any region below 0: image softness. Otherwise: explained by zoom.

- rfm: extra softening beyond zoom across the image: proceed, calling the target image softness
- meandiff: extra softening beyond zoom across the image: proceed, calling the target image softness

`pairs.png` (grayscale, as measured): columns unsteered, zoom-only control, steered. Rows: per condition 3 random comparable pairs and the least comparable pair.

