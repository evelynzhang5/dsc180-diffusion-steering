# Meter checks on 1,000 real CIFAR-10 training images

Change in each meter, in SDs of the meter over the 1,000 originals. A meter is useful if it responds clearly to the edit it is named for and its responses to unrelated edits are understood.

| edit | brightness | contrast | saturation | colorfulness | blueness | greenness | redness | border_clutter | figure_ground | sharpness |
|---|---|---|---|---|---|---|---|---|---|---|
| brighter (+0.1) | +0.79 | -0.07 | -0.44 | -0.02 | -0.00 | +0.01 | -0.00 | -0.07 | -0.04 | -0.03 |
| desaturate (halfway to grey) | +0.00 | +0.00 | -0.87 | -0.95 | +0.20 | -0.16 | -0.14 | +0.00 | -0.06 | +0.00 |
| bluer (+0.08 blue) | +0.07 | -0.01 | -0.01 | +0.01 | +0.78 | -0.92 | -0.42 | -0.01 | -0.01 | -0.00 |
| greener (+0.08 green) | +0.37 | -0.03 | +0.50 | +0.21 | -0.39 | +1.84 | -0.42 | -0.03 | -0.01 | -0.01 |
| lower contrast (x0.6 around the mean) | -0.02 | -1.24 | -0.81 | -0.76 | +0.16 | -0.13 | -0.12 | -0.99 | -0.53 | -1.17 |
| blur (Gaussian, 1 px) | -0.00 | -0.38 | -0.11 | -0.16 | +0.00 | -0.00 | +0.00 | -0.26 | -0.03 | -2.03 |
| flat background (border ring set to its mean) | -0.00 | -0.65 | -0.18 | -0.30 | -0.00 | +0.00 | +0.00 | -2.47 | +0.00 | -0.20 |

Set meters: 500 real images of all classes vs. 500 of one class (less variety; all three should fall). Ratio one-class / mixed:

| one class | feature_diversity | background_diversity | pixel_diversity |
|---|---|---|---|
| airplane | 0.16 | 1.18 | 1.01 |
| cat | 0.21 | 1.06 | 1.06 |
| deer | 0.18 | 0.89 | 0.85 |
| ship | 0.17 | 0.94 | 0.92 |

Meters: **brightness**: mean luma; **contrast**: SD of luma within the image; **saturation**: mean HSV saturation; **colorfulness**: Hasler-Suesstrunk colourfulness; **blueness**: mean B - (R+G)/2; **greenness**: mean G - (R+B)/2; **redness**: mean R - (G+B)/2; **border_clutter**: SD of luma in the 4-pixel border ring (background busyness); **figure_ground**: RGB distance between the centre 16x16 mean and the border-ring mean; **sharpness**: mean absolute Laplacian of luma; **feature_diversity**: mean pairwise cosine distance of ConvNeXt features; **background_diversity**: mean pairwise distance of border-ring mean colours; **pixel_diversity**: mean pairwise L2 distance of 8x8 downsampled images
