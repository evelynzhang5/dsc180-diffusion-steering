# SD 1.5 depth-of-field steering: the paper's setup reproduced, then compared

Direction data: 1200 photos, label names ['0', '1'], shallow = label 0 (mean log Laplacian variance by label {"0": 4.36, "1": 5.19}).
RFM held-out AUC 0.964; mean-difference held-out AUC 0.948; cosine between the two directions +0.727.
Steering window: DDIM steps 0-49 of 50.
Test: 25 captions x seeds [42, 123, 256, 789].

## 1. Paper's numbers vs ours

| | background sharpness (mean Laplacian variance) | change | FG / BG ratio (mean of images) |
|---|---|---|---|
| paper, unsteered -> steered | 1085.03 -> 785.89 | -27.6% | 2.247 -> 2.585 |
| ours, rfm | 1069.4 -> 688.5 | -35.6% | 2.565 -> 2.669 |
| ours, meandiff | 1069.4 -> 690.5 | -35.4% | 2.565 -> 1.951 |
| ours, prompt | 1069.4 -> 894.8 | -16.3% | 2.565 -> 4.461 |

## 2. Paired changes vs. unedited (same caption and seed)

Intervals: bootstrap over captions, then seeds within caption. Effect: background blur. Selectivity: the foreground and the caption should not change. Pixel change is mean |edited - unedited| / 255 inside the unedited image's foreground and background.

| condition | s | log BG sharpness | log FG sharpness | log FG/BG ratio | CLIP blur score | CLIP caption score | pixel change, FG | pixel change, BG |
|---|---|---|---|---|---|---|---|---|
| rfm | 4 | -0.390 (-0.612, -0.149) | -0.370 (-0.545, -0.206) | +0.020 (-0.223, +0.240) | +0.746 (+0.458, +1.055) | +0.165 (-0.694, +1.111) | 0.127 | 0.122 |
| meandiff | 2 | -0.368 (-0.594, -0.148) | -0.503 (-0.669, -0.334) | -0.136 (-0.406, +0.170) | +0.717 (+0.415, +1.019) | +0.031 (-0.750, +0.812) | 0.109 | 0.105 |
| prompt | 4 | -0.219 (-0.457, +0.047) | -0.096 (-0.257, +0.051) | +0.123 (-0.157, +0.380) | +0.505 (+0.262, +0.777) | -0.593 (-1.273, +0.100) | 0.112 | 0.109 |

Dev target (RFM's change in log background sharpness on dev captions): -0.503; picked s: meandiff 2, prompt 4.

`examples.png`: columns none, rfm, meandiff, prompt; rows test captions 0-5, seed 42.

Captions: 0: A child holding a flowered umbrella and petting a yak; 1: A narrow kitchen filled with appliances and cooking utensils; 2: A little girl holding a kitten next to a blue fence; 3: A toilet sitting in a bathroom next to a sink; 4: There are two sinks next to two mirrors; 5: A woman rides a bicycle on a road next to the median

