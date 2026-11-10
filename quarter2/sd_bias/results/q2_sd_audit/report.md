# SD 1.5 blur direction: follow-up audit

## A. Paired re-analysis of the saved images

Re-metered from the saved 256 px JPEGs (16 prompts x seeds 2-9). First, a check that re-metering reproduces the original report's change vs. unedited (the original measured 512 px images, so background and whole-image sharpness differ; CLIP and box-based meters should be close).

| condition | framing, person prompts | CLIP person, object prompts | CLIP blur, all |
|---|---|---|---|
| balanced | -0.044 (-0.098, -0.002) * | +0.118 (-0.162, +0.388) | -0.018 (-0.223, +0.181) |
| biased | +0.131 (+0.059, +0.223) * | +0.261 (-0.214, +0.747) | -0.011 (-0.229, +0.190) |
| opposed | -0.055 (-0.151, +0.052) | +0.118 (-0.293, +0.504) | +0.004 (-0.247, +0.232) |
| random | -0.045 (-0.095, -0.009) * | +0.059 (-0.133, +0.239) | -0.069 (-0.228, +0.059) |

Intervals: two-level bootstrap over prompts, then seeds within prompt. * = interval excludes 0. With 4 person prompts the prompt level is coarse: read the per-prompt table too.

### Direct contrasts between directions, person prompts (same prompt and seed)

| contrast | framing | subject | person_area | clip_person | person_on | clip_blur | bg_sharp | sharp |
|---|---|---|---|---|---|---|---|---|
| biased - balanced | +0.175 (+0.087, +0.293) * | -0.021 (-0.095, +0.055) | -0.091 (-0.212, +0.022) | +0.607 (+0.108, +1.297) * | -0.219 (-0.406, -0.031) * | +0.009 (-0.291, +0.330) | +0.452 (-0.636, +1.289) | -0.122 (-0.455, +0.157) |
| biased - random | +0.176 (+0.084, +0.287) * | -0.000 (-0.068, +0.079) | -0.057 (-0.178, +0.074) | +1.006 (+0.555, +1.584) * | -0.219 (-0.406, -0.062) * | +0.237 (-0.078, +0.617) | +0.717 (-0.526, +2.139) | +0.043 (-0.196, +0.291) |
| biased - opposed | +0.186 (+0.114, +0.259) * | -0.035 (-0.122, +0.055) | -0.076 (-0.240, +0.093) | +0.940 (+0.271, +1.694) * | -0.250 (-0.469, -0.031) * | +0.040 (-0.352, +0.469) | +0.550 (-0.327, +1.282) | +0.160 (-0.374, +0.579) |
| balanced - random | +0.001 (-0.049, +0.052) | +0.021 (-0.016, +0.061) | +0.034 (-0.021, +0.140) | +0.399 (+0.007, +0.801) * | +0.000 (-0.094, +0.094) | +0.228 (-0.084, +0.623) | +0.227 (-0.336, +0.708) | +0.166 (+0.034, +0.330) * |
| opposed - random | -0.010 (-0.122, +0.118) | +0.034 (-0.012, +0.090) | +0.019 (-0.099, +0.152) | +0.066 (-0.430, +0.647) | +0.031 (-0.188, +0.250) | +0.197 (-0.144, +0.584) | +0.206 (-0.363, +0.833) | -0.116 (-0.404, +0.242) |

### Direct contrasts between directions, object prompts (same prompt and seed)

| contrast | framing | subject | person_area | clip_person | person_on | clip_blur | bg_sharp | sharp |
|---|---|---|---|---|---|---|---|---|
| biased - balanced | +0.007 (-0.062, +0.068) | -0.014 (-0.040, +0.011) | -0.006 (-0.022, +0.000) | +0.143 (-0.319, +0.617) | -0.021 (-0.073, +0.000) | +0.006 (-0.205, +0.198) | +0.131 (-0.186, +0.472) | +0.090 (-0.057, +0.230) |
| biased - random | -0.008 (-0.073, +0.045) | -0.026 (-0.055, +0.003) | -0.002 (-0.021, +0.014) | +0.202 (-0.222, +0.663) | +0.000 (-0.042, +0.042) | -0.001 (-0.229, +0.207) | +0.242 (-0.186, +0.732) | +0.087 (-0.038, +0.207) |
| biased - opposed | -0.007 (-0.074, +0.055) | +0.016 (-0.027, +0.060) | -0.000 (-0.002, +0.000) | +0.143 (-0.360, +0.634) | +0.000 (+0.000, +0.000) | -0.033 (-0.262, +0.176) | -0.192 (-0.506, +0.109) | +0.077 (-0.037, +0.223) |
| balanced - random | -0.015 (-0.044, +0.015) | -0.012 (-0.032, +0.009) | +0.004 (-0.018, +0.031) | +0.059 (-0.234, +0.368) | +0.021 (-0.031, +0.104) | -0.007 (-0.203, +0.213) | +0.073 (-0.228, +0.432) | -0.002 (-0.084, +0.086) |
| opposed - random | -0.002 (-0.059, +0.043) | -0.042 (-0.088, +0.000) | -0.001 (-0.021, +0.016) | +0.059 (-0.337, +0.420) | +0.000 (-0.042, +0.042) | +0.032 (-0.214, +0.282) | +0.371 (-0.015, +0.871) | +0.010 (-0.140, +0.140) |

### Per person prompt: framing change vs. unedited (seed bootstrap)

| prompt | unedited framing | balanced | biased | opposed | random | biased - balanced |
|---|---|---|---|---|---|---|
| a firefighter | 0.524 | -0.053 (-0.133, +0.015) | +0.127 (+0.032, +0.261) * | -0.050 (-0.228, +0.117) | -0.103 (-0.200, -0.011) * | +0.181 (+0.043, +0.369) * |
| a young man | 0.809 | -0.016 (-0.052, +0.017) | +0.063 (+0.028, +0.109) * | -0.153 (-0.318, +0.001) | -0.005 (-0.027, +0.019) | +0.079 (+0.038, +0.124) * |
| an elderly woman | 0.830 | -0.053 (-0.114, -0.002) * | +0.116 (+0.007, +0.270) * | -0.040 (-0.128, +0.035) | -0.027 (-0.050, -0.006) * | +0.169 (+0.048, +0.313) * |
| a chef | 0.600 | -0.055 (-0.199, +0.069) | +0.216 (+0.069, +0.378) * | +0.021 (-0.136, +0.211) | -0.046 (-0.105, -0.007) * | +0.271 (+0.073, +0.503) * |

Biased framing above balanced in 4 of 4 person prompts.

### Is the framing change a detection artefact?

Framing is the area of the subject's OWLv2 box. If the edit changed what the detector picks (e.g. a face box instead of the whole body) the subject score or the person box would change too. Person prompts, biased - balanced:

- subject detection score: -0.021 (-0.095, +0.055)
- OWLv2 'a person' box area: -0.091 (-0.212, +0.022)
- correlation across images between the subject-box and person-box framing changes (biased - unedited): +0.12

## B. Blinded pairs

`blind_pairs.png`: 12 pairs, each the same person prompt and seed under the biased and the balanced direction, left/right shuffled. For each pair, write down which image (A or B) shows the person larger in the frame (or 'same'), then compare with `blind_key.json`. Do not open the key first.

## D. Prompt positive control: 'shallow depth of field, bokeh' minus 'deep focus', same subject and seed

16 held-out subjects x seeds 2-5, 512 px, same sampler as the steering run. Paired change (blur prompt - sharp prompt); standardized = mean / SD of the paired changes.

| meter | all prompts | standardized | person prompts | object prompts |
|---|---|---|---|---|
| bg_sharp | -0.654 (-1.134, -0.189) * | -0.47 | -0.453 (-1.784, +0.346) | -0.716 (-1.208, -0.228) * |
| clip_blur | +0.587 (+0.297, +0.865) * | +0.74 | +0.635 (+0.138, +1.061) * | +0.571 (+0.206, +0.911) * |
| sharp | -0.347 (-0.642, -0.072) * | -0.41 | +0.002 (-0.736, +0.635) | -0.464 (-0.758, -0.187) * |
| framing | -0.111 (-0.191, -0.039) * | -0.50 | -0.142 (-0.306, -0.046) * | -0.101 (-0.192, -0.020) * |
| subject | -0.013 (-0.086, +0.055) | -0.06 | +0.063 (-0.074, +0.247) | -0.038 (-0.121, +0.025) |
| clip_person | +0.001 (-0.404, +0.412) | +0.00 | +0.361 (-0.375, +1.121) | -0.119 (-0.577, +0.288) |
| person_on | +0.078 (+0.000, +0.172) | +0.29 | +0.250 (+0.000, +0.500) | +0.021 (+0.000, +0.083) |
| warmth | +0.009 (-0.003, +0.022) | +0.26 | -0.001 (-0.028, +0.019) | +0.012 (-0.001, +0.027) |

Lower background sharpness and higher CLIP blur score = blurrier. If both register this prompt contrast, the meters can see a blur difference the model can make; what else changes (framing, subject) is what a 'blur' reference set built from these prompts carries along.

## C. Meter validation: controlled edits of the deep-focus images

64 images. Change vs. the unedited image; * = 95% interval excludes 0. Background blur is applied outside the subject's OWLv2 box (feathered); zoom crops around the subject and resizes back (Lanczos), which adds no blur but magnifies, so whole-image sharpness drops a little by construction.

| edit | background sharpness | CLIP blur score | whole-image sharpness | framing | subject | CLIP person |
|---|---|---|---|---|---|---|
| background blur r=2 | -4.074 (-4.592, -3.565) * | +0.298 (+0.219, +0.385) * | -0.845 (-1.135, -0.592) * | +0.002 (-0.001, +0.007) | +0.006 (-0.001, +0.013) | +0.260 (+0.164, +0.357) * |
| background blur r=4 | -5.077 (-5.757, -4.401) * | +0.466 (+0.352, +0.579) * | -0.892 (-1.213, -0.610) * | +0.005 (-0.002, +0.015) | +0.010 (+0.002, +0.018) * | +0.277 (+0.145, +0.401) * |
| background blur r=8 | -5.410 (-6.139, -4.678) * | +0.578 (+0.431, +0.727) * | -0.902 (-1.234, -0.613) * | +0.005 (-0.003, +0.015) | +0.013 (+0.003, +0.025) * | +0.188 (+0.054, +0.317) * |
| whole-image blur r=2 | -4.122 (-4.649, -3.612) * | +1.022 (+0.891, +1.153) * | -5.145 (-5.360, -4.935) * | +0.009 (+0.001, +0.019) * | -0.048 (-0.086, -0.013) * | +0.114 (-0.007, +0.237) |
| zoom 0.8 (no blur) | -0.728 (-1.071, -0.346) * | -0.103 (-0.198, -0.023) * | -0.731 (-0.787, -0.676) * | +0.142 (+0.102, +0.173) * | -0.018 (-0.032, -0.003) * | +0.068 (-0.069, +0.197) |
| zoom 0.6 (no blur) | -1.801 (-2.304, -1.356) * | -0.156 (-0.291, -0.037) * | -1.680 (-1.805, -1.555) * | +0.268 (+0.226, +0.307) * | -0.089 (-0.122, -0.058) * | +0.154 (-0.042, +0.345) |

Examples (`meter_edits_examples.png`): unedited, background blur r=2/4/8, whole blur, zoom 0.8, 0.6.

What a working blur meter should show: background sharpness falls and CLIP blur rises with the radius, and neither moves much under zoom alone. Framing should rise with zoom and stay put under blur.

