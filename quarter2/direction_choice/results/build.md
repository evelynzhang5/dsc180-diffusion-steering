# Candidate directions (block 9, 10,000 training images at sigma 0.21)

Probe AUC: held-out (20%) AUC of the raw-feature projection onto each direction, class vs. the other nine. Reported for comparison only; selection uses generated images.

| class | rfm_top3 | rfm_eig1 | rfm_eig2 | meandiff | meandiff_std | cos(top3, eig1) | cos(top3, meandiff) | cos(top3, meandiff_std) |
|---|---|---|---|---|---|---|---|---|
| airplane | 0.984 | 0.983 | 0.696 | 0.907 | 0.906 | 1.00 | 0.36 | 0.35 |
| automobile | 0.990 | 0.990 | 0.894 | 0.942 | 0.943 | 1.00 | 0.54 | 0.55 |
| bird | 0.979 | 0.978 | 0.817 | 0.843 | 0.842 | 1.00 | 0.33 | 0.33 |
| cat | 0.945 | 0.945 | 0.754 | 0.842 | 0.841 | 1.00 | 0.44 | 0.45 |
| deer | 0.980 | 0.980 | 0.542 | 0.873 | 0.870 | 1.00 | 0.35 | 0.35 |
| dog | 0.964 | 0.964 | 0.842 | 0.868 | 0.864 | 1.00 | 0.47 | 0.47 |
| frog | 0.992 | 0.992 | 0.553 | 0.939 | 0.936 | 1.00 | 0.56 | 0.55 |
| horse | 0.979 | 0.979 | 0.855 | 0.910 | 0.910 | 1.00 | 0.57 | 0.57 |
| ship | 0.996 | 0.996 | 0.610 | 0.935 | 0.933 | 1.00 | 0.35 | 0.34 |
| truck | 0.983 | 0.983 | 0.860 | 0.922 | 0.920 | 1.00 | 0.49 | 0.49 |
