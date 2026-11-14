# Candidate directions (round 4's activations: 1,200 photos at sigma 0.344, empty prompt)

Refit: held-out AUC 0.964; |cos(top eigenvector, round 4 direction)| = 1.0000; eigenvalues 1.04e+04, 0.555, 0.319, 0.219, 0.187 (lambda1 / lambda2 = 18729.98).

| candidate | probe AUC (held-out 20%) | kept |
|---|---|---|
| rfm_eig1 | 1.000 | yes |
| rfm_eig2 | 0.669 | yes |
| rfm_eig3 | 0.700 | yes |
| meandiff | 0.948 | yes |
| meandiff_std | 0.949 | no, duplicate of meandiff |

Probe AUCs of the RFM eigenvectors use a split that overlaps the RFM's own training split, so they are optimistic; they are not used to choose.

| pair | cosine |
|---|---|
| rfm_eig1|rfm_eig2 | +0.000 |
| rfm_eig1|rfm_eig3 | +0.000 |
| rfm_eig1|meandiff | +0.727 |
| rfm_eig1|meandiff_std | +0.757 |
| rfm_eig2|rfm_eig3 | +0.000 |
| rfm_eig2|meandiff | +0.278 |
| rfm_eig2|meandiff_std | +0.289 |
| rfm_eig3|meandiff | +0.304 |
| rfm_eig3|meandiff_std | +0.312 |
| meandiff|meandiff_std | +0.968 |
