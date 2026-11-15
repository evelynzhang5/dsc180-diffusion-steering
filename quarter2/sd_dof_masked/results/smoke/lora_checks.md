# Smoke test 3: LoRA on the rendered pairs (8 images at sigma 8; pipeline check)

| check | result | detail |
|---|---|---|
| training runs; loss finite | PASS | 50 steps, batch 2, 797,184 LoRA parameters, 66 s, peak 4.88 GB |
| fixed evaluation loss goes down | PASS | 0.1463 -> 0.1419 |
| adapter saved and reloaded into a fresh pipeline | PASS | pytorch_lora_weights.safetensors |
| adapter weight 0 reproduces the plain model (mean pixel change < 0.004) | PASS | max 0.0000 |
| adapter weight 1 changes the image (mean pixel change > 0.01) | PASS | min 0.1370 |
| paired changes vs plain at weights 0.5 / 1 (bg, fg) | info | bg +0.035 fg -0.085 / bg -0.347 fg +0.061 |
| contact sheet <= 25 images | PASS | 8 images |

Columns in `lora_examples.png`: plain model, adapter weight 0, 0.5, 1 (same caption and seed). 50 steps on 8 images only checks the training, saving and loading path; it is not a trained baseline.

Total time 194 s.
