# Final Results

## Main real-data test results

| Method | PSNR | SSIM | RMSE |
| --- | ---: | ---: | ---: |
| LDCT | 28.2181 | 0.6244 | 0.08176 |
| RED-CNN | 35.0199 | 0.9028 | 0.03598 |
| No-Edge EG-LDM | 27.6809 | 0.7063 | 0.08328 |
| Edge-Guided EG-LDM + tuned | 28.4629 | 0.7102 | 0.07652 |
| Hybrid EG-LDM | 35.1197 | 0.9040 | 0.03558 |

## Key conclusion

The final Hybrid EG-LDM slightly exceeded the RED-CNN anchor on the held-out test split:

- `PSNR +0.0998`
- `SSIM +0.001227`
- `RMSE -0.000400`

## Result files

- `results/summaries/results_summary.json`
- `results/summaries/hybrid_refinement_summary.json`
- `results/metrics/hybrid_test_best_metrics.json`
- `results/metrics/redcnn_test_metrics.json`
- `results/metrics/edge_guided_ldm_test_metrics.json`
- `results/metrics/no_edge_ldm_test_metrics.json`

## Figures

- `results/figures/performance_overview.png`
- `results/figures/hybrid_comparison.png`
- `results/figures/hybrid_roi.png`
- `results/figures/train_loss_curves.png`
- `results/figures/val_loss_curves.png`
