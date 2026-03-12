# Hybrid Refinement Summary

## Test Metrics

| Method | PSNR | SSIM | RMSE |
| --- | ---: | ---: | ---: |
| RED-CNN | 35.0199 | 0.902793 | 0.035981 |
| Edge-Guided EG-LDM + Tuned | 28.4629 | 0.710169 | 0.076518 |
| Hybrid EG-LDM Refinement | 35.1197 | 0.904020 | 0.035582 |

## Delta vs RED-CNN

- PSNR: +0.0998
- SSIM: +0.001227
- RMSE: -0.000400

## Delta vs Edge-Guided EG-LDM + Tuned

- PSNR: +6.6569
- SSIM: +0.193851
- RMSE: -0.040937

## Val Metrics

- Hybrid EG-LDM Refinement (val): PSNR 35.4163, SSIM 0.911527, RMSE 0.034471
