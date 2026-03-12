# LDCT Hybrid EG-LDM Final Public Release

This folder is a curated, GitHub-ready release package for the final state of the LDCT denoising project.

Before making the repository public, choose an explicit open-source license and replace `LICENSE_PENDING.md` with the final `LICENSE` file.

It includes:

- Final paper source and PDF
- Key code implementation
- Final training and evaluation configs
- Patient-level split metadata
- Curated result summaries, metrics, figures, and logs
- The public-size-compatible baseline checkpoint (`redcnn_best.pt`)

It intentionally excludes:

- Raw LIDC-IDRI DICOM data
- Large cached arrays and full prediction dumps
- Large hybrid checkpoints that exceed normal GitHub file limits

## Folder layout

- `code/`: source code, scripts, and configs
- `docs/`: environment notes, data preparation, experiment commands, result summary, and manifest
- `metadata/`: patient split metadata and raw-data placement note
- `paper/`: final NeurIPS-style paper source, figures, and compiled PDF
- `results/`: curated metrics, summaries, figures, logs, and release-compatible checkpoints

## Recommended entry points

- Start with `docs/MANIFEST.md`
- Reproduction commands are in `docs/EXPERIMENTS.md`
- Final quantitative summary is in `docs/RESULTS.md`
- Checkpoint handling notes are in `docs/CHECKPOINTS.md`

## Final headline result

Under the real-data medium-scale protocol, the final Hybrid EG-LDM slightly outperformed the RED-CNN anchor on the held-out test split:

- Hybrid EG-LDM: `PSNR 35.1197 / SSIM 0.9040 / RMSE 0.03558`
- RED-CNN: `PSNR 35.0199 / SSIM 0.9028 / RMSE 0.03598`

See `results/summaries/hybrid_refinement_summary.md` for the exact comparison.
