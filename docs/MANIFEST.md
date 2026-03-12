# Package Manifest

## Root

- `README.md`: release overview
- `requirements.txt`, `pyproject.toml`: environment and packaging metadata

## code/

- `code/src/egldm/`: core implementation
- `code/scripts/`: runnable experiment scripts
- `code/configs/`: curated training and evaluation YAML files

## metadata/

- `metadata/lidc_index/patient_splits.json`
- `metadata/lidc_index/split_manifest.json`
- `metadata/lidc_raw/README.md`

## results/

- `results/summaries/`: final quantitative summaries and tuning outputs
- `results/metrics/`: key `metrics.json` files from the final reported runs
- `results/figures/`: representative paper figures and training curves
- `results/logs/`: training histories, run metadata, and archived stderr logs
- `results/checkpoints/`: release-compatible checkpoints and checkpoint manifest

## paper/

- `paper/neurips_2024.tex`
- `paper/neurips_2024.pdf`
- `paper/neurips_2024.sty`
- `paper/figures/`

## docs/

- `ENVIRONMENT.md`
- `DATA_PREPARATION.md`
- `EXPERIMENTS.md`
- `RESULTS.md`
- `CHECKPOINTS.md`
- `MANIFEST.md`
- `reference/`: archived project report documents
