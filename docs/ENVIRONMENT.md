# Environment

## Recommended setup

- OS: Windows
- GPU: NVIDIA RTX 5080
- Python environment: Conda
- CUDA-enabled PyTorch verified on the original run machine

## Create environment

```bash
conda create -n egldm-lidc python=3.11 -y
conda activate egldm-lidc
pip install -r requirements.txt
```

Then install the CUDA-enabled PyTorch build appropriate for your machine.

## Verified runtime note

The original final runs were executed in a dedicated Conda environment with CUDA enabled. `torch.cuda.is_available()` was verified as `True` on the experiment machine.

## Important repo-local paths

Inside this release package, use:

- `code/src/` for the Python package
- `code/scripts/` for runnable scripts
- `code/configs/` for experiment configs

The scripts are already structured so that running them from this package works with the included `code/src` tree.
