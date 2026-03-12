# Open-Source Release Notes

This folder is the cleaned publication companion package intended for a public GitHub repository.

## Included

- Final paper source and compiled PDF
- Core code, scripts, and configs
- Patient-level split metadata
- Final quantitative metrics and result summaries
- Key figures used in the paper
- Release-compatible baseline checkpoint (`redcnn_best.pt`)
- Short conference presentation and script

## Intentionally excluded

- Raw LIDC-IDRI DICOM data
- Large cached tensors and prediction dumps
- Full hybrid training checkpoint that exceeds standard GitHub file limits
- Local operating-system artifacts and LaTeX build byproducts

## Pre-publish checklist

1. Choose an open-source license and replace `LICENSE_PENDING.md` with `LICENSE`.
2. Review `docs/CHECKPOINTS.md` if you want to publish the omitted hybrid checkpoint through Git LFS or release assets.
3. Verify that no raw patient data or private paths were accidentally added after this package was prepared.
4. Optionally create a GitHub release entry pointing to the main paper PDF and the baseline checkpoint.
