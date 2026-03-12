# Checkpoints

## Included directly

- `results/checkpoints/redcnn_best.pt`

This baseline checkpoint is small enough for a normal public GitHub repository.

## Not included directly

The final Hybrid EG-LDM checkpoint is not included in this package because the original training checkpoint is too large for a normal GitHub repository:

- Original file: `outputs/egldm_lidc_hybrid_medium/checkpoints/controlnet_best.pt`
- Approximate size: `1.51 GB`

The main reason is that the training checkpoint stores:

- ControlNet weights
- Trainable U-Net weights
- Optimizer state
- Training history and config payload

## Public-release recommendation

For a real public release, handle the hybrid checkpoint in one of these ways:

1. Publish it as a GitHub Release asset.
2. Store it with Git LFS.
3. Export a stripped inference-only checkpoint and host it externally.

## Manifest

See `results/checkpoints/checkpoint_manifest.json` for the packaged checkpoint manifest.
