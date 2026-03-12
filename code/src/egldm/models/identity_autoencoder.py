from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityAutoencoder(nn.Module):
    """Deterministic downsample/upsample autoencoder for stable debugging and fast convergence."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 4, downsample_factor: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=in_channels,
            latent_channels=latent_channels,
            downsample_factor=downsample_factor,
        )

    def encode(self, x: torch.Tensor):
        h = x.shape[-2] // self.config.downsample_factor
        w = x.shape[-1] // self.config.downsample_factor
        z = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        if z.shape[1] != self.config.latent_channels:
            z = z.repeat(1, self.config.latent_channels, 1, 1)

        return SimpleNamespace(latents=z)

    def decode(self, z: torch.Tensor):
        x = z.mean(dim=1, keepdim=True)
        h = z.shape[-2] * self.config.downsample_factor
        w = z.shape[-1] * self.config.downsample_factor
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return SimpleNamespace(sample=x)
