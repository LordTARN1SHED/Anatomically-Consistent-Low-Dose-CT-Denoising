from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _downsample_latent(latent: torch.Tensor, downsample_factor: int) -> torch.Tensor:
    factor = max(1, int(downsample_factor))
    if factor <= 1:
        return latent
    h, w = latent.shape[-2:]
    if h < factor or w < factor:
        return latent
    return F.avg_pool2d(latent, kernel_size=factor, stride=factor)


class LatentConditionProjector(nn.Module):
    """Maps latent channels C_latent -> cross_attention_dim for UNet conditioning."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, latent: torch.Tensor, downsample_factor: int = 1) -> torch.Tensor:
        # latent: [B, C, H, W] -> tokens [B, H*W, C]
        latent = _downsample_latent(latent, downsample_factor)
        b, c, h, w = latent.shape
        tokens = latent.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.proj(tokens)


def latent_to_condition_tokens(
    latent: torch.Tensor,
    cross_attention_dim: int,
    projector: nn.Module | None,
    downsample_factor: int = 1,
) -> torch.Tensor:
    latent = _downsample_latent(latent, downsample_factor)
    b, c, h, w = latent.shape
    tokens = latent.permute(0, 2, 3, 1).reshape(b, h * w, c)

    if c == cross_attention_dim and projector is None:
        return tokens

    if projector is None:
        raise ValueError(
            f"cross_attention_dim={cross_attention_dim} does not match latent channels={c}. "
            "Provide a condition projector or use matching dimensions."
        )

    return projector(latent, downsample_factor=downsample_factor)
