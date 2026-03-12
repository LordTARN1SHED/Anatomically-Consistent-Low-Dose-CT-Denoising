from __future__ import annotations

import torch
from diffusers import AutoencoderKL, VQModel

from egldm.models.identity_autoencoder import IdentityAutoencoder


def _adapt_input_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
    if x.shape[1] == target_channels:
        return x
    if x.shape[1] == 1 and target_channels == 3:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] == 3 and target_channels == 1:
        return x.mean(dim=1, keepdim=True)
    raise ValueError(f"Unsupported channel adaptation: input={x.shape[1]}, target={target_channels}")


def _adapt_output_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
    if x.shape[1] == target_channels:
        return x
    if x.shape[1] == 3 and target_channels == 1:
        return x.mean(dim=1, keepdim=True)
    if x.shape[1] == 1 and target_channels == 3:
        return x.repeat(1, 3, 1, 1)
    raise ValueError(f"Unsupported output channel adaptation: output={x.shape[1]}, target={target_channels}")


def encode_to_latent(vae: AutoencoderKL | VQModel | IdentityAutoencoder, x: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    x = _adapt_input_channels(x, vae.config.in_channels)
    encoded = vae.encode(x)

    if isinstance(vae, AutoencoderKL):
        z = encoded.latent_dist.sample()
    else:
        z = encoded.latents

    return z * scaling_factor


def decode_from_latent(
    vae: AutoencoderKL | VQModel | IdentityAutoencoder,
    z: torch.Tensor,
    scaling_factor: float,
    out_channels: int = 1,
) -> torch.Tensor:
    z = z / scaling_factor
    decoded = vae.decode(z).sample
    return _adapt_output_channels(decoded, out_channels)
