from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel, VQModel

from egldm.config import ModelConfig, TrainConfig
from egldm.models.conditioning import LatentConditionProjector
from egldm.models.identity_autoencoder import IdentityAutoencoder


@dataclass
class ModelBundle:
    vae: AutoencoderKL | VQModel | IdentityAutoencoder
    unet: UNet2DConditionModel
    controlnet: ControlNetModel
    noise_scheduler: DDPMScheduler
    condition_projector: nn.Module | None


def _freeze_params(mod: nn.Module, freeze: bool) -> None:
    for p in mod.parameters():
        p.requires_grad = not freeze


def _infer_cross_attention_dim(unet: UNet2DConditionModel) -> int:
    cross_dim = unet.config.cross_attention_dim
    if isinstance(cross_dim, int):
        return cross_dim
    if isinstance(cross_dim, (tuple, list)):
        return int(cross_dim[0])
    raise ValueError(f"Unsupported cross_attention_dim type: {type(cross_dim)}")


def _build_tiny_vae(image_size: int = 256, latent_channels: int = 4) -> AutoencoderKL:
    return AutoencoderKL(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128, 256, 256),
        layers_per_block=2,
        latent_channels=latent_channels,
        norm_num_groups=32,
    )


def _build_tiny_unet(sample_size: int = 32, latent_channels: int = 4) -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=latent_channels,
        out_channels=latent_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=latent_channels,
        attention_head_dim=8,
    )


def _load_autoencoder(model_cfg: ModelConfig):
    if model_cfg.use_identity_autoencoder:
        return IdentityAutoencoder(in_channels=1, latent_channels=model_cfg.latent_channels, downsample_factor=8)

    vae_path = model_cfg.pretrained_vae_name_or_path or model_cfg.pretrained_model_name_or_path
    if vae_path is None:
        return _build_tiny_vae(image_size=model_cfg.sample_size * 8, latent_channels=model_cfg.latent_channels)

    # Prefer VQModel for explicit VQ-GAN behavior; fallback to AutoencoderKL.
    vae_subfolder = model_cfg.vae_subfolder or None
    try:
        return VQModel.from_pretrained(vae_path, subfolder=vae_subfolder)
    except Exception:
        return AutoencoderKL.from_pretrained(vae_path, subfolder=vae_subfolder)


def _load_unet(model_cfg: ModelConfig):
    unet_path = model_cfg.pretrained_unet_name_or_path or model_cfg.pretrained_model_name_or_path
    if unet_path is None or model_cfg.use_tiny_models:
        return _build_tiny_unet(sample_size=model_cfg.sample_size, latent_channels=model_cfg.latent_channels)

    unet_subfolder = model_cfg.unet_subfolder or None
    return UNet2DConditionModel.from_pretrained(
        unet_path,
        subfolder=unet_subfolder,
    )


def build_models(model_cfg: ModelConfig, train_cfg: TrainConfig) -> ModelBundle:
    vae = _load_autoencoder(model_cfg)
    unet = _load_unet(model_cfg)

    controlnet = ControlNetModel.from_unet(
        unet,
        conditioning_channels=model_cfg.controlnet_conditioning_channels,
    )

    _freeze_params(vae, freeze=model_cfg.vae_frozen)
    _freeze_params(unet, freeze=model_cfg.unet_frozen)

    cross_dim = _infer_cross_attention_dim(unet)
    projector: nn.Module | None = None

    if model_cfg.enable_ldct_condition and cross_dim != model_cfg.latent_channels:
        projector = LatentConditionProjector(model_cfg.latent_channels, cross_dim)
        _freeze_params(projector, freeze=not train_cfg.train_condition_projector)

    scheduler = DDPMScheduler(num_train_timesteps=train_cfg.num_train_timesteps)

    return ModelBundle(
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        noise_scheduler=scheduler,
        condition_projector=projector,
    )


def zero_conv_parameters(controlnet: ControlNetModel) -> Iterable[tuple[str, torch.Tensor, torch.Tensor]]:
    for name, module in controlnet.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            if "controlnet_down_blocks" in name or "controlnet_mid_block" in name:
                bias = module.bias if module.bias is not None else torch.zeros(1)
                yield name, module.weight.data, bias.data


def summarize_zero_conv_init(controlnet: ControlNetModel, atol: float = 0.0) -> dict[str, object]:
    stats: dict[str, object] = {"total": 0, "all_zero": True, "layers": []}

    for name, weight, bias in zero_conv_parameters(controlnet):
        w_abs_max = float(weight.abs().max().item())
        b_abs_max = float(bias.abs().max().item())
        is_zero = (w_abs_max <= atol) and (b_abs_max <= atol)
        stats["total"] += 1
        if not is_zero:
            stats["all_zero"] = False
        stats["layers"].append(
            {
                "name": name,
                "weight_abs_max": w_abs_max,
                "bias_abs_max": b_abs_max,
                "is_zero": is_zero,
            }
        )

    return stats
