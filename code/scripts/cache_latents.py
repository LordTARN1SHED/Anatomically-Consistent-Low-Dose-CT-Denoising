#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.config import load_project_config
from egldm.data import build_dataloaders
from egldm.data.latent_cache import build_latent_cache
from egldm.models.autoencoder_utils import encode_to_latent
from egldm.models.factory import build_models
from egldm.utils import get_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache VAE latents for faster training")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    seed_everything(cfg.seed)

    device = get_device()
    train_loader, val_loader = build_dataloaders(cfg.data, seed=cfg.seed)

    bundle = build_models(cfg.model, cfg.train)
    bundle.vae.to(device).eval()

    def encode_fn(x):
        with torch.no_grad():
            return encode_to_latent(bundle.vae, x, cfg.model.latent_scaling_factor)

    import torch

    build_latent_cache(
        dataloader=train_loader,
        encode_fn=encode_fn,
        output_dir=cfg.train.latent_cache_dir,
        split="train",
        device=device,
    )
    build_latent_cache(
        dataloader=val_loader,
        encode_fn=encode_fn,
        output_dir=cfg.train.latent_cache_dir,
        split="val",
        device=device,
    )

    print(f"Latent cache written to {Path(cfg.train.latent_cache_dir).resolve()}")


if __name__ == "__main__":
    main()
