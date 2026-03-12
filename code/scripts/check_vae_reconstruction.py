#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.config import load_project_config
from egldm.data import build_dataloaders
from egldm.train import ControlNetTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Check VAE encode-decode quality")
    parser.add_argument("--config", type=str, default="configs/train_tiny.yaml")
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    trainer = ControlNetTrainer(cfg)
    train_loader, _ = build_dataloaders(cfg.data, seed=cfg.seed)

    stats = trainer.vae_reconstruction_check(train_loader, n_batches=args.batches)
    print(stats)


if __name__ == "__main__":
    main()
