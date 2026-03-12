#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.config import load_project_config
from egldm.train import ControlNetTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one sanity train step")
    parser.add_argument("--config", type=str, default="configs/train_tiny.yaml")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    trainer = ControlNetTrainer(cfg)

    train_loader, _, cached = trainer._build_loaders()
    batch = next(iter(train_loader))

    z0, c_ldct, edge_map = trainer._prepare_latents(batch, cached=cached)
    loss = trainer._forward_loss(z0, c_ldct, edge_map)

    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainer.optimizer.step()

    print({"sanity_loss": float(loss.item())})


if __name__ == "__main__":
    main()
