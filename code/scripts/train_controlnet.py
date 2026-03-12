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
    parser = argparse.ArgumentParser(description="Train EG-LDM ControlNet")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    trainer = ControlNetTrainer(cfg)
    if args.resume_from:
        trainer.resume_from_checkpoint(args.resume_from)
    trainer.run()


if __name__ == "__main__":
    main()
