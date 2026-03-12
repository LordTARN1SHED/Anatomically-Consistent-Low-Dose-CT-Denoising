#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.baselines import REDCNNTrainer, load_redcnn_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RED-CNN baseline")
    parser.add_argument("--config", type=str, default="configs/train_redcnn_medium.yaml")
    args = parser.parse_args()

    cfg = load_redcnn_config(args.config)
    trainer = REDCNNTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
