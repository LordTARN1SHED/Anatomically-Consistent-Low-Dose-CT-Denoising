#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.config import load_project_config
from egldm.models.factory import build_models, summarize_zero_conv_init
from egldm.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ControlNet zero-conv initialization")
    parser.add_argument("--config", type=str, default="configs/train_tiny.yaml")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    bundle = build_models(cfg.model, cfg.train)

    stats = summarize_zero_conv_init(bundle.controlnet, atol=0.0)

    out_dir = ensure_dir(Path(cfg.output_dir) / "checks")
    out_path = out_dir / "zero_conv_init.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    if not stats.get("all_zero", False):
        raise SystemExit("Zero-conv check failed: found non-zero initialization.")


if __name__ == "__main__":
    main()
