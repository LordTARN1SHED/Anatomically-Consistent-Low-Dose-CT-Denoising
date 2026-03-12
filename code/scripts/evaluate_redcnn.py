#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.baselines import REDCNNEvaluator, load_redcnn_config
from egldm.config import load_eval_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RED-CNN baseline")
    parser.add_argument("--config", type=str, default="configs/eval_redcnn_medium_test.yaml")
    args = parser.parse_args()

    run_cfg = load_eval_run_config(args.config)
    train_cfg = load_redcnn_config(run_cfg.train_config_path)
    train_cfg.seed = run_cfg.seed
    if run_cfg.dataset_split is not None:
        train_cfg.eval.dataset_split = run_cfg.dataset_split

    evaluator = REDCNNEvaluator(
        cfg=train_cfg,
        checkpoint_path=run_cfg.checkpoint_path,
        output_dir=run_cfg.output_dir,
    )
    metrics = evaluator.evaluate(
        num_samples=run_cfg.num_samples,
        save_predictions=run_cfg.save_predictions,
        compute_fid=run_cfg.compute_fid,
    )
    print(metrics)


if __name__ == "__main__":
    main()
