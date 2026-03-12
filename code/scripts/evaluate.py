#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.config import load_eval_run_config, load_project_config
from egldm.eval import EGLDMEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EG-LDM checkpoints")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    run_cfg = load_eval_run_config(args.config)
    train_cfg = load_project_config(run_cfg.train_config_path)
    train_cfg.seed = run_cfg.seed
    if run_cfg.num_inference_steps is not None:
        train_cfg.eval.num_inference_steps = run_cfg.num_inference_steps
    if run_cfg.dataset_split is not None:
        train_cfg.eval.dataset_split = run_cfg.dataset_split
    if run_cfg.inference_mode is not None:
        train_cfg.eval.inference_mode = run_cfg.inference_mode
    if run_cfg.strength is not None:
        train_cfg.eval.strength = run_cfg.strength
    if run_cfg.controlnet_conditioning_scale is not None:
        train_cfg.eval.controlnet_conditioning_scale = run_cfg.controlnet_conditioning_scale
    if run_cfg.direct_timestep is not None:
        train_cfg.eval.direct_timestep = run_cfg.direct_timestep
    if run_cfg.output_blend_alpha is not None:
        train_cfg.eval.output_blend_alpha = run_cfg.output_blend_alpha
    if run_cfg.edge_adaptive_blend_strength is not None:
        train_cfg.eval.edge_adaptive_blend_strength = run_cfg.edge_adaptive_blend_strength
    if run_cfg.edge_adaptive_blur_sigma is not None:
        train_cfg.eval.edge_adaptive_blur_sigma = run_cfg.edge_adaptive_blur_sigma
    if run_cfg.anchor_blend_alpha is not None:
        train_cfg.eval.anchor_blend_alpha = run_cfg.anchor_blend_alpha

    evaluator = EGLDMEvaluator(
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
