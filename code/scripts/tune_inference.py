#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import itertools

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.config import load_project_config
from egldm.eval import EGLDMEvaluator
from egldm.utils import ensure_dir


def parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def trial_score(trial: dict[str, float], objective: str) -> float:
    if objective == "psnr":
        return float(trial["PSNR"])
    if objective == "ssim":
        return float(trial["SSIM"])
    if objective == "rmse":
        return -float(trial["RMSE"])
    if objective == "balanced":
        return (
            float(trial.get("Delta_PSNR", trial["PSNR"]))
            + 10.0 * float(trial.get("Delta_SSIM", trial["SSIM"]))
            - 10.0 * float(trial.get("Delta_RMSE", trial["RMSE"]))
        )
    raise ValueError(f"Unsupported objective: {objective}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune inference hyperparameters for EG-LDM")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--steps", type=str, default="20,30")
    parser.add_argument("--strengths", type=str, default="0.25,0.35,0.5")
    parser.add_argument("--scales", type=str, default="0.8,1.0")
    parser.add_argument("--mode", type=str, default="img2img")
    parser.add_argument("--direct_timesteps", type=str, default="50")
    parser.add_argument("--blend_alphas", type=str, default="1.0")
    parser.add_argument("--anchor_blend_alphas", type=str, default=None)
    parser.add_argument("--edge_strengths", type=str, default="0.0")
    parser.add_argument("--edge_blur_sigmas", type=str, default="0.0")
    parser.add_argument("--objective", type=str, default="balanced")
    parser.add_argument("--dataset_split", type=str, default=None)
    args = parser.parse_args()

    cfg = load_project_config(args.train_config)
    if args.dataset_split is not None:
        cfg.eval.dataset_split = args.dataset_split
    out_dir = ensure_dir(args.out_dir)

    steps = parse_list(args.steps, int)
    strengths = parse_list(args.strengths, float)
    scales = parse_list(args.scales, float)
    direct_ts = parse_list(args.direct_timesteps, int)
    blend_alphas = parse_list(args.blend_alphas, float)
    if args.anchor_blend_alphas is None:
        anchor_blend_alphas = [float(cfg.eval.anchor_blend_alpha)]
    else:
        anchor_blend_alphas = parse_list(args.anchor_blend_alphas, float)
    edge_strengths = parse_list(args.edge_strengths, float)
    edge_blur_sigmas = parse_list(args.edge_blur_sigmas, float)

    evaluator = EGLDMEvaluator(cfg=cfg, checkpoint_path=args.checkpoint, output_dir=out_dir / "tmp_eval")

    trials = []
    if args.mode == "direct_x0":
        grid = itertools.product(
            steps,
            strengths,
            scales,
            direct_ts,
            blend_alphas,
            anchor_blend_alphas,
            edge_strengths,
            edge_blur_sigmas,
        )
    else:
        grid = itertools.product(
            steps,
            strengths,
            scales,
            [cfg.eval.direct_timestep],
            blend_alphas,
            anchor_blend_alphas,
            edge_strengths,
            edge_blur_sigmas,
        )

    for n_steps, strength, scale, t_direct, blend_alpha, anchor_blend_alpha, edge_strength, edge_blur_sigma in grid:
        cfg.eval.num_inference_steps = int(n_steps)
        cfg.eval.inference_mode = args.mode
        cfg.eval.strength = float(strength)
        cfg.eval.controlnet_conditioning_scale = float(scale)
        cfg.eval.direct_timestep = int(t_direct)
        cfg.eval.output_blend_alpha = float(blend_alpha)
        cfg.eval.anchor_blend_alpha = float(anchor_blend_alpha)
        cfg.eval.edge_adaptive_blend_strength = float(edge_strength)
        cfg.eval.edge_adaptive_blur_sigma = float(edge_blur_sigma)

        metrics = evaluator.evaluate(
            num_samples=args.num_samples,
            save_predictions=False,
            compute_fid=False,
        )

        trial = {
            "num_inference_steps": n_steps,
            "strength": strength,
            "controlnet_conditioning_scale": scale,
            "direct_timestep": int(t_direct),
            "output_blend_alpha": float(blend_alpha),
            "anchor_blend_alpha": float(anchor_blend_alpha),
            "edge_adaptive_blend_strength": float(edge_strength),
            "edge_adaptive_blur_sigma": float(edge_blur_sigma),
            **metrics,
        }
        trial["objective_score"] = trial_score(trial, args.objective)
        print(trial)
        trials.append(trial)

    trials_sorted = sorted(trials, key=lambda x: (x["objective_score"], x["PSNR"], x["SSIM"]), reverse=True)
    best = trials_sorted[0]

    summary = {
        "best": best,
        "trials": trials_sorted,
    }

    (out_dir / "inference_tuning.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Best:", best)


if __name__ == "__main__":
    main()
