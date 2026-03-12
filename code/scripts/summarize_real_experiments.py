#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_history_points(path: Path) -> tuple[list[int], list[float], list[int], list[float]]:
    payload = load_json(path)
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_epochs: list[int] = []
    val_losses: list[float] = []

    for item in payload.get("history", []):
        if "train_loss" in item:
            train_steps.append(int(item["step"]))
            train_losses.append(float(item["train_loss"]))
        if "val_loss" in item:
            val_epochs.append(int(item["epoch"]))
            val_losses.append(float(item["val_loss"]))

    return train_steps, train_losses, val_epochs, val_losses


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    out: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        denom = min(idx + 1, window)
        out.append(running / denom)
    return out


def plot_train_curves(curves: list[dict], out_path: Path, smooth_window: int) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for curve in curves:
        steps = curve["train_steps"]
        losses = curve["train_losses"]
        if not steps:
            continue
        ax.plot(steps, moving_average(losses, smooth_window), label=curve["label"], linewidth=2.0)
    ax.set_title("Real LIDC Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_val_curves(curves: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for curve in curves:
        epochs = curve["val_epochs"]
        losses = curve["val_losses"]
        if not epochs:
            continue
        ax.plot(epochs, losses, marker="o", label=curve["label"], linewidth=2.0)
    ax.set_title("Real LIDC Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def metrics_row(label: str, metrics: dict[str, float]) -> dict[str, float | str]:
    return {
        "Method": label,
        "PSNR": float(metrics["PSNR"]),
        "SSIM": float(metrics["SSIM"]),
        "RMSE": float(metrics["RMSE"]),
        "LDCT_PSNR": float(metrics["LDCT_PSNR"]),
        "LDCT_SSIM": float(metrics["LDCT_SSIM"]),
        "LDCT_RMSE": float(metrics["LDCT_RMSE"]),
        "Delta_PSNR": float(metrics["Delta_PSNR"]),
        "Delta_SSIM": float(metrics["Delta_SSIM"]),
        "Delta_RMSE": float(metrics["Delta_RMSE"]),
    }


def write_markdown(rows: list[dict[str, float | str]], out_path: Path) -> None:
    header = [
        "| Method | PSNR | SSIM | RMSE | Delta_PSNR | Delta_SSIM | Delta_RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = []
    for row in rows:
        body.append(
            "| {Method} | {PSNR:.4f} | {SSIM:.4f} | {RMSE:.5f} | {Delta_PSNR:+.4f} | {Delta_SSIM:+.4f} | {Delta_RMSE:+.5f} |".format(
                **row
            )
        )
    out_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize real LIDC experiment metrics and training curves")
    parser.add_argument("--out_dir", type=str, default="outputs/summary_real_lidc")
    parser.add_argument("--smooth_window", type=int, default=25)
    parser.add_argument(
        "--edge_metrics",
        type=str,
        default="outputs/egldm_lidc_medium_stable/eval_test_tuned/metrics.json",
    )
    parser.add_argument(
        "--no_edge_metrics",
        type=str,
        default="outputs/egldm_lidc_medium_no_edge_stable/eval_test/metrics.json",
    )
    parser.add_argument(
        "--redcnn_metrics",
        type=str,
        default="outputs/redcnn_lidc_medium/eval_test/metrics.json",
    )
    parser.add_argument(
        "--edge_history",
        type=str,
        default="outputs/egldm_lidc_medium_stable/logs/loss_history.json",
    )
    parser.add_argument(
        "--no_edge_history",
        type=str,
        default="outputs/egldm_lidc_medium_no_edge_stable/logs/loss_history.json",
    )
    parser.add_argument(
        "--redcnn_history",
        type=str,
        default="outputs/redcnn_lidc_medium/logs/loss_history.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_metrics = load_json(Path(args.edge_metrics))["metrics"]
    no_edge_metrics = load_json(Path(args.no_edge_metrics))["metrics"]
    redcnn_metrics = load_json(Path(args.redcnn_metrics))["metrics"]

    rows = [
        metrics_row("LDCT", {
            "PSNR": edge_metrics["LDCT_PSNR"],
            "SSIM": edge_metrics["LDCT_SSIM"],
            "RMSE": edge_metrics["LDCT_RMSE"],
            "LDCT_PSNR": edge_metrics["LDCT_PSNR"],
            "LDCT_SSIM": edge_metrics["LDCT_SSIM"],
            "LDCT_RMSE": edge_metrics["LDCT_RMSE"],
            "Delta_PSNR": 0.0,
            "Delta_SSIM": 0.0,
            "Delta_RMSE": 0.0,
        }),
        metrics_row("RED-CNN", redcnn_metrics),
        metrics_row("No-Edge EG-LDM", no_edge_metrics),
        metrics_row("Edge-Guided EG-LDM + Tuned", edge_metrics),
    ]

    curves = []
    for label, history_path in [
        ("Edge-Guided EG-LDM", Path(args.edge_history)),
        ("No-Edge EG-LDM", Path(args.no_edge_history)),
        ("RED-CNN", Path(args.redcnn_history)),
    ]:
        train_steps, train_losses, val_epochs, val_losses = load_history_points(history_path)
        curves.append(
            {
                "label": label,
                "train_steps": train_steps,
                "train_losses": train_losses,
                "val_epochs": val_epochs,
                "val_losses": val_losses,
            }
        )

    plot_train_curves(curves, out_dir / "train_loss_curves.png", smooth_window=max(1, args.smooth_window))
    plot_val_curves(curves, out_dir / "val_loss_curves.png")
    write_markdown(rows, out_dir / "results_summary.md")
    (out_dir / "results_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved summary artifacts to {out_dir}")


if __name__ == "__main__":
    main()
