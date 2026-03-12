#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_triplet(psnr: float, ssim: float, rmse: float) -> dict[str, float]:
    return {"PSNR": float(psnr), "SSIM": float(ssim), "RMSE": float(rmse)}


def build_methods() -> tuple[list[str], dict[str, dict[str, float]], dict[str, str]]:
    summary = _load_json(ROOT / "outputs" / "summary_real_lidc" / "results_summary.json")["rows"]
    summary_by_name = {row["Method"]: row for row in summary}
    classical_naive = _load_json(ROOT / "outputs" / "classical_naive_baselines.json")
    classical_extra = _load_json(ROOT / "outputs" / "classical_extra_baselines.json")
    hybrid = _load_json(ROOT / "outputs" / "summary_real_lidc" / "hybrid_refinement_summary.json")

    methods = {
        "LDCT": _metric_triplet(
            summary_by_name["LDCT"]["PSNR"],
            summary_by_name["LDCT"]["SSIM"],
            summary_by_name["LDCT"]["RMSE"],
        ),
        "Gaussian $\\sigma=2$": _metric_triplet(
            classical_naive["Gaussian_over_smooth_sigma2.0"]["PSNR"],
            classical_naive["Gaussian_over_smooth_sigma2.0"]["SSIM"],
            classical_naive["Gaussian_over_smooth_sigma2.0"]["RMSE"],
        ),
        "Bilateral": _metric_triplet(
            classical_extra["Bilateral_default"]["PSNR"],
            classical_extra["Bilateral_default"]["SSIM"],
            classical_extra["Bilateral_default"]["RMSE"],
        ),
        "TV": _metric_triplet(
            classical_extra["TV_default_weight0.1"]["PSNR"],
            classical_extra["TV_default_weight0.1"]["SSIM"],
            classical_extra["TV_default_weight0.1"]["RMSE"],
        ),
        "No-Edge EG-LDM": _metric_triplet(
            summary_by_name["No-Edge EG-LDM"]["PSNR"],
            summary_by_name["No-Edge EG-LDM"]["SSIM"],
            summary_by_name["No-Edge EG-LDM"]["RMSE"],
        ),
        "Edge-Guided EG-LDM": _metric_triplet(
            summary_by_name["Edge-Guided EG-LDM + Tuned"]["PSNR"],
            summary_by_name["Edge-Guided EG-LDM + Tuned"]["SSIM"],
            summary_by_name["Edge-Guided EG-LDM + Tuned"]["RMSE"],
        ),
        "RED-CNN": _metric_triplet(
            hybrid["redcnn_test"]["PSNR"],
            hybrid["redcnn_test"]["SSIM"],
            hybrid["redcnn_test"]["RMSE"],
        ),
        "Hybrid": _metric_triplet(
            hybrid["hybrid_test"]["PSNR"],
            hybrid["hybrid_test"]["SSIM"],
            hybrid["hybrid_test"]["RMSE"],
        ),
    }

    colors = {
        "LDCT": "#9CA3AF",
        "Gaussian $\\sigma=2$": "#D1D5DB",
        "Bilateral": "#94A3B8",
        "TV": "#64748B",
        "No-Edge EG-LDM": "#60A5FA",
        "Edge-Guided EG-LDM": "#2563EB",
        "RED-CNN": "#F59E0B",
        "Hybrid": "#DC2626",
    }
    order = [
        "LDCT",
        "Gaussian $\\sigma=2$",
        "Bilateral",
        "TV",
        "No-Edge EG-LDM",
        "Edge-Guided EG-LDM",
        "RED-CNN",
        "Hybrid",
    ]
    return order, methods, colors


def plot_performance_overview(out_path: Path) -> None:
    order, methods, colors = build_methods()
    hybrid_summary = _load_json(ROOT / "outputs" / "hybrid_anchor_best_full.json")

    y = np.arange(len(order))
    fig = plt.figure(figsize=(10.8, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.38, wspace=0.28)
    metric_specs = [
        ("PSNR", gs[0, 0], True, "PSNR (dB)"),
        ("SSIM", gs[0, 1], True, "SSIM"),
        ("RMSE", gs[1, 0], False, "RMSE"),
    ]

    for metric, slot, higher_is_better, xlabel in metric_specs:
        ax = fig.add_subplot(slot)
        vals = [methods[name][metric] for name in order]
        ax.barh(y, vals, color=[colors[name] for name in order], edgecolor="none", height=0.72)
        ax.set_yticks(y, order, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(axis="x", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        if metric == "PSNR":
            ax.set_title("Test-Set Fidelity Overview", fontsize=11, pad=8)
        if metric == "RMSE":
            ax.set_title("Lower Is Better", fontsize=11, pad=8)
        if metric == "SSIM":
            ax.set_title("Structural Similarity", fontsize=11, pad=8)

        margin = 0.010 if metric == "SSIM" else (0.45 if metric == "PSNR" else 0.0015)
        for idx, val in enumerate(vals):
            text = f"{val:.4f}" if metric != "SSIM" else f"{val:.3f}"
            ax.text(val + margin, idx, text, va="center", ha="left", fontsize=8.5)

        if metric == "PSNR":
            ax.set_xlim(23.8, 36.2)
        elif metric == "SSIM":
            ax.set_xlim(0.58, 0.93)
        else:
            ax.set_xlim(0.03, 0.12)

    ax_line = fig.add_subplot(gs[1, 1])
    splits = ["Validation", "Test"]
    x = np.arange(len(splits))
    anchor_psnr = [hybrid_summary["anchor_val"]["PSNR"], hybrid_summary["anchor_test"]["PSNR"]]
    hybrid_psnr = [hybrid_summary["hybrid_val"]["PSNR"], hybrid_summary["hybrid_test"]["PSNR"]]
    anchor_ssim = [hybrid_summary["anchor_val"]["SSIM"], hybrid_summary["anchor_test"]["SSIM"]]
    hybrid_ssim = [hybrid_summary["hybrid_val"]["SSIM"], hybrid_summary["hybrid_test"]["SSIM"]]

    ax_line.plot(x, anchor_psnr, marker="o", linewidth=2.2, color="#F59E0B", label="RED-CNN (PSNR)")
    ax_line.plot(x, hybrid_psnr, marker="o", linewidth=2.2, color="#DC2626", label="Hybrid (PSNR)")
    ax_line.set_xticks(x, splits, fontsize=9)
    ax_line.set_ylabel("PSNR (dB)", fontsize=10)
    ax_line.set_ylim(34.95, 35.5)
    ax_line.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax_line.set_title("Hybrid Improves over the Anchor on Both Splits", fontsize=11, pad=8)

    ax_line2 = ax_line.twinx()
    ax_line2.plot(x, anchor_ssim, marker="s", linestyle="--", linewidth=1.8, color="#B45309", label="RED-CNN (SSIM)")
    ax_line2.plot(x, hybrid_ssim, marker="s", linestyle="--", linewidth=1.8, color="#991B1B", label="Hybrid (SSIM)")
    ax_line2.set_ylabel("SSIM", fontsize=10)
    ax_line2.set_ylim(0.900, 0.913)

    offsets = [(-10, 8), (14, 8)]
    for idx, (a, h) in enumerate(zip(anchor_psnr, hybrid_psnr)):
        ax_line.annotate(
            f"{h - a:+.3f} dB",
            xy=(x[idx], h),
            xytext=offsets[idx],
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#7F1D1D",
        )

    handles1, labels1 = ax_line.get_legend_handles_labels()
    handles2, labels2 = ax_line2.get_legend_handles_labels()
    ax_line.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="lower center", ncol=2, frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_performance_overview(ROOT / "NeurIPS_285" / "figures" / "performance_overview.png")


if __name__ == "__main__":
    main()
