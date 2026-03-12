from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def compute_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.abs(pred - gt)


def save_comparison(
    ldct: np.ndarray,
    redcnn: np.ndarray | None,
    ldm: np.ndarray | None,
    egldm: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
    redcnn_label: str = "RED-CNN",
    ldm_label: str = "Standard LDM",
    egldm_label: str = "EG-LDM",
    vmax_err: float = 0.25,
) -> None:
    entries = [("LDCT", ldct)]
    if redcnn is not None:
        entries.append((redcnn_label, redcnn))
    if ldm is not None:
        entries.append((ldm_label, ldm))
    entries.extend([
        (egldm_label, egldm),
        ("HDCT (GT)", gt),
        (f"|{egldm_label} - GT|", compute_error_map(egldm, gt)),
    ])

    fig, axes = plt.subplots(1, len(entries), figsize=(3.2 * len(entries), 3.8))
    if len(entries) == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, entries):
        if "|" in title:
            err_vmax = min(vmax_err, max(float(np.percentile(img, 99.5)), 1e-6))
            ax.imshow(img, cmap="viridis", vmin=0.0, vmax=err_vmax)
        else:
            ax.imshow(img, cmap="gray", vmin=-1.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_roi_zoom(
    image_map: dict[str, np.ndarray],
    roi: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    x1, y1, x2, y2 = roi
    labels = list(image_map.keys())

    fig, axes = plt.subplots(2, len(labels), figsize=(3.2 * len(labels), 6.2))
    for idx, key in enumerate(labels):
        img = image_map[key]
        axes[0, idx].imshow(img, cmap="gray", vmin=-1.0, vmax=1.0)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="cyan", linewidth=1.5)
        axes[0, idx].add_patch(rect)
        axes[0, idx].set_title(key)
        axes[0, idx].axis("off")

        crop = img[y1:y2, x1:x2]
        axes[1, idx].imshow(crop, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[1, idx].set_title(f"{key} ROI")
        axes[1, idx].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
