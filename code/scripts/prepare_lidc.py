#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from egldm.config import load_project_config
from egldm.data import build_dataloaders
from egldm.data.datasets import _prepare_sample
from egldm.data.lidc import build_patient_splits, load_dicom_hu, scan_lidc_dicom_root, write_prepared_lidc_dataset
from egldm.utils import ensure_dir, save_json


def _preview_triplet(clean: np.ndarray, noisy: np.ndarray, edge: np.ndarray, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(clean, cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title(f"{title}: Clean")
    axes[1].imshow(noisy, cmap="gray", vmin=-1, vmax=1)
    axes[1].set_title(f"{title}: Noisy")
    axes[2].imshow(edge, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"{title}: Edge")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare patient-level LIDC-IDRI indices and previews")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    if cfg.data.mode != "dicom":
        raise SystemExit("scripts/prepare_lidc.py expects data.mode=dicom")
    if cfg.data.dicom_root is None:
        raise SystemExit("data.dicom_root must be set for DICOM preparation")

    index_dir = Path(cfg.data.prepared_index_dir or "data/lidc_index")
    out_dir = ensure_dir(Path(cfg.output_dir) / "data_preview")

    records, scan_stats = scan_lidc_dicom_root(
        cfg.data.dicom_root,
        hu_clip_min=cfg.data.hu_clip_min,
        hu_clip_max=cfg.data.hu_clip_max,
        strict_ct_only=cfg.data.strict_ct_only,
        min_slice_size=cfg.data.min_slice_size,
    )
    split_records, split_patients = build_patient_splits(
        records,
        val_ratio=cfg.data.val_split_ratio,
        test_ratio=cfg.data.test_split_ratio,
        seed=cfg.data.split_seed,
    )
    manifest = write_prepared_lidc_dataset(
        index_dir=index_dir,
        full_records=records,
        split_records=split_records,
        split_patients=split_patients,
        scan_stats=scan_stats,
        split_seed=cfg.data.split_seed,
        val_ratio=cfg.data.val_split_ratio,
        test_ratio=cfg.data.test_split_ratio,
    )

    preview_paths: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        if not split_records[split_name]:
            continue
        entry = split_records[split_name][0]
        hu = load_dicom_hu(entry["dicom_path"])
        sample = _prepare_sample(hu, cfg.data, np.random.default_rng(cfg.seed))
        out_path = out_dir / f"{split_name}_sample_triplet.png"
        _preview_triplet(
            sample["clean_ct"][0].numpy(),
            sample["noisy_ct"][0].numpy(),
            sample["edge_map"][0].numpy(),
            split_name,
            out_path,
        )
        preview_paths[split_name] = str(out_path.resolve())

    train_loader, val_loader = build_dataloaders(cfg.data, seed=cfg.seed)
    _, test_loader = build_dataloaders(cfg.data, seed=cfg.seed, eval_split="test")
    batch = next(iter(train_loader))

    top_clipped_slices = sorted(records, key=lambda item: item["clipped_fraction"], reverse=True)[:10]
    top_hu_extreme_slices = sorted(records, key=lambda item: max(abs(item["hu_min"]), abs(item["hu_max"])), reverse=True)[
        :10
    ]

    save_json(
        {
            "index_dir": str(index_dir.resolve()),
            "manifest": manifest,
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "test_batches": len(test_loader),
            "batch_size": cfg.data.batch_size,
            "sample_paths": [str(x) for x in batch["path"][: min(3, len(batch["path"]))]],
            "preview_paths": preview_paths,
            "top_clipped_slices": top_clipped_slices,
            "top_hu_extreme_slices": top_hu_extreme_slices,
        },
        out_dir / "dataset_stats.json",
    )

    print(f"Prepared {len(records)} slices across {scan_stats['patients']} patients")
    print(f"Index files written to {index_dir.resolve()}")
    print(f"Preview written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
