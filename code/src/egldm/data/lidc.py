from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from egldm.utils import ensure_dir, save_json


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_patient_id(ds: pydicom.Dataset, path: Path) -> str:
    patient_id = str(getattr(ds, "PatientID", "")).strip()
    if patient_id:
        return patient_id

    match = re.search(r"LIDC-IDRI-\d+", str(path))
    if match is not None:
        return match.group(0)

    if len(path.parents) >= 2:
        return path.parents[1].name
    return path.parent.name


def _infer_series_uid(ds: pydicom.Dataset, path: Path) -> str:
    series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
    if series_uid:
        return series_uid
    return f"path_series::{path.parent.name}"


def _infer_study_uid(ds: pydicom.Dataset, path: Path) -> str:
    study_uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
    if study_uid:
        return study_uid
    if len(path.parents) >= 3:
        return f"path_study::{path.parents[2].name}"
    return f"path_study::{path.parent.name}"


def _image_position_z(ds: pydicom.Dataset) -> float | None:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return _safe_float(ipp[2])
    return _safe_float(getattr(ds, "SliceLocation", None))


def load_dicom_hu(path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    pixel = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixel * slope + intercept


def scan_lidc_dicom_root(
    dicom_root: str | Path,
    hu_clip_min: float,
    hu_clip_max: float,
    strict_ct_only: bool = True,
    min_slice_size: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(dicom_root)
    all_paths = sorted(root.rglob("*.dcm"))
    if not all_paths:
        raise FileNotFoundError(f"No DICOM files found under {root}")

    stats: dict[str, Any] = {
        "root": str(root.resolve()),
        "candidate_files": len(all_paths),
        "valid_slices": 0,
        "filtered": {
            "non_ct_modality": 0,
            "missing_pixel_data": 0,
            "pixel_decode_failed": 0,
            "non_2d_slice": 0,
            "too_small": 0,
            "invalid_rescale": 0,
            "non_finite_hu": 0,
            "duplicate_sop_instance_uid": 0,
        },
        "anomalies": {
            "missing_patient_id_metadata": 0,
            "missing_series_uid_metadata": 0,
            "missing_study_uid_metadata": 0,
            "missing_instance_number": 0,
            "missing_image_position_z": 0,
        },
    }

    records: list[dict[str, Any]] = []
    seen_sop_uids: set[str] = set()
    hu_mins: list[float] = []
    hu_maxs: list[float] = []
    hu_means: list[float] = []
    clipped_fractions: list[float] = []
    pixel_spacings: list[float] = []

    for path in all_paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
        except Exception:
            stats["filtered"]["pixel_decode_failed"] += 1
            continue

        modality = str(getattr(ds, "Modality", "")).upper().strip()
        if strict_ct_only and modality and modality != "CT":
            stats["filtered"]["non_ct_modality"] += 1
            continue

        if not hasattr(ds, "PixelData"):
            stats["filtered"]["missing_pixel_data"] += 1
            continue

        try:
            pixel = ds.pixel_array.astype(np.float32)
        except Exception:
            stats["filtered"]["pixel_decode_failed"] += 1
            continue

        if pixel.ndim != 2:
            stats["filtered"]["non_2d_slice"] += 1
            continue

        rows, cols = int(pixel.shape[0]), int(pixel.shape[1])
        if min(rows, cols) < int(min_slice_size):
            stats["filtered"]["too_small"] += 1
            continue

        slope = _safe_float(getattr(ds, "RescaleSlope", 1.0))
        intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0))
        if slope is None or intercept is None or abs(slope) < 1e-12:
            stats["filtered"]["invalid_rescale"] += 1
            continue

        hu = pixel * slope + intercept
        if not np.isfinite(hu).all():
            stats["filtered"]["non_finite_hu"] += 1
            continue

        if not str(getattr(ds, "PatientID", "")).strip():
            stats["anomalies"]["missing_patient_id_metadata"] += 1
        if not str(getattr(ds, "SeriesInstanceUID", "")).strip():
            stats["anomalies"]["missing_series_uid_metadata"] += 1
        if not str(getattr(ds, "StudyInstanceUID", "")).strip():
            stats["anomalies"]["missing_study_uid_metadata"] += 1

        instance_number = _safe_int(getattr(ds, "InstanceNumber", None))
        if instance_number is None:
            stats["anomalies"]["missing_instance_number"] += 1

        position_z = _image_position_z(ds)
        if position_z is None:
            stats["anomalies"]["missing_image_position_z"] += 1

        sop_uid = str(getattr(ds, "SOPInstanceUID", "")).strip()
        if sop_uid:
            if sop_uid in seen_sop_uids:
                stats["filtered"]["duplicate_sop_instance_uid"] += 1
                continue
            seen_sop_uids.add(sop_uid)
        else:
            sop_uid = f"path_sop::{path.name}"

        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            sx = _safe_float(spacing[0])
            sy = _safe_float(spacing[1])
            if sx is not None:
                pixel_spacings.append(sx)
            if sy is not None:
                pixel_spacings.append(sy)

        hu_min = float(np.min(hu))
        hu_max = float(np.max(hu))
        hu_mean = float(np.mean(hu))
        clipped_fraction = float(np.mean((hu < hu_clip_min) | (hu > hu_clip_max)))

        hu_mins.append(hu_min)
        hu_maxs.append(hu_max)
        hu_means.append(hu_mean)
        clipped_fractions.append(clipped_fraction)

        records.append(
            {
                "dicom_path": str(path.resolve()),
                "patient_id": _infer_patient_id(ds, path),
                "study_instance_uid": _infer_study_uid(ds, path),
                "series_instance_uid": _infer_series_uid(ds, path),
                "sop_instance_uid": sop_uid,
                "instance_number": instance_number,
                "image_position_z": position_z,
                "rows": rows,
                "cols": cols,
                "rescale_slope": slope,
                "rescale_intercept": intercept,
                "hu_min": hu_min,
                "hu_max": hu_max,
                "hu_mean": hu_mean,
                "clipped_fraction": clipped_fraction,
                "slice_thickness": _safe_float(getattr(ds, "SliceThickness", None)),
                "series_slice_index": -1,
                "series_num_slices": -1,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["patient_id"], record["series_instance_uid"])].append(record)

    ordered_records: list[dict[str, Any]] = []
    for group in grouped.values():
        group.sort(
            key=lambda item: (
                item["image_position_z"] is None,
                0.0 if item["image_position_z"] is None else float(item["image_position_z"]),
                item["instance_number"] is None,
                0 if item["instance_number"] is None else int(item["instance_number"]),
                item["dicom_path"],
            )
        )
        for slice_index, item in enumerate(group):
            item["series_slice_index"] = slice_index
            item["series_num_slices"] = len(group)
            ordered_records.append(item)

    ordered_records.sort(
        key=lambda item: (
            item["patient_id"],
            item["study_instance_uid"],
            item["series_instance_uid"],
            item["series_slice_index"],
            item["dicom_path"],
        )
    )

    stats["valid_slices"] = len(ordered_records)
    stats["patients"] = len({item["patient_id"] for item in ordered_records})
    stats["series"] = len({(item["patient_id"], item["series_instance_uid"]) for item in ordered_records})
    stats["hu_summary"] = {
        "global_min": float(min(hu_mins)) if hu_mins else None,
        "global_max": float(max(hu_maxs)) if hu_maxs else None,
        "mean_of_slice_means": float(np.mean(hu_means)) if hu_means else None,
        "mean_clipped_fraction": float(np.mean(clipped_fractions)) if clipped_fractions else None,
        "max_clipped_fraction": float(max(clipped_fractions)) if clipped_fractions else None,
    }
    stats["pixel_spacing_summary"] = {
        "mean": float(np.mean(pixel_spacings)) if pixel_spacings else None,
        "min": float(min(pixel_spacings)) if pixel_spacings else None,
        "max": float(max(pixel_spacings)) if pixel_spacings else None,
    }

    return ordered_records, stats


def build_patient_splits(
    records: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    if not records:
        raise ValueError("Cannot split an empty DICOM index.")

    patients = sorted({item["patient_id"] for item in records})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_patients = len(patients)
    n_test = int(round(n_patients * max(test_ratio, 0.0)))
    n_val = int(round(n_patients * max(val_ratio, 0.0)))

    if n_patients >= 3 and test_ratio > 0 and n_test == 0:
        n_test = 1
    if n_patients >= 2 and val_ratio > 0 and n_val == 0:
        n_val = 1

    max_reserved = max(0, n_patients - 1)
    while n_val + n_test > max_reserved:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break

    test_patients = patients[:n_test]
    val_patients = patients[n_test : n_test + n_val]
    train_patients = patients[n_test + n_val :]
    if not train_patients:
        raise ValueError("Patient-level split produced an empty train set. Reduce val/test ratios.")

    split_patients = {
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients),
    }
    split_sets = {name: set(items) for name, items in split_patients.items()}

    split_records = {
        "train": [item for item in records if item["patient_id"] in split_sets["train"]],
        "val": [item for item in records if item["patient_id"] in split_sets["val"]],
        "test": [item for item in records if item["patient_id"] in split_sets["test"]],
    }
    return split_records, split_patients


def load_prepared_split(index_dir: str | Path, split: str) -> list[dict[str, Any]]:
    path = Path(index_dir) / f"{split}_index.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared split index: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_prepared_lidc_dataset(
    index_dir: str | Path,
    full_records: list[dict[str, Any]],
    split_records: dict[str, list[dict[str, Any]]],
    split_patients: dict[str, list[str]],
    scan_stats: dict[str, Any],
    split_seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    out_dir = ensure_dir(index_dir)

    full_index_path = out_dir / "full_index.jsonl"
    with full_index_path.open("w", encoding="utf-8") as f:
        for record in full_records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    for split_name, items in split_records.items():
        save_json(items, out_dir / f"{split_name}_index.json")

    patient_split_path = out_dir / "patient_splits.json"
    save_json(split_patients, patient_split_path)

    split_summary: dict[str, Any] = {}
    for split_name, items in split_records.items():
        split_summary[split_name] = {
            "num_patients": len(split_patients.get(split_name, [])),
            "num_slices": len(items),
            "num_series": len({(item["patient_id"], item["series_instance_uid"]) for item in items}),
        }

    manifest = {
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "scan_stats": scan_stats,
        "split_summary": split_summary,
        "paths": {
            "full_index": str(full_index_path.resolve()),
            "patient_splits": str(patient_split_path.resolve()),
            "train_index": str((out_dir / "train_index.json").resolve()),
            "val_index": str((out_dir / "val_index.json").resolve()),
            "test_index": str((out_dir / "test_index.json").resolve()),
        },
    }
    save_json(manifest, out_dir / "split_manifest.json")
    return manifest
