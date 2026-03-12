#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def _make_slice(size: int, patient_idx: int, slice_idx: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    center_x = size * (0.45 + 0.03 * patient_idx)
    center_y = size * (0.48 + 0.02 * math.sin(slice_idx))
    radius_x = size * (0.18 + 0.01 * patient_idx)
    radius_y = size * (0.22 + 0.01 * slice_idx)

    hu = np.full((size, size), -950.0, dtype=np.float32)
    body_mask = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0
    hu[body_mask] = -120.0

    lesion_center_x = center_x + size * 0.06 * math.cos(slice_idx)
    lesion_center_y = center_y - size * 0.05 * math.sin(patient_idx)
    lesion_mask = (((xx - lesion_center_x) / (radius_x * 0.28)) ** 2 + ((yy - lesion_center_y) / (radius_y * 0.22)) ** 2) <= 1.0
    hu[lesion_mask] = 120.0 + 10.0 * patient_idx

    vessel_mask = np.abs(yy - (0.35 * xx + 10 + 2 * slice_idx)) < 2.0
    hu[vessel_mask & body_mask] = 220.0
    return hu


def _write_dicom(path: Path, patient_id: str, study_uid: str, series_uid: str, hu: np.ndarray, slice_idx: int) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    now = datetime.now()
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = slice_idx + 1
    ds.ImagePositionPatient = [0.0, 0.0, float(slice_idx)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [0.7, 0.7]
    ds.SliceThickness = 1.0
    ds.Rows, ds.Columns = hu.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1.0
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    pixel = np.clip(np.round(hu - ds.RescaleIntercept), -32768, 32767).astype(np.int16)
    ds.PixelData = pixel.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)


def _write_invalid_small_dicom(path: Path) -> None:
    hu = np.full((32, 32), -900.0, dtype=np.float32)
    _write_dicom(path, "LIDC-IDRI-BADSMALL", generate_uid(), generate_uid(), hu, slice_idx=0)


def _write_invalid_modality(path: Path) -> None:
    hu = np.full((128, 128), -800.0, dtype=np.float32)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.PatientID = "LIDC-IDRI-BADMR"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = hu.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1.0
    pixel = np.clip(np.round(hu - ds.RescaleIntercept), -32768, 32767).astype(np.int16)
    ds.PixelData = pixel.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path), write_like_original=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small mock LIDC-style DICOM tree for debug runs")
    parser.add_argument("--out_dir", type=str, default="data/mock_lidc_raw")
    parser.add_argument("--patients", type=int, default=4)
    parser.add_argument("--slices_per_patient", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    for patient_idx in range(args.patients):
        patient_id = f"LIDC-IDRI-{patient_idx + 1:04d}"
        study_uid = generate_uid()
        series_uid = generate_uid()
        for slice_idx in range(args.slices_per_patient):
            hu = _make_slice(args.image_size, patient_idx, slice_idx)
            path = out_dir / patient_id / "Study-1" / "Series-1" / f"slice_{slice_idx:03d}.dcm"
            _write_dicom(path, patient_id, study_uid, series_uid, hu, slice_idx=slice_idx)

    _write_invalid_small_dicom(out_dir / "invalid" / "small_slice.dcm")
    _write_invalid_modality(out_dir / "invalid" / "wrong_modality.dcm")

    print(f"Mock DICOM tree written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
