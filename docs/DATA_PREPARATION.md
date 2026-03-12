# Data Preparation

## What is included

Included metadata:

- `metadata/lidc_index/patient_splits.json`
- `metadata/lidc_index/split_manifest.json`
- `metadata/lidc_raw/README.md`

These are the patient-level split artifacts that are small enough for a public repository.

## What is intentionally omitted

The following are not included because they are too large for a normal GitHub repository:

- Raw LIDC-IDRI DICOM files
- Full per-slice index files such as `full_index.jsonl`, `train_index.json`, `val_index.json`, and `test_index.json`

Those large index files can be regenerated from raw data with the preparation script.

## Expected raw-data layout

Place the LIDC-IDRI DICOM tree under:

```text
metadata/lidc_raw/
  LIDC-IDRI-0001/
    Study-*/
      Series-*/
        *.dcm
```

## Regenerate prepared indices

From the package root:

```bash
python code/scripts/prepare_lidc.py --config code/configs/train_lidc_hybrid_medium.yaml
```

This will regenerate the full prepared index files using the included patient-level split logic.
