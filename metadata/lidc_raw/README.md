Place the full LIDC-IDRI DICOM tree here before running real-data experiments.

Expected layout:

```text
data/lidc_raw/
  LIDC-IDRI-0001/
    Study-*/
      Series-*/
        *.dcm
```

Then run:

```bash
conda activate egldm-lidc
python scripts/prepare_lidc.py --config configs/train.yaml
```

The preparation step writes deterministic patient-level split files under `data/lidc_index/`.
