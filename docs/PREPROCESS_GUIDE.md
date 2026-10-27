# Preprocess LIDC-IDRI (TCIA) → PulmoScan `processed/`

**CPU only — no GPU pod needed.** GPU is only for training afterward.

## What preprocessing does

```text
TCIA DICOM (.dcm) + LIDC XML
        │
        ▼
  HU conversion + 1×1×1 mm resample → volumes/*.nii.gz
  Parse XML malignancy / ROIs     → metadata/lidc/*.json
  Cluster reader marks            → metadata/physical_nodules/*.json
  Patient-level split             → manifests/*.json
```

Training then crops nodule VOIs lazily from those volumes (see `--preset 4gb`).

## Before you start

1. Pause TCIA download around **~80–100 Completed** series (not all 1,308).
2. Download **Radiologist Annotations** (~8.6 MB) from TCIA and unzip them, e.g. `D:\lidc\annotations\`.
3. Know your raw folder, e.g.  
   `C:\Users\Lenovo\Downloads\TCIA_LIDC-IDRI_20200921`

## Install (Windows PowerShell / WSL)

```powershell
cd path\to\lcp-cnn
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[preprocess,dev]"
```

## Run preprocess

```powershell
python -m pulmoscan.data.preprocess_lidc `
  --raw-root "C:\Users\Lenovo\Downloads\TCIA_LIDC-IDRI_20200921" `
  --xml-root "D:\lidc\annotations" `
  --out "D:\lidc\processed" `
  --max-series 100
```

- First time / test: keep `--max-series 5` to verify paths.
- Full batch: omit `--max-series` (processes everything under `--raw-root`).
- Already-done series are skipped unless you pass `--no-skip-existing`.

## Verify labels

```powershell
python scripts/run_local.py verify --data-root "D:\lidc\processed"
```

You want many nodules with malignancy ≤2 (benign) or ≥4 (malignant). Score 3 is excluded by design.

## Then train (GPU / Colab)

Upload or mount `D:\lidc\processed` (or copy to Google Drive), then:

```bash
python -m pulmoscan.classification.train \
  --preset 4gb \
  --data-root /path/to/processed \
  --device cuda \
  --output-dir ./artifacts/checkpoints_4gb
```

## After a good preprocess

You may **delete** that batch’s raw DICOM to free disk, keep `processed/` + annotations XML, then download the next 50–100 patients and run preprocess again into the **same** `--out` folder.
