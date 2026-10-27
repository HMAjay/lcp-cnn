# Preprocess LIDC-IDRI (TCIA) → PulmoScan `processed/`

**CPU only — no GPU pod needed.** GPU is only for training afterward.

## Important: chest vs angio is still lung CT

LIDC-IDRI is a **lung / thoracic CT** collection. Folders named `chest` and `angio` under `LIDC-IDRI-XXXX` are usually **different CT series of the same patient** (standard chest CT vs CT angiography), not liver/brain/other organs.

For malignancy training we only need the series that **LIDC XML annotated** (nodule outlines + malignancy scores). This preprocessor:

1. Keeps **CT only** (drops DX/CR X-rays)
2. Keeps series with a matching **LIDC XML** (`--annotated-only` default)
3. Prefers **chest** over **angio** when both exist
4. Keeps **one series per patient** by default

You do **not** need a separate lung-segmentation step for Stage-2 malignancy scoring: we crop nodule VOIs around annotated centroids.

## When to pause TCIA

Pause around **~100–150 Completed** series (you already had ~131 / 10% — good first batch). Do **not** wait for 1,308 on a nearly full C: drive.

## Folder layout we expect

```text
D:\lidc\raw\
  LIDC-IDRI-0001\
    ...\chest\*.dcm
    ...\angio\*.dcm      # ignored unless it's the only annotated series
  LIDC-IDRI-0002\
  ...

D:\lidc\annotations\     # unzipped Radiologist Annotations (~8.6 MB)
  *.xml

D:\lidc\processed\       # output (fresh rebuild OK)
```

`--raw-root` can be `D:\lidc\raw` **or** the TCIA folder  
`C:\Users\Lenovo\Downloads\TCIA_LIDC-IDRI_20200921` if that is where `LIDC-IDRI-XXXX` lives.

## Install

```powershell
cd path\to\lcp-cnn
git pull
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[preprocess,dev]"
```

## Fresh preprocess (recommended)

Wipes `processed/` and rebuilds cleanly:

```powershell
python -m pulmoscan.data.preprocess_lidc `
  --raw-root "D:\lidc\raw" `
  --xml-root "D:\lidc\annotations" `
  --out "D:\lidc\processed" `
  --fresh `
  --max-series 5
```

If that works (no errors, volumes appear), rerun **without** `--max-series`:

```powershell
python -m pulmoscan.data.preprocess_lidc `
  --raw-root "D:\lidc\raw" `
  --xml-root "D:\lidc\annotations" `
  --out "D:\lidc\processed" `
  --fresh
```

## Verify

```powershell
python scripts\run_local.py verify --data-root "D:\lidc\processed"
```

Expect many labeled nodules (malignancy ≤2 benign / ≥4 malignant). Score 3 is excluded on purpose.

## Train (GPU / Colab)

```bash
python -m pulmoscan.classification.train \
  --preset 4gb \
  --data-root /path/to/processed \
  --device cuda \
  --output-dir ./artifacts/checkpoints_4gb
```

## Scale up for better AUC

| Patients in `processed/` | Purpose |
|--------------------------|---------|
| ~100 | First serious train (replace 7-patient 0.76) |
| 200–400 | More stable AUC |
| more batches | Pause TCIA → preprocess → delete that batch’s DICOM → download next |

After a successful preprocess you can delete raw DICOM for those patients to free disk; keep `processed/` + XML.
