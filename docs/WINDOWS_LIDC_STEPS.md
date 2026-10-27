# Windows steps — `C:\lidc` (keep old preprocessing)

This guide assumes:

```text
C:\lidc\
  raw\                 # LIDC-IDRI-0001 ... (TCIA DICOM)
  tcia-lidc-xml\       # LIDC annotation XMLs
  processed\           # OLD preprocessing — DO NOT overwrite
  processed_v2\        # NEW output (created by you)
```

If your XML folder name differs slightly, only change `--xml-root`.

**Old `C:\lidc\processed` is never deleted** if you always use `--out C:\lidc\processed_v2`.

---

## A) On your Windows PC (CPU) — code + preprocess

### 1. Pause TCIA if still running

~100–150 Completed series is enough for the first v2 batch.

### 2. Get the code on a **separate branch**

```powershell
cd C:\path\to\lcp-cnn
git fetch origin
git checkout cursor/lidc-fresh-preprocess-1296
git pull origin cursor/lidc-fresh-preprocess-1296
```

If the repo is not cloned yet:

```powershell
cd C:\
git clone https://github.com/HMAjay/lcp-cnn.git
cd lcp-cnn
git checkout cursor/lidc-fresh-preprocess-1296
```

### 3. Python env + preprocess deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[preprocess,dev]"
```

### 4. Smoke-test preprocess (5 series)

```powershell
python -m pulmoscan.data.preprocess_lidc `
  --raw-root "C:\lidc\raw" `
  --xml-root "C:\lidc\tcia-lidc-xml" `
  --out "C:\lidc\processed_v2" `
  --fresh `
  --max-series 5
```

Check that `C:\lidc\processed_v2\volumes\` has a few `.nii.gz` files and that **`C:\lidc\processed` is unchanged**.

### 5. Full preprocess for this batch

```powershell
python -m pulmoscan.data.preprocess_lidc `
  --raw-root "C:\lidc\raw" `
  --xml-root "C:\lidc\tcia-lidc-xml" `
  --out "C:\lidc\processed_v2" `
  --fresh
```

(`--fresh` only wipes `processed_v2`, not `processed`.)

### 6. Verify labels

```powershell
python scripts\run_local.py verify --data-root "C:\lidc\processed_v2"
```

You want many train nodules with malignancy ≤2 or ≥4.

### 7. (Optional) Free disk after success

Only delete raw DICOM for patients you already processed into `processed_v2`. Keep XML and both `processed` + `processed_v2`.

---

## B) On your GPU pod / Colab — training only

Preprocessing stays on the PC. The pod only needs **`processed_v2`** + the repo.

### 1. Copy data to the pod

Upload/sync `C:\lidc\processed_v2` to the pod, e.g.:

```text
/data/lidc/processed_v2
```

(or Google Drive path if using Colab)

### 2. Get the same branch on the pod

```bash
git clone https://github.com/HMAjay/lcp-cnn.git
cd lcp-cnn
git checkout cursor/lidc-fresh-preprocess-1296
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Train (4GB-safe preset)

```bash
python -m pulmoscan.classification.train \
  --preset 4gb \
  --data-root /data/lidc/processed_v2 \
  --device cuda \
  --output-dir ./artifacts/checkpoints_v2_4gb
```

### 4. Evaluate on test split

```bash
python -m pulmoscan.evaluation.evaluate \
  --data-root /data/lidc/processed_v2 \
  --checkpoint ./artifacts/checkpoints_v2_4gb/best.pt \
  --split test \
  --device cuda \
  --out ./artifacts/checkpoints_v2_4gb/test_metrics.json
```

Download `best.pt` + `test_metrics.json` when done.

---

## What runs where

| Step | Machine | GPU? |
|------|---------|------|
| TCIA download | Windows | No |
| `preprocess_lidc` → `processed_v2` | Windows CPU | **No** |
| `verify` | Windows | No |
| `train` / `evaluate` | GPU pod or Colab | **Yes** |

---

## Accuracy path

1. Finish first `processed_v2` (~100+ patients) → train on pod → note real test AUC  
2. Download more TCIA batches into `C:\lidc\raw` → preprocess **without** deleting old v2 volumes (omit `--fresh` on later batches, or merge) → retrain  
3. Keep old `C:\lidc\processed` for comparison only until v2 beats it
