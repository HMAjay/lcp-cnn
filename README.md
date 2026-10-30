# PulmoScan AI

Two-stage pipeline for pulmonary nodule malignancy risk (LIDC-IDRI):

1. **Stage 1** — 3D U-Net detects / segments candidate nodules → VOI centroids  
2. **Stage 2** — Hybrid **3D CNN + Swin Transformer** scores each VOI

> Research use only. Not a medical diagnosis.

## What this repo includes

| Piece | Role |
|-------|------|
| `src/pulmoscan/detection/` | Stage-1 3D U-Net, Dice+BCE, sliding-window infer → candidates |
| `src/pulmoscan/classification/` | Stage-2 CNN → Swin3D → sigmoid malignancy risk |
| `src/pulmoscan/data/` | LIDC preprocess/loaders + synthetic smoke data |
| `src/pulmoscan/evaluation/` | AUC, sensitivity/specificity, Brier, confusion counts |
| `src/pulmoscan/xai/` | CLS-attention overlay helpers |
| `src/pulmoscan/inference/demo_app.py` | Local Gradio demo |
| `notebooks/PulmoScan_Colab_Train.ipynb` | **GPU training on Google Colab** (recommended) |

## Defaults chosen

- **Architecture:** hybrid CNN backbone + Swin Transformer (not pure Swin, not ViT)
- **Labels:** physical nodules; mean LIDC malignancy ≥4 = malignant, ≤2 = benign; score 3 excluded
- **VOI (4GB preset):** 32³ with lazy mmap load (`num_workers=0`); full preset uses 48³
- **Training:** Colab GPU for real data; local CPU for synthetic smoke tests

## Data

Point `PULMOSCAN_DATA_ROOT` at your processed tree, matching `PREPROCESSING_README.md`:

```text
data/processed/
  volumes/<SeriesInstanceUID>.nii.gz
  metadata/physical_nodules/<SeriesInstanceUID>.json
  manifests/dataset_manifest.json
  manifests/patient_splits.json
```

### Preprocess from TCIA (CPU, no GPU)

After downloading a LIDC-IDRI batch with TCIA Data Retriever + the annotations ZIP:

```bash
pip install -e ".[preprocess]"
python -m pulmoscan.data.preprocess_lidc \
  --raw-root /path/to/TCIA_LIDC-IDRI_... \
  --xml-root /path/to/lidc_annotations \
  --out /path/to/processed \
  --max-series 100
```

Full steps: [`docs/PREPROCESS_GUIDE.md`](docs/PREPROCESS_GUIDE.md).

## Local setup (CPU smoke + demo)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# synthetic data + 1-epoch smoke trains
python -m pulmoscan.data.synthetic --out ./data/synthetic

# Stage 1 — detect
python -m pulmoscan.detection.train \
  --data-root ./data/synthetic \
  --epochs 1 --batch-size 1 --device cpu \
  --output-dir ./artifacts/detection
python -m pulmoscan.detection.infer \
  --data-root ./data/synthetic \
  --checkpoint ./artifacts/detection/best.pt \
  --split test --device cpu \
  --output-dir ./artifacts/detections

# Stage 2 — classify (uses physical-nodule centroids; or detections above)
python -m pulmoscan.classification.train \
  --data-root ./data/synthetic \
  --epochs 1 --batch-size 2 --device cpu \
  --output-dir ./artifacts/checkpoints

# Gradio demo
python -m pulmoscan.inference.demo_app --port 7865
```

Or: `make smoke-detect` / `make smoke-train`.

## Train on Colab (4GB-safe preset)

Optimized for ≤4GB VRAM / Colab: lazy NIfTI mmap, no DataLoader workers, small model, AMP, grad accum 8, batch size 1.

1. Upload `data/processed/` to Google Drive (~3 GB).
2. Open [`notebooks/PulmoScan_Colab_Train.ipynb`](notebooks/PulmoScan_Colab_Train.ipynb) in Colab (**GPU runtime**).
3. Clone/upload this repo, set `DATA_ROOT`, run train + eval cells.
4. Download `best.pt` and run the local Gradio demo with that checkpoint.

```bash
# after pip install -e . and setting DATA_ROOT to your processed tree
python -m pulmoscan.classification.train \
  --preset 4gb \
  --data-root "$DATA_ROOT" \
  --device cuda \
  --output-dir ./artifacts/checkpoints_4gb
```

```bash
PULMOSCAN_CHECKPOINT=./best.pt python -m pulmoscan.inference.demo_app --port 7865
```

## Evaluate

```bash
python -m pulmoscan.evaluation.evaluate \
  --data-root "$PULMOSCAN_DATA_ROOT" \
  --checkpoint ./artifacts/checkpoints/best.pt \
  --split test
```

## Stage 1 on real processed LIDC

Consensus masks are built automatically (reader-session union if `masks/<series>/` exists; otherwise spheres around physical-nodule centroids):

```bash
python -m pulmoscan.detection.masks --data-root "$PULMOSCAN_DATA_ROOT"
python -m pulmoscan.detection.train --preset 4gb --data-root "$PULMOSCAN_DATA_ROOT" --device cuda
python -m pulmoscan.detection.infer \
  --data-root "$PULMOSCAN_DATA_ROOT" \
  --checkpoint ./artifacts/detection/best.pt \
  --split test --device cuda
```

Detections land in `artifacts/detections/` (`detections_manifest.json`, per-series candidates, Stage-2-shaped `detected_nodules/`).

## Configs

| Stage | Preset | Model | Data | Train |
|-------|--------|-------|------|-------|
| 2 | **4gb** (default CLI) | `configs/model/hybrid_cnn_swin_4gb.yaml` | `configs/data/lidc_4gb.yaml` | `configs/train/colab_4gb.yaml` |
| 2 | default | `configs/model/hybrid_cnn_swin.yaml` | `configs/data/lidc_processed.yaml` | `configs/train/default.yaml` |
| 1 | **4gb** | `configs/model/unet3d_4gb.yaml` | `configs/data/detection_4gb.yaml` | `configs/train/detection_4gb.yaml` |
| 1 | default | `configs/model/unet3d.yaml` | `configs/data/detection.yaml` | `configs/train/detection.yaml` |

Use `--preset default` for larger patches / capacity when you have more VRAM.

## Tests

```bash
pytest -q
```

## Docs

- `docs/PRD_LungNoduleAI.md` — full product/engineering PRD
- `docs/PREPROCESSING_README.md` — processed dataset contract
