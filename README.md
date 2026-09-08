# PulmoScan AI

Hybrid **3D CNN + Swin Transformer** for pulmonary nodule malignancy risk prediction (LIDC-IDRI).

> Research use only. Not a medical diagnosis.

## What this repo includes

| Piece | Role |
|-------|------|
| `src/pulmoscan/classification/` | 3D ResNet-style CNN → Swin3D encoder → sigmoid risk score |
| `src/pulmoscan/data/` | Loaders for your preprocessed LIDC layout + synthetic smoke data |
| `src/pulmoscan/evaluation/` | AUC, sensitivity/specificity, Brier, confusion counts |
| `src/pulmoscan/xai/` | CLS-attention overlay helpers |
| `src/pulmoscan/inference/demo_app.py` | Local Gradio demo |
| `notebooks/PulmoScan_Colab_Train.ipynb` | **GPU training on Google Colab** (recommended) |

Stage-1 detection (3D U-Net) is deferred; this slice scores VOIs from existing physical-nodule metadata.

## Defaults chosen

- **Architecture:** hybrid CNN backbone + Swin Transformer (not pure Swin, not ViT)
- **Labels:** physical nodules; mean LIDC malignancy ≥4 = malignant, ≤2 = benign; score 3 excluded
- **VOI:** 48³ mm at your 1 mm isotropic spacing
- **Training:** Colab GPU for real data; local CPU for synthetic smoke tests

## Data

Point `PULMOSCAN_DATA_ROOT` at your processed tree (~3 GB), matching `PREPROCESSING_README.md`:

```text
data/processed/
  volumes/<SeriesInstanceUID>.nii.gz
  metadata/physical_nodules/<SeriesInstanceUID>.json
  manifests/dataset_manifest.json
  manifests/patient_splits.json
```

## Local setup (CPU smoke + demo)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# synthetic data + 1-epoch smoke train
python -m pulmoscan.data.synthetic --out ./data/synthetic
python -m pulmoscan.classification.train \
  --data-root ./data/synthetic \
  --epochs 1 --batch-size 2 --device cpu \
  --output-dir ./artifacts/checkpoints

# Gradio demo
python -m pulmoscan.inference.demo_app --port 7865
```

## Train on Colab (your ~3 GB LIDC set)

This environment has no GPU. Use Colab:

1. Upload `data/processed/` to Google Drive (~3 GB).
2. Open [`notebooks/PulmoScan_Colab_Train.ipynb`](notebooks/PulmoScan_Colab_Train.ipynb) in Colab (**GPU runtime**).
3. Clone/upload this repo, set `DATA_ROOT`, run train + eval cells.
4. Download `best.pt` and run the local Gradio demo with that checkpoint.

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

## Configs

- `configs/model/hybrid_cnn_swin.yaml` — CNN + Swin capacity
- `configs/data/lidc_processed.yaml` — VOI, HU window, label rule
- `configs/train/default.yaml` — epochs, AMP, early stopping

## Tests

```bash
pytest -q
```

## Docs

- `docs/PRD_LungNoduleAI.md` — full product/engineering PRD
- `docs/PREPROCESSING_README.md` — processed dataset contract
