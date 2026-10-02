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
- **VOI (4GB preset):** 32³ with lazy mmap load (`num_workers=0`); full preset uses 48³
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

## Configs

| Preset | Model | Data | Train |
|--------|-------|------|-------|
| **4gb** (default CLI) | `configs/model/hybrid_cnn_swin_4gb.yaml` | `configs/data/lidc_4gb.yaml` | `configs/train/colab_4gb.yaml` |
| default | `configs/model/hybrid_cnn_swin.yaml` | `configs/data/lidc_processed.yaml` | `configs/train/default.yaml` |

Use `--preset default` for the larger 48³ / higher-capacity setup when you have more VRAM.

## Tests

```bash
pytest -q
```

## Docs

- `docs/PRD_LungNoduleAI.md` — full product/engineering PRD
- `docs/PREPROCESSING_README.md` — processed dataset contract
