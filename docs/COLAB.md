# Training on Google Colab

This Cloud Agent VM is **CPU-only**. Train the hybrid CNN–Swin model on **Google Colab with a GPU**.

## Steps

1. Upload your ~3 GB `data/processed/` folder to Google Drive  
   (layout from `docs/PREPROCESSING_README.md`).
2. Open [`notebooks/PulmoScan_Colab_Train.ipynb`](../notebooks/PulmoScan_Colab_Train.ipynb) in Colab.
3. Runtime → Change runtime type → **GPU**.
4. Run setup → mount Drive → set `DATA_ROOT` → train → evaluate.
5. Download `best.pt` and run the local Gradio demo:

```bash
PULMOSCAN_CHECKPOINT=./best.pt python3 -m pulmoscan.inference.demo_app --port 7865
```

## Colab tips

- T4 16 GB: `BATCH_SIZE=4`, `voi_size=48`, AMP on (default).
- If OOM: set batch size to 2, or reduce `backbone.base_channels` / Swin `embed_dim` in `configs/model/hybrid_cnn_swin.yaml`.
- Keep patient-level splits from `manifests/patient_splits.json` — do not reshuffle by nodule.
- Checkpoints under Drive survive Colab disconnects if `OUTPUT_DIR` is on Drive.
