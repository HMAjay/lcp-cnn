# Local setup (Windows / WSL / Linux / macOS) — no Colab

Research use only. Not a medical diagnosis.

## 1) Get the code

```bash
git clone https://github.com/HMAjay/lcp-cnn.git
cd lcp-cnn
```

## 2) Create venv + install

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**WSL / Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 3) Prepare data on disk

Put a full processed tree somewhere, e.g. `D:\data\lidc\processed` or `~/data/lidc/processed`:

```text
processed/
  volumes/*.nii.gz
  metadata/physical_nodules/*.json
  metadata/lidc/**/*.json
  manifests/dataset_manifest.json
  manifests/patient_splits.json
```

If manifests are missing:

```bash
python scripts/run_local.py build-manifests --data-root /path/to/processed
```

Set:

```bash
# bash
export DATA_ROOT=/path/to/processed

# PowerShell
$env:DATA_ROOT="D:\data\lidc\processed"
```

## 4) Verify labels

```bash
python scripts/run_local.py verify --data-root "$DATA_ROOT"
```

## 5) Train

Uses GPU if available, else CPU (`--device auto`).

```bash
python scripts/run_local.py train \
  --data-root "$DATA_ROOT" \
  --output-dir ./artifacts/checkpoints_v2 \
  --epochs 30 \
  --batch-size 4 \
  --device auto
```

CPU-only / low memory:

```bash
python scripts/run_local.py train \
  --data-root "$DATA_ROOT" \
  --output-dir ./artifacts/checkpoints_v2 \
  --epochs 5 \
  --batch-size 1 \
  --device cpu
```

## 6) Eval (+ copy outputs)

```bash
python scripts/run_local.py eval \
  --data-root "$DATA_ROOT" \
  --checkpoint ./artifacts/checkpoints_v2/best.pt \
  --device cpu \
  --out ./artifacts/checkpoints_v2/test_metrics.json \
  --copy-to ./artifacts/export
```

## 7) Real-data predictions + attention PNGs

```bash
python scripts/run_local.py predict \
  --data-root "$DATA_ROOT" \
  --checkpoint ./artifacts/checkpoints_v2/best.pt \
  --device cpu \
  --out-dir ./artifacts/real_inference
```

## 8) Demo UI

```bash
python scripts/run_local.py demo \
  --checkpoint ./artifacts/checkpoints_v2/best.pt \
  --device cpu \
  --port 7865
```

Open [http://127.0.0.1:7865](http://127.0.0.1:7865)

## One-shot PowerShell example

```powershell
cd "$HOME\OneDrive\Desktop\Lets Code\Major Project Final 1.0\lcp-cnn"
.\.venv\Scripts\Activate.ps1
$env:DATA_ROOT="D:\data\lidc\processed"

python scripts/run_local.py verify --data-root $env:DATA_ROOT
python scripts/run_local.py train --data-root $env:DATA_ROOT --output-dir .\artifacts\checkpoints_v2 --device auto
python scripts/run_local.py eval --data-root $env:DATA_ROOT --checkpoint .\artifacts\checkpoints_v2\best.pt --device cpu --out .\artifacts\checkpoints_v2\test_metrics.json
python scripts/run_local.py predict --data-root $env:DATA_ROOT --checkpoint .\artifacts\checkpoints_v2\best.pt --device cpu
```

## Notes

- Local training without NVIDIA GPU will be very slow for 3D CNN+Swin; Colab GPU is still recommended for full 30-epoch runs.
- `best.pt` and `test_metrics.json` land under `./artifacts/`.
- Keep dataset / checkpoints out of git (already gitignored patterns for `*.pt` / data).
