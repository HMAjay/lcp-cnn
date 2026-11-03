"""Score every nodule with the crop centre deliberately moved by a known distance.

Run from the repository root, with the package installed or PYTHONPATH=src:

    python tolerance_study/shift_eval.py \
        --data-root /mnt/c/lidc/processed_v3 \
        --checkpoint /mnt/c/lidc/models/best.pt \
        --out tolerance_preds.csv

Cropping follows NoduleVOIDataset.__getitem__ exactly, with augmentation off. Each nodule
gets its own random directions, and the same directions are reused at every distance, so
the curves compare like with like. No retraining is involved.
"""

from __future__ import annotations

import argparse
import csv
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from pulmoscan.data.io import crop_voi, load_nifti, patient_to_voxel_zyx, window_normalize
from pulmoscan.data.labels import NoduleSample, build_samples

FIELDS = [
    "split", "patient_id", "series_uid", "physical_id", "label", "malignancy_mean", "confidence",
    "shift_mm", "repeat", "offset_x", "offset_y", "offset_z", "prob",
]


def unit_direction(nodule_key: str, repeat: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, zlib.crc32(nodule_key.encode()), repeat])
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def predict_series(samples, predict, *, shifts, repeats, voi_size, hu_window, seed, batch_size) -> list[dict]:
    """Volumes are gzipped, so each one is read once and cropped many times."""
    volume, affine = load_nifti(samples[0].volume_path, mmap=False)
    volume = np.asarray(volume)
    rows: list[dict] = []
    batch: list[np.ndarray] = []
    pending: list[dict] = []

    def flush() -> None:
        if not batch:
            return
        probs = np.asarray(predict(np.stack(batch))).reshape(-1)
        for row, prob in zip(pending, probs):
            row["prob"] = float(prob)
            rows.append(row)
        batch.clear()
        pending.clear()

    for sample in samples:
        key = f"{sample.series_uid}:{sample.physical_id}"
        for shift in shifts:
            for repeat in range(1 if shift == 0 else repeats):
                offset = np.zeros(3) if shift == 0 else unit_direction(key, repeat, seed) * float(shift)
                centre_mm = np.asarray(sample.centroid_patient_mm, dtype=np.float64) + offset
                centre = patient_to_voxel_zyx(centre_mm, affine)
                voi = crop_voi(volume, centre, voi_size, fill=float(hu_window[0]))
                batch.append(window_normalize(voi, hu_window, True))
                pending.append(
                    {
                        "split": sample.split,
                        "patient_id": sample.patient_id,
                        "series_uid": sample.series_uid,
                        "physical_id": sample.physical_id,
                        "label": sample.label,
                        "malignancy_mean": sample.malignancy_mean,
                        "confidence": sample.confidence,
                        "shift_mm": shift,
                        "repeat": repeat,
                        "offset_x": float(offset[0]),
                        "offset_y": float(offset[1]),
                        "offset_z": float(offset[2]),
                    }
                )
                if len(batch) >= batch_size:
                    flush()
    flush()
    return rows


def run(data_root, predict, *, data_cfg, splits, shifts, repeats, seed, batch_size, out) -> int:
    labeling = data_cfg.get("labeling", {})
    samples: list[NoduleSample] = build_samples(
        data_root,
        malignant_min=float(labeling.get("malignant_min", 4.0)),
        benign_max=float(labeling.get("benign_max", 2.0)),
        min_confidence=labeling.get("min_confidence", "medium"),
        use_physical_nodules=bool(labeling.get("use_physical_nodules", True)),
    )
    samples = [s for s in samples if s.split in splits]
    if not samples:
        raise SystemExit(f"No samples for splits {splits} under {data_root}")
    by_series: dict[str, list[NoduleSample]] = defaultdict(list)
    for s in samples:
        by_series[s.series_uid].append(s)

    voi_size = tuple(int(x) for x in data_cfg.get("voi_size", [48, 48, 48]))
    hu_window = tuple(float(x) for x in data_cfg.get("hu_window", [-1000, 400]))
    print(f"{len(samples)} nodules in {len(by_series)} series, splits {splits}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, group in enumerate(by_series.values(), start=1):
            rows = predict_series(
                group, predict, shifts=shifts, repeats=repeats, voi_size=voi_size,
                hu_window=hu_window, seed=seed, batch_size=batch_size,
            )
            writer.writerows(rows)
            f.flush()
            written += len(rows)
            print(f"series {i}/{len(by_series)} done ({written} crops)", flush=True)
    return written


def make_predictor(checkpoint: str, device_pref: str):
    import torch

    from pulmoscan.classification.model import build_model
    from pulmoscan.common.utils import get_device, load_yaml

    device = get_device(device_pref)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg") or load_yaml("configs/model/hybrid_cnn_swin.yaml")
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {checkpoint} (epoch index {ckpt.get('epoch')}) on {device}")

    @torch.no_grad()
    def predict(batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
        return model(x).detach().float().cpu().numpy()

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/lidc_processed.yaml")
    parser.add_argument("--splits", nargs="+", default=["test", "validation"])
    parser.add_argument("--shifts", nargs="+", type=int, default=[0, 2, 4, 8, 12, 16, 20])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="tolerance_preds.csv")
    args = parser.parse_args()

    with open(args.data_config, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    run(
        args.data_root, make_predictor(args.checkpoint, args.device), data_cfg=data_cfg, splits=args.splits,
        shifts=args.shifts, repeats=args.repeats, seed=args.seed, batch_size=args.batch_size, out=args.out,
    )


if __name__ == "__main__":
    main()
