"""Generate a tiny synthetic processed LIDC-like dataset for smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def _write_nifti(path: Path, volume: np.ndarray, spacing: float = 1.0) -> None:
    # Store as (X, Y, Z) for nibabel; our loader transposes to (Z, Y, X)
    data = np.transpose(volume.astype(np.float32), (2, 1, 0))
    affine = np.diag([spacing, spacing, spacing, 1.0]).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_volume(malignant: bool, size: int = 96, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    vol = rng.normal(-800, 40, size=(size, size, size)).astype(np.float32)
    # lung-ish ball
    zz, yy, xx = np.ogrid[:size, :size, :size]
    center = size // 2
    lung = (zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2 <= (size * 0.35) ** 2
    vol[lung] = rng.normal(-700, 50, size=int(lung.sum()))

    # nodule near center with slight offset
    offset = rng.integers(-8, 9, size=3)
    cz, cy, cx = (center + offset).tolist()
    radius = 6 if malignant else 4
    nodule = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    intensity = -100 if malignant else -200
    vol[nodule] = rng.normal(intensity, 30, size=int(nodule.sum()))
    if malignant:
        # spiculation-ish brighter rim noise
        rim = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= (radius + 2) ** 2) & ~nodule
        vol[rim] = rng.normal(-50, 40, size=int(rim.sum()))

    # patient coords with identity affine: voxel (x,y,z) == mm
    # our volume layout is (Z,Y,X); centroid patient mm is (x,y,z)
    centroid_patient_mm = [float(cx), float(cy), float(cz)]
    return vol, centroid_patient_mm


def generate_synthetic_dataset(
    out_root: str | Path,
    n_train: int = 8,
    n_val: int = 2,
    n_test: int = 2,
    seed: int = 42,
) -> Path:
    root = Path(out_root)
    for sub in (
        "volumes",
        "masks",
        "metadata/lidc",
        "metadata/physical_nodules",
        "lung_masks",
        "lung_qc",
        "manifests",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}
    series_items = []
    counts = {"train": n_train, "validation": n_val, "test": n_test}
    idx = 0

    for split, n in counts.items():
        for i in range(n):
            patient_id = f"SYN{idx:03d}"
            series_uid = f"1.2.syn.{idx}"
            malignant = bool(rng.integers(0, 2))
            vol, centroid = _make_volume(malignant=malignant, size=96, seed=seed + idx)
            vol_path = root / "volumes" / f"{series_uid}.nii.gz"
            _write_nifti(vol_path, vol)

            malignancy = 5.0 if malignant else 1.0
            phys = {
                "series_uid": series_uid,
                "physical_nodules": [
                    {
                        "physical_id": f"{series_uid}-n0",
                        "confidence": "high",
                        "representative_centroid_patient_mm": centroid,
                        "malignancy_mean": malignancy,
                        "observations": [
                            {
                                "session_index": 0,
                                "characteristics": {"malignancy": malignancy},
                            },
                            {
                                "session_index": 1,
                                "characteristics": {"malignancy": malignancy},
                            },
                        ],
                    }
                ],
                "ambiguous_observations": [],
                "unmatched_observations": [],
            }
            with (root / "metadata" / "physical_nodules" / f"{series_uid}.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(phys, f, indent=2)

            splits[split].append(patient_id)
            series_items.append(
                {
                    "series_uid": series_uid,
                    "patient_id": patient_id,
                    "split": split,
                    "volume": f"volumes/{series_uid}.nii.gz",
                    "physical_nodules": f"metadata/physical_nodules/{series_uid}.json",
                }
            )
            idx += 1

    with (root / "manifests" / "patient_splits.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    manifest = {
        "n_series": len(series_items),
        "n_patients": len(series_items),
        "splits": {k: len(v) for k, v in splits.items()},
        "series": series_items,
    }
    with (root / "manifests" / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic processed LIDC-like data")
    parser.add_argument("--out", default="./data/synthetic", help="Output root")
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-val", type=int, default=2)
    parser.add_argument("--n-test", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = generate_synthetic_dataset(args.out, args.n_train, args.n_val, args.n_test, args.seed)
    print(f"Wrote synthetic dataset to {root.resolve()}")


if __name__ == "__main__":
    main()
