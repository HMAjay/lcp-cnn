"""Build dataset_manifest.json + patient_splits.json from a processed LIDC tree.

Use when a processed export has volumes/metadata but missing manifests/.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def _patient_from_phys(phys: dict, series_uid: str) -> str:
    for key in ("patient_id", "PatientID", "patient"):
        if phys.get(key):
            return str(phys[key])
    # fallback: try LIDC-IDRI-xxxx from nested fields
    text = json.dumps(phys)
    m = re.search(r"LIDC-IDRI-\d+", text)
    return m.group(0) if m else series_uid


def build_manifests(
    data_root: str | Path,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    root = Path(data_root)
    vol_dir = root / "volumes"
    phys_dir = root / "metadata" / "physical_nodules"
    lidc_dir = root / "metadata" / "lidc"
    man_dir = root / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)

    if not vol_dir.exists() or not phys_dir.exists():
        raise FileNotFoundError(f"Need {vol_dir} and {phys_dir}")

    volumes = {p.stem.replace(".nii", ""): p for p in vol_dir.glob("*.nii*")}
    # stem for .nii.gz is file.nii → normalize
    volumes = {}
    for p in vol_dir.glob("*.nii*"):
        name = p.name
        if name.endswith(".nii.gz"):
            uid = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            uid = name[: -len(".nii")]
        else:
            uid = p.stem
        volumes[uid] = p

    phys_files = list(phys_dir.glob("*.json"))
    series_rows = []
    patients = set()

    for pf in phys_files:
        series_uid = pf.stem
        if series_uid not in volumes:
            # fuzzy: file name contains uid or uid contains file stem
            match = next((v for u, v in volumes.items() if u == series_uid or series_uid in u), None)
            if match is None:
                continue
            vol_path = match
        else:
            vol_path = volumes[series_uid]

        phys = json.loads(pf.read_text())
        patient_id = _patient_from_phys(phys, series_uid)
        patients.add(patient_id)
        nodules = phys.get("physical_nodules", phys.get("nodules", []))

        # optional lidc path guess
        lidc_path = None
        if lidc_dir.exists():
            # keep relative path if series entry already had one later; else None
            # training code can still find malignancy via nested lidc index when path is set
            # Prefer any explicit field
            lidc_path = phys.get("lidc_metadata_path")

        series_rows.append(
            {
                "series_instance_uid": series_uid,
                "series_uid": series_uid,
                "patient_id": patient_id,
                "volume_path": f"volumes/{vol_path.name}",
                "physical_nodule_metadata_path": f"metadata/physical_nodules/{pf.name}",
                "lidc_metadata_path": lidc_path,
                "num_physical_nodules": len(nodules) if isinstance(nodules, list) else 0,
            }
        )

    if not series_rows:
        raise RuntimeError("No series matched between volumes/ and physical_nodules/")

    # patient-level split
    patient_list = sorted(patients)
    rng = random.Random(seed)
    rng.shuffle(patient_list)
    n = len(patient_list)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n >= 5 else max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 3:
        n_test = 1
        n_train = max(1, n - n_val - n_test)

    split_map = {}
    for i, pid in enumerate(patient_list):
        if i < n_train:
            split_map[pid] = "train"
        elif i < n_train + n_val:
            split_map[pid] = "validation"
        else:
            split_map[pid] = "test"

    for row in series_rows:
        row["split"] = split_map.get(row["patient_id"], "train")

    patient_splits = {
        "schema_version": "1.0-generated",
        "splits": split_map,
        "metadata": {
            "seed": seed,
            "n_patients": n,
            "n_series": len(series_rows),
            "ratios": train_ratio,
            "val": val_ratio,
            "generated_by": "scripts/build_manifests_from_processed.py",
        },
    }
    dataset_manifest = {
        "schema_version": "1.0-generated",
        "n_series": len(series_rows),
        "n_patients": n,
        "series": series_rows,
    }

    splits_path = man_dir / "patient_splits.json"
    manifest_path = man_dir / "dataset_manifest.json"
    splits_path.write_text(json.dumps(patient_splits, indent=2))
    manifest_path.write_text(json.dumps(dataset_manifest, indent=2))

    counts = {"train": 0, "validation": 0, "test": 0}
    for s in split_map.values():
        counts[s] = counts.get(s, 0) + 1

    return {
        "data_root": str(root.resolve()),
        "manifest": str(manifest_path),
        "splits": str(splits_path),
        "n_series": len(series_rows),
        "n_patients": n,
        "patient_split_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Folder containing volumes/ and metadata/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    info = build_manifests(args.data_root, seed=args.seed)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
