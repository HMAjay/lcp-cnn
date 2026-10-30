"""Build / load Stage-1 nodule segmentation masks for processed volumes.

Preferred source: existing ``masks/`` NIfTIs (union across reader sessions).
Fallback: sphere masks around physical-nodule centroids (works with current
preprocess output that stores centroids but not polygon rasters).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from tqdm import tqdm

from pulmoscan.common.utils import ensure_dir
from pulmoscan.data.io import load_nifti, patient_to_voxel_zyx
from pulmoscan.data.labels import load_json


def _write_nifti_zyx(path: Path, volume_zyx: np.ndarray, affine: np.ndarray) -> None:
    """Save (Z,Y,X) array as NIfTI in nibabel (X,Y,Z) layout with given affine."""
    data = np.transpose(volume_zyx.astype(np.uint8, copy=False), (2, 1, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def sphere_mask(
    shape: tuple[int, int, int],
    center_zyx: tuple[float, float, float],
    radius_mm: float,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Binary sphere mask in (Z,Y,X)."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center_zyx
    sz, sy, sx = spacing_zyx
    dist2 = (
        ((zz - cz) * sz) ** 2
        + ((yy - cy) * sy) ** 2
        + ((xx - cx) * sx) ** 2
    )
    return (dist2 <= float(radius_mm) ** 2).astype(np.uint8)


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    # affine maps (X,Y,Z) voxels → patient mm; our array is (Z,Y,X)
    sx = float(np.linalg.norm(affine[:3, 0]))
    sy = float(np.linalg.norm(affine[:3, 1]))
    sz = float(np.linalg.norm(affine[:3, 2]))
    return (sz, sy, sx)


def _estimate_radius_mm(nodule: dict[str, Any], default: float = 6.0) -> float:
    """Prefer diameter_mm / 2 when present; else default sphere radius."""
    for key in ("diameter_mm", "approx_diameter_mm", "mean_diameter_mm"):
        if key in nodule and nodule[key] is not None:
            try:
                d = float(nodule[key])
                if d > 0:
                    return max(d / 2.0, 2.0)
            except (TypeError, ValueError):
                pass
    # Heuristic from n_rois / malignancy not reliable — fixed default
    return float(default)


def load_union_session_masks(
    masks_root: Path,
    series_uid: str,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Union all session NIfTI masks under masks/<series_uid>/ if present."""
    series_dir = masks_root / series_uid
    if not series_dir.is_dir():
        # Also accept a single consensus file
        single = masks_root / f"{series_uid}.nii.gz"
        if single.is_file():
            mask, _ = load_nifti(single, mmap=False)
            if mask.shape != shape:
                return None
            return (mask > 0).astype(np.uint8)
        return None

    union = np.zeros(shape, dtype=np.uint8)
    found = False
    for path in sorted(series_dir.rglob("*.nii.gz")):
        try:
            m, _ = load_nifti(path, mmap=False)
        except Exception:  # noqa: BLE001
            continue
        if m.shape != shape:
            continue
        union |= (m > 0).astype(np.uint8)
        found = True
    return union if found else None


def build_sphere_mask_from_physical(
    vol_shape: tuple[int, int, int],
    affine: np.ndarray,
    physical_doc: dict[str, Any],
    *,
    default_radius_mm: float = 6.0,
    min_confidence: str = "low",
) -> np.ndarray:
    """Rasterize physical nodules as spheres into a consensus mask."""
    from pulmoscan.data.labels import CONFIDENCE_RANK

    min_rank = CONFIDENCE_RANK.get(str(min_confidence).lower(), 0)
    mask = np.zeros(vol_shape, dtype=np.uint8)
    spacing = _spacing_from_affine(affine)
    for nod in physical_doc.get("physical_nodules", []):
        conf = str(nod.get("confidence", "low")).lower()
        if CONFIDENCE_RANK.get(conf, 0) < min_rank:
            continue
        centroid = nod.get("representative_centroid_patient_mm") or nod.get("centroid_patient_mm")
        if not centroid or len(centroid) < 3:
            continue
        center = patient_to_voxel_zyx(centroid, affine)
        radius = _estimate_radius_mm(nod, default=default_radius_mm)
        mask |= sphere_mask(vol_shape, tuple(float(c) for c in center), radius, spacing)
    return mask


def resolve_mask_path(data_root: Path, series_uid: str) -> Path:
    return data_root / "masks" / "consensus" / f"{series_uid}.nii.gz"


def ensure_consensus_mask(
    data_root: str | Path,
    series_uid: str,
    volume_rel: str,
    physical_rel: str | None = None,
    *,
    default_radius_mm: float = 6.0,
    min_confidence: str = "low",
    rebuild: bool = False,
) -> Path:
    """Return path to a consensus mask NIfTI, building sphere fallback if needed."""
    root = Path(data_root)
    out = resolve_mask_path(root, series_uid)
    if out.is_file() and not rebuild:
        return out

    vol_path = root / volume_rel
    vol, affine = load_nifti(vol_path, mmap=True)
    shape = tuple(int(s) for s in vol.shape)

    # Prefer existing reader-session union
    union = load_union_session_masks(root / "masks", series_uid, shape)
    if union is not None and union.any():
        _write_nifti_zyx(out, union, affine)
        return out

    if physical_rel is None:
        physical_rel = f"metadata/physical_nodules/{series_uid}.json"
    phys_path = root / physical_rel
    if not phys_path.is_file():
        raise FileNotFoundError(f"No physical nodules for {series_uid}: {phys_path}")
    phys = load_json(phys_path)
    mask = build_sphere_mask_from_physical(
        shape,
        affine,
        phys,
        default_radius_mm=default_radius_mm,
        min_confidence=min_confidence,
    )
    _write_nifti_zyx(out, mask, affine)
    return out


def build_masks_for_dataset(
    data_root: str | Path,
    *,
    default_radius_mm: float = 6.0,
    min_confidence: str = "low",
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build consensus masks for every series in the dataset manifest."""
    root = Path(data_root)
    manifest = load_json(root / "manifests" / "dataset_manifest.json")
    series_list = manifest.get("series", [])
    built = 0
    skipped = 0
    empty = 0
    for item in tqdm(series_list, desc="build masks"):
        series_uid = item["series_uid"]
        vol_rel = item.get("volume") or f"volumes/{series_uid}.nii.gz"
        phys_rel = item.get("physical_nodules") or f"metadata/physical_nodules/{series_uid}.json"
        try:
            path = ensure_consensus_mask(
                root,
                series_uid,
                vol_rel,
                phys_rel,
                default_radius_mm=default_radius_mm,
                min_confidence=min_confidence,
                rebuild=rebuild,
            )
            m, _ = load_nifti(path, mmap=False)
            if not m.any():
                empty += 1
            built += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            tqdm.write(f"skip {series_uid}: {exc}")
    stats = {"built": built, "skipped": skipped, "empty": empty, "n_series": len(series_list)}
    ensure_dir(root / "masks" / "consensus")
    with (root / "masks" / "consensus" / "build_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage-1 consensus nodule masks")
    parser.add_argument("--data-root", required=True, help="Processed data root")
    parser.add_argument("--radius-mm", type=float, default=6.0, help="Sphere radius fallback")
    parser.add_argument("--min-confidence", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    stats = build_masks_for_dataset(
        args.data_root,
        default_radius_mm=args.radius_mm,
        min_confidence=args.min_confidence,
        rebuild=args.rebuild,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
