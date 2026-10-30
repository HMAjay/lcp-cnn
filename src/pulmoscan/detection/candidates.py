"""Connected-component candidate extraction from Stage-1 probability maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage


@dataclass
class NoduleCandidate:
    centroid_zyx: list[float]
    bbox_zyx: list[int]  # z0,y0,x0,z1,y1,x1 (exclusive end)
    volume_voxels: int
    score: float
    series_uid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probability_to_candidates(
    prob: np.ndarray,
    *,
    threshold: float = 0.5,
    min_voxels: int = 8,
    max_candidates: int = 50,
    series_uid: str = "",
) -> list[NoduleCandidate]:
    """Threshold probability map → connected components → ranked candidates."""
    binary = np.asarray(prob >= float(threshold), dtype=np.uint8)
    if not binary.any():
        return []
    labeled, n = ndimage.label(binary)
    if n == 0:
        return []

    candidates: list[NoduleCandidate] = []
    for lab in range(1, n + 1):
        coords = np.argwhere(labeled == lab)  # (N, 3) in z,y,x
        n_vox = int(coords.shape[0])
        if n_vox < int(min_voxels):
            continue
        # Score = mean probability inside component
        zs, ys, xs = coords[:, 0], coords[:, 1], coords[:, 2]
        score = float(prob[zs, ys, xs].mean())
        centroid = [
            float(zs.mean()),
            float(ys.mean()),
            float(xs.mean()),
        ]
        z0, y0, x0 = (int(zs.min()), int(ys.min()), int(xs.min()))
        z1, y1, x1 = (int(zs.max()) + 1, int(ys.max()) + 1, int(xs.max()) + 1)
        candidates.append(
            NoduleCandidate(
                centroid_zyx=centroid,
                bbox_zyx=[z0, y0, x0, z1, y1, x1],
                volume_voxels=n_vox,
                score=score,
                series_uid=series_uid,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[: int(max_candidates)]


def voxel_zyx_to_patient_mm(
    centroid_zyx: Sequence[float],
    affine: np.ndarray,
) -> list[float]:
    """Inverse of patient_to_voxel_zyx: (z,y,x) → patient (x,y,z) mm."""
    z, y, x = float(centroid_zyx[0]), float(centroid_zyx[1]), float(centroid_zyx[2])
    ijk1 = np.array([x, y, z, 1.0], dtype=np.float64)
    xyz = affine @ ijk1
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def candidates_to_stage2_records(
    candidates: list[NoduleCandidate],
    *,
    series_uid: str,
    patient_id: str,
    affine: np.ndarray,
    split: str = "",
) -> list[dict[str, Any]]:
    """Serialize detections into Stage-2-compatible nodule records."""
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        rows.append(
            {
                "series_uid": series_uid,
                "patient_id": patient_id,
                "split": split,
                "physical_id": f"{series_uid}-det{i}",
                "detection_id": f"{series_uid}-det{i}",
                "centroid_patient_mm": voxel_zyx_to_patient_mm(c.centroid_zyx, affine),
                "centroid_zyx": c.centroid_zyx,
                "bbox_zyx": c.bbox_zyx,
                "volume_voxels": c.volume_voxels,
                "detection_score": c.score,
                "source": "stage1_unet",
            }
        )
    return rows
