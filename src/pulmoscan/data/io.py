"""Volume I/O, HU windowing, VOI cropping for processed LIDC data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI as float32 array (Z, Y, X) and affine."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    # nibabel is often (X, Y, Z); convert to (D, H, W) = (Z, Y, X)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} from {path}")
    data = np.transpose(data, (2, 1, 0))
    return data, img.affine


def window_normalize(
    volume: np.ndarray,
    hu_window: Sequence[float] = (-1000.0, 400.0),
    normalize: bool = True,
) -> np.ndarray:
    lo, hi = float(hu_window[0]), float(hu_window[1])
    out = np.clip(volume, lo, hi)
    if normalize:
        out = (out - lo) / max(hi - lo, 1e-6)
    return out.astype(np.float32)


def crop_voi(
    volume: np.ndarray,
    center_zyx: Sequence[float],
    size: Sequence[int],
    fill: float = 0.0,
) -> np.ndarray:
    """Crop a VOI of shape (D, H, W) centered at voxel coordinates (z, y, x)."""
    d, h, w = (int(size[0]), int(size[1]), int(size[2]))
    cz, cy, cx = [float(c) for c in center_zyx]
    z0 = int(round(cz - d / 2))
    y0 = int(round(cy - h / 2))
    x0 = int(round(cx - w / 2))

    out = np.full((d, h, w), fill, dtype=np.float32)
    src_z0, src_y0, src_x0 = max(z0, 0), max(y0, 0), max(x0, 0)
    src_z1 = min(z0 + d, volume.shape[0])
    src_y1 = min(y0 + h, volume.shape[1])
    src_x1 = min(x0 + w, volume.shape[2])

    dst_z0, dst_y0, dst_x0 = src_z0 - z0, src_y0 - y0, src_x0 - x0
    dst_z1 = dst_z0 + (src_z1 - src_z0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_z1 > src_z0 and src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = volume[
            src_z0:src_z1, src_y0:src_y1, src_x0:src_x1
        ]
    return out


def patient_to_voxel_zyx(centroid_patient_mm: Sequence[float], affine: np.ndarray) -> np.ndarray:
    """Map patient-space mm (x,y,z) to voxel indices in (Z,Y,X) volume layout."""
    xyz1 = np.array(
        [centroid_patient_mm[0], centroid_patient_mm[1], centroid_patient_mm[2], 1.0],
        dtype=np.float64,
    )
    ijk = np.linalg.inv(affine) @ xyz1  # nibabel voxel order (X, Y, Z)
    x, y, z = ijk[0], ijk[1], ijk[2]
    return np.array([z, y, x], dtype=np.float32)
