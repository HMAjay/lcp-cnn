"""PyTorch dataset for nodule VOIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from pulmoscan.data.io import crop_voi, load_nifti, patient_to_voxel_zyx, window_normalize
from pulmoscan.data.labels import NoduleSample, build_samples


class NoduleVOIDataset(Dataset):
    """Crops 3D VOIs around physical nodules and returns (1, D, H, W) tensors."""

    def __init__(
        self,
        samples: list[NoduleSample],
        *,
        voi_size: tuple[int, int, int] = (48, 48, 48),
        hu_window: tuple[float, float] = (-1000.0, 400.0),
        normalize: bool = True,
        augment: bool = False,
        augment_cfg: dict[str, Any] | None = None,
        volume_cache: bool = False,
    ) -> None:
        self.samples = samples
        self.voi_size = tuple(int(x) for x in voi_size)
        self.hu_window = (float(hu_window[0]), float(hu_window[1]))
        self.normalize = normalize
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.volume_cache = volume_cache
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_volume(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if self.volume_cache and path in self._cache:
            return self._cache[path]
        vol, affine = load_nifti(path)
        vol = window_normalize(vol, self.hu_window, self.normalize)
        if self.volume_cache:
            self._cache[path] = (vol, affine)
        return vol, affine

    def _augment(self, voi: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg
        if np.random.rand() < float(cfg.get("flip_prob", 0.5)):
            axis = np.random.randint(0, 3)
            voi = np.flip(voi, axis=axis).copy()
        if np.random.rand() < float(cfg.get("rotate90_prob", 0.5)):
            k = np.random.randint(0, 4)
            axes = [(1, 2), (0, 2), (0, 1)][np.random.randint(0, 3)]
            voi = np.rot90(voi, k=k, axes=axes).copy()
        scale_rng = cfg.get("intensity_scale", [1.0, 1.0])
        shift_rng = cfg.get("intensity_shift", [0.0, 0.0])
        scale = np.random.uniform(float(scale_rng[0]), float(scale_rng[1]))
        shift = np.random.uniform(float(shift_rng[0]), float(shift_rng[1]))
        voi = np.clip(voi * scale + shift, 0.0, 1.0).astype(np.float32)
        return voi

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        vol, affine = self._get_volume(sample.volume_path)
        center = patient_to_voxel_zyx(sample.centroid_patient_mm, affine)
        voi = crop_voi(vol, center, self.voi_size, fill=0.0)
        if self.augment:
            voi = self._augment(voi)
        tensor = torch.from_numpy(voi).unsqueeze(0).float()  # (1, D, H, W)
        return {
            "image": tensor,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "series_uid": sample.series_uid,
            "physical_id": sample.physical_id,
            "patient_id": sample.patient_id,
            "malignancy_mean": sample.malignancy_mean,
        }


def make_datasets(
    data_root: str | Path,
    data_cfg: dict[str, Any],
) -> dict[str, NoduleVOIDataset]:
    labeling = data_cfg.get("labeling", {})
    samples = build_samples(
        data_root,
        malignant_min=float(labeling.get("malignant_min", 4.0)),
        benign_max=float(labeling.get("benign_max", 2.0)),
        min_confidence=labeling.get("min_confidence", "medium"),
        use_physical_nodules=bool(labeling.get("use_physical_nodules", True)),
    )
    voi_size = tuple(data_cfg.get("voi_size", [48, 48, 48]))
    hu_window = tuple(data_cfg.get("hu_window", [-1000, 400]))
    normalize = bool(data_cfg.get("normalize", True))
    aug = data_cfg.get("augmentation", {})

    out: dict[str, NoduleVOIDataset] = {}
    for split, augment in (("train", True), ("validation", False), ("test", False)):
        split_samples = [s for s in samples if s.split == split]
        if not split_samples:
            continue
        out[split] = NoduleVOIDataset(
            split_samples,
            voi_size=voi_size,
            hu_window=hu_window,
            normalize=normalize,
            augment=augment,
            augment_cfg=aug.get("train", {}) if augment else {},
        )
    return out
