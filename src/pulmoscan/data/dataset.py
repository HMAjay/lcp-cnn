"""PyTorch dataset for nodule VOIs — lazy load by default (4GB-friendly)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from pulmoscan.data.io import crop_voi, load_nifti, patient_to_voxel_zyx, window_normalize
from pulmoscan.data.labels import NoduleSample, build_samples


class NoduleVOIDataset(Dataset):
    """Crops 3D VOIs around physical nodules and returns (1, D, H, W) tensors.

    Lazy by default:
    - no volumes loaded in __init__
    - each __getitem__ memory-maps NIfTI, crops VOI, discards volume
    - optional tiny LRU cache of cropped VOIs only (not full CTs)
    """

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
        voi_cache_size: int = 0,
        soft_malignancy: bool = True,
    ) -> None:
        self.samples = samples
        self.voi_size = tuple(int(x) for x in voi_size)
        self.hu_window = (float(hu_window[0]), float(hu_window[1]))
        self.normalize = normalize
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.volume_cache = volume_cache
        self.voi_cache_size = int(voi_cache_size)
        self.soft_malignancy = bool(soft_malignancy)
        self._volume_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._voi_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_key(self, sample: NoduleSample) -> str:
        return f"{sample.series_uid}:{sample.physical_id}:{self.voi_size}"

    def _get_volume(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if self.volume_cache and path in self._volume_cache:
            return self._volume_cache[path]
        # mmap=True: avoid eager full-volume RAM copy at open
            vol, affine = load_nifti(path, mmap=True)
            if self.volume_cache:
                # Caching full volumes is expensive; only when explicitly enabled
                self._volume_cache[path] = (vol, affine)
            return vol, affine

    def _jitter_center(self, center: np.ndarray) -> np.ndarray:
        jitter = float(self.augment_cfg.get("centroid_jitter_mm", 0.0))
        if not self.augment or jitter <= 0:
            return center
        noise = np.random.uniform(-jitter, jitter, size=3).astype(np.float32)
        return center + noise

    def _augment(self, voi: np.ndarray) -> np.ndarray:
        cfg = self.augment_cfg
        if np.random.rand() < float(cfg.get("flip_prob", 0.5)):
            axis = np.random.randint(0, 3)
            voi = np.flip(voi, axis=axis).copy()
        if np.random.rand() < float(cfg.get("rotate90_prob", 0.5)):
            k = np.random.randint(0, 4)
            axes = [(1, 2), (0, 2), (0, 1)][np.random.randint(0, 3)]
            voi = np.rot90(voi, k=k, axes=axes).copy()
        noise_std = float(cfg.get("gaussian_noise_std", 0.0))
        if noise_std > 0:
            voi = voi + np.random.normal(0.0, noise_std, size=voi.shape).astype(np.float32)
        scale_rng = cfg.get("intensity_scale", [1.0, 1.0])
        shift_rng = cfg.get("intensity_shift", [0.0, 0.0])
        scale = np.random.uniform(float(scale_rng[0]), float(scale_rng[1]))
        shift = np.random.uniform(float(shift_rng[0]), float(shift_rng[1]))
        voi = np.clip(voi * scale + shift, 0.0, 1.0).astype(np.float32)
        return voi

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        key = self._cache_key(sample)

        if self.voi_cache_size > 0 and key in self._voi_cache and not self.augment:
            voi = self._voi_cache[key]
            self._voi_cache.move_to_end(key)
        else:
            vol, affine = self._get_volume(sample.volume_path)
            center = patient_to_voxel_zyx(sample.centroid_patient_mm, affine)
            center = self._jitter_center(center)
            # Fill OOB with HU window floor so normalization maps padding to ~0
            voi = crop_voi(vol, center, self.voi_size, fill=float(self.hu_window[0]))
            voi = window_normalize(voi, self.hu_window, self.normalize)
            # Drop volume reference ASAP (lazy: do not keep full CT)
            del vol
            if self.augment:
                voi = self._augment(voi)
            elif self.voi_cache_size > 0:
                self._voi_cache[key] = voi
                while len(self._voi_cache) > self.voi_cache_size:
                    self._voi_cache.popitem(last=False)

        tensor = torch.from_numpy(np.ascontiguousarray(voi)).unsqueeze(0).float()
        mal_label = 0.0 if sample.label < 0 else float(sample.label)
        # LIDC 1–5 → [0, 1]. |score-3|/2 down-weights indeterminate nodules.
        soft = float(np.clip((float(sample.malignancy_mean) - 1.0) / 4.0, 0.0, 1.0))
        agree = float(min(abs(float(sample.malignancy_mean) - 3.0) / 2.0, 1.0))
        mal_target = soft if self.soft_malignancy else mal_label
        mal_weight = agree if self.soft_malignancy else (0.0 if sample.label < 0 else 1.0)
        concepts = torch.tensor(
            [sample.texture, sample.spiculation, sample.margin, sample.sphericity],
            dtype=torch.float32,
        )
        return {
            "image": tensor,
            "label": torch.tensor(mal_label, dtype=torch.float32),
            "malignancy_target": torch.tensor(mal_target, dtype=torch.float32),
            "malignancy_weight": torch.tensor(mal_weight, dtype=torch.float32),
            "eval_mask": torch.tensor(0.0 if sample.label < 0 else 1.0, dtype=torch.float32),
            "concepts": concepts,
            "attr_mask": torch.tensor(sample.attr_mask, dtype=torch.float32),
            "subsolid": torch.tensor(float(sample.subsolid), dtype=torch.float32),
            "uncertain": torch.tensor(1.0 if sample.is_uncertain else 0.0, dtype=torch.float32),
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
        include_uncertain=bool(labeling.get("include_uncertain", False)),
        subsolid_texture_max=float(labeling.get("subsolid_texture_max", 3.0)),
    )
    voi_size = tuple(data_cfg.get("voi_size", [48, 48, 48]))
    hu_window = tuple(data_cfg.get("hu_window", [-1000, 400]))
    normalize = bool(data_cfg.get("normalize", True))
    aug = data_cfg.get("augmentation", {})
    volume_cache = bool(data_cfg.get("volume_cache", False))
    voi_cache_size = int(data_cfg.get("voi_cache_size", 0))

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
            volume_cache=volume_cache,
            voi_cache_size=0 if augment else voi_cache_size,
            soft_malignancy=bool(labeling.get("soft_malignancy", True)),
        )
    return out
