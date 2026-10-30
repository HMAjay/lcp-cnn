"""Patch dataset for Stage-1 3D U-Net training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from pulmoscan.data.io import crop_voi, load_nifti, patient_to_voxel_zyx, window_normalize
from pulmoscan.data.labels import CONFIDENCE_RANK, load_json, _strip_processed_prefix
from pulmoscan.detection.masks import ensure_consensus_mask, resolve_mask_path


@dataclass
class SeriesSample:
    series_uid: str
    patient_id: str
    split: str
    volume_path: str
    mask_path: str
    centroids_zyx: list[list[float]]


def _load_splits(root: Path) -> dict[str, str]:
    """Map patient_id → split name."""
    path = root / "manifests" / "patient_splits.json"
    if not path.is_file():
        return {}
    raw = load_json(path)
    out: dict[str, str] = {}
    for split, patients in raw.items():
        if not isinstance(patients, list):
            continue
        for pid in patients:
            out[str(pid)] = split
    return out


def build_series_samples(
    data_root: str | Path,
    *,
    min_confidence: str = "low",
    default_radius_mm: float = 6.0,
    build_missing_masks: bool = True,
) -> list[SeriesSample]:
    """One sample per CT series with a consensus nodule mask."""
    root = Path(data_root)
    manifest = load_json(root / "manifests" / "dataset_manifest.json")
    splits = _load_splits(root)
    min_rank = CONFIDENCE_RANK.get(str(min_confidence).lower(), 0)
    samples: list[SeriesSample] = []

    for item in manifest.get("series", []):
        series_uid = str(item["series_uid"])
        patient_id = str(item.get("patient_id", series_uid))
        split = str(item.get("split") or splits.get(patient_id, "train"))
        vol_rel = _strip_processed_prefix(item.get("volume") or f"volumes/{series_uid}.nii.gz")
        phys_rel = _strip_processed_prefix(
            item.get("physical_nodules") or f"metadata/physical_nodules/{series_uid}.json"
        )
        vol_path = root / vol_rel
        if not vol_path.is_file():
            continue

        mask_path = resolve_mask_path(root, series_uid)
        if not mask_path.is_file():
            if not build_missing_masks:
                continue
            try:
                mask_path = ensure_consensus_mask(
                    root,
                    series_uid,
                    vol_rel,
                    phys_rel,
                    default_radius_mm=default_radius_mm,
                    min_confidence=min_confidence,
                )
            except Exception:  # noqa: BLE001
                continue

        centroids: list[list[float]] = []
        phys_path = root / phys_rel
        if phys_path.is_file():
            phys = load_json(phys_path)
            vol, affine = load_nifti(vol_path, mmap=True)
            for nod in phys.get("physical_nodules", []):
                conf = str(nod.get("confidence", "low")).lower()
                if CONFIDENCE_RANK.get(conf, 0) < min_rank:
                    continue
                c = nod.get("representative_centroid_patient_mm") or nod.get("centroid_patient_mm")
                if not c or len(c) < 3:
                    continue
                zyx = patient_to_voxel_zyx(c, affine)
                centroids.append([float(zyx[0]), float(zyx[1]), float(zyx[2])])
            del vol

        samples.append(
            SeriesSample(
                series_uid=series_uid,
                patient_id=patient_id,
                split=split,
                volume_path=str(vol_path),
                mask_path=str(mask_path),
                centroids_zyx=centroids,
            )
        )
    return samples


class NodulePatchDataset(Dataset):
    """Random / centered 3D patches of (image, mask) for U-Net training.

    Positive patches are centered near known nodule centroids (with jitter).
    Negative patches are random crops that preferably avoid known centroids.
    """

    def __init__(
        self,
        samples: list[SeriesSample],
        *,
        patch_size: tuple[int, int, int] = (64, 64, 64),
        hu_window: tuple[float, float] = (-1000.0, 400.0),
        normalize: bool = True,
        patches_per_series: int = 4,
        positive_fraction: float = 0.75,
        augment: bool = False,
        augment_cfg: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.patch_size = tuple(int(x) for x in patch_size)
        self.hu_window = (float(hu_window[0]), float(hu_window[1]))
        self.normalize = normalize
        self.patches_per_series = max(1, int(patches_per_series))
        self.positive_fraction = float(positive_fraction)
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.rng = np.random.default_rng(seed)
        # Flat index → (series_idx, patch_slot)
        self._index = [
            (i, j)
            for i in range(len(samples))
            for j in range(self.patches_per_series)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def _augment_pair(self, img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.augment_cfg
        if self.rng.random() < float(cfg.get("flip_prob", 0.5)):
            axis = int(self.rng.integers(0, 3))
            img = np.flip(img, axis=axis).copy()
            msk = np.flip(msk, axis=axis).copy()
        if self.rng.random() < float(cfg.get("rotate90_prob", 0.5)):
            k = int(self.rng.integers(0, 4))
            axes = [(1, 2), (0, 2), (0, 1)][int(self.rng.integers(0, 3))]
            img = np.rot90(img, k=k, axes=axes).copy()
            msk = np.rot90(msk, k=k, axes=axes).copy()
        scale_rng = cfg.get("intensity_scale", [1.0, 1.0])
        shift_rng = cfg.get("intensity_shift", [0.0, 0.0])
        scale = float(self.rng.uniform(float(scale_rng[0]), float(scale_rng[1])))
        shift = float(self.rng.uniform(float(shift_rng[0]), float(shift_rng[1])))
        img = np.clip(img * scale + shift, 0.0, 1.0).astype(np.float32)
        return img, msk

    def _pick_center(self, sample: SeriesSample, shape: tuple[int, int, int]) -> np.ndarray:
        d, h, w = self.patch_size
        want_pos = self.rng.random() < self.positive_fraction and bool(sample.centroids_zyx)
        if want_pos:
            c = np.asarray(sample.centroids_zyx[int(self.rng.integers(0, len(sample.centroids_zyx)))])
            jitter = self.rng.uniform(-d * 0.25, d * 0.25, size=3)
            center = c + jitter
        else:
            # Random location; try a few times to land away from nodules if any
            center = None
            for _ in range(8):
                cand = np.array(
                    [
                        self.rng.uniform(d / 2, max(shape[0] - d / 2, d / 2 + 1e-3)),
                        self.rng.uniform(h / 2, max(shape[1] - h / 2, h / 2 + 1e-3)),
                        self.rng.uniform(w / 2, max(shape[2] - w / 2, w / 2 + 1e-3)),
                    ],
                    dtype=np.float64,
                )
                if not sample.centroids_zyx:
                    center = cand
                    break
                dists = [float(np.linalg.norm(cand - np.asarray(c))) for c in sample.centroids_zyx]
                if min(dists) > max(d, h, w) * 0.6:
                    center = cand
                    break
                center = cand
            assert center is not None
        # Clamp so crop stays mostly inside
        center[0] = float(np.clip(center[0], 0, shape[0] - 1))
        center[1] = float(np.clip(center[1], 0, shape[1] - 1))
        center[2] = float(np.clip(center[2], 0, shape[2] - 1))
        return center.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        series_i, _ = self._index[idx]
        sample = self.samples[series_i]
        vol, _ = load_nifti(sample.volume_path, mmap=True)
        mask, _ = load_nifti(sample.mask_path, mmap=True)
        if mask.shape != vol.shape:
            raise ValueError(
                f"Mask/volume shape mismatch for {sample.series_uid}: "
                f"{mask.shape} vs {vol.shape}"
            )
        center = self._pick_center(sample, tuple(int(s) for s in vol.shape))
        img = crop_voi(vol, center, self.patch_size, fill=float(self.hu_window[0]))
        msk = crop_voi(mask.astype(np.float32), center, self.patch_size, fill=0.0)
        img = window_normalize(img, self.hu_window, self.normalize)
        msk = (msk > 0.5).astype(np.float32)
        del vol, mask
        if self.augment:
            img, msk = self._augment_pair(img, msk)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(msk)).unsqueeze(0).float(),
            "series_uid": sample.series_uid,
            "patient_id": sample.patient_id,
        }


def make_detection_datasets(
    data_root: str | Path,
    data_cfg: dict[str, Any],
) -> dict[str, NodulePatchDataset]:
    samples = build_series_samples(
        data_root,
        min_confidence=str(data_cfg.get("min_confidence", "low")),
        default_radius_mm=float(data_cfg.get("default_radius_mm", 6.0)),
        build_missing_masks=bool(data_cfg.get("build_missing_masks", True)),
    )
    patch_size = tuple(data_cfg.get("patch_size", [64, 64, 64]))
    hu_window = tuple(data_cfg.get("hu_window", [-1000, 400]))
    normalize = bool(data_cfg.get("normalize", True))
    patches_per = int(data_cfg.get("patches_per_series", 4))
    pos_frac = float(data_cfg.get("positive_fraction", 0.75))
    aug = data_cfg.get("augmentation", {})

    out: dict[str, NodulePatchDataset] = {}
    for split, do_aug in (("train", True), ("validation", False), ("test", False)):
        split_samples = [s for s in samples if s.split == split]
        if not split_samples:
            continue
        out[split] = NodulePatchDataset(
            split_samples,
            patch_size=patch_size,
            hu_window=hu_window,
            normalize=normalize,
            patches_per_series=patches_per if split == "train" else max(1, patches_per // 2),
            positive_fraction=pos_frac if split == "train" else 1.0,
            augment=do_aug and bool(aug.get("train")),
            augment_cfg=aug.get("train", {}) if do_aug else {},
            seed=0 if split == "train" else 1,
        )
    return out


def dump_series_index(samples: list[SeriesSample], path: str | Path) -> None:
    rows = [
        {
            "series_uid": s.series_uid,
            "patient_id": s.patient_id,
            "split": s.split,
            "n_centroids": len(s.centroids_zyx),
            "mask_path": s.mask_path,
        }
        for s in samples
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
