"""Unit tests for Stage-1 detection (U-Net, masks, candidates, pipeline)."""

from __future__ import annotations

import numpy as np
import torch

from pulmoscan.common.utils import load_yaml
from pulmoscan.data.synthetic import generate_synthetic_dataset
from pulmoscan.detection.candidates import (
    probability_to_candidates,
    voxel_zyx_to_patient_mm,
)
from pulmoscan.detection.dataset import make_detection_datasets
from pulmoscan.detection.losses import DiceBCELoss, build_seg_loss
from pulmoscan.detection.masks import build_sphere_mask_from_physical, sphere_mask
from pulmoscan.detection.metrics import froc_points, match_detections_to_gt, soft_dice
from pulmoscan.detection.unet3d import UNet3D, build_unet


def test_unet3d_forward_shape():
    model = UNet3D(in_channels=1, base_channels=8, depth=3)
    x = torch.randn(2, 1, 32, 32, 32)
    y = model(x)
    assert y.shape == (2, 1, 32, 32, 32)


def test_unet3d_4gb_config():
    cfg = load_yaml("configs/model/unet3d_4gb.yaml")
    model = build_unet(cfg)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64, 64)


def test_dice_bce_loss_decreases_on_easy_batch():
    loss_fn = build_seg_loss("dice_bce", pos_weight=2.0)
    logits_good = torch.full((1, 1, 8, 8, 8), 5.0)
    logits_bad = torch.full((1, 1, 8, 8, 8), -5.0)
    target = torch.ones(1, 1, 8, 8, 8)
    assert float(loss_fn(logits_good, target)) < float(loss_fn(logits_bad, target))


def test_sphere_mask_and_candidates():
    mask = sphere_mask((32, 32, 32), (16.0, 16.0, 16.0), radius_mm=4.0)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0
    # Soften to a probability peak
    prob = mask.astype(np.float32)
    cands = probability_to_candidates(prob, threshold=0.5, min_voxels=4)
    assert len(cands) == 1
    cz, cy, cx = cands[0].centroid_zyx
    assert abs(cz - 16) < 1.5 and abs(cy - 16) < 1.5 and abs(cx - 16) < 1.5


def test_patient_voxel_roundtrip():
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float64)
    xyz = voxel_zyx_to_patient_mm([10.0, 20.0, 30.0], affine)
    assert xyz == [30.0, 20.0, 10.0]


def test_match_and_froc():
    tp, fp, fn = match_detections_to_gt(
        [[10.0, 10.0, 10.0], [50.0, 50.0, 50.0]],
        [[11.0, 10.0, 10.0]],
        match_distance_mm=5.0,
    )
    assert (tp, fp, fn) == (1, 1, 0)
    froc = froc_points(
        [
            {
                "gt": [[10.0, 10.0, 10.0]],
                "preds": [
                    {"centroid_zyx": [10.5, 10.0, 10.0], "score": 0.9},
                    {"centroid_zyx": [40.0, 40.0, 40.0], "score": 0.2},
                ],
            }
        ]
    )
    assert froc["n_gt"] == 1.0
    assert froc["sens@fp8"] >= 0.99


def test_soft_dice():
    t = np.zeros((8, 8, 8), dtype=np.float32)
    t[2:5, 2:5, 2:5] = 1
    p = t.copy()
    assert soft_dice(p, t) > 0.99


def test_sphere_from_physical():
    affine = np.eye(4, dtype=np.float64)
    phys = {
        "physical_nodules": [
            {
                "confidence": "high",
                "representative_centroid_patient_mm": [20.0, 20.0, 20.0],
                "diameter_mm": 10.0,
            }
        ]
    }
    mask = build_sphere_mask_from_physical((48, 48, 48), affine, phys)
    assert mask.sum() > 0
    assert mask[20, 20, 20] == 1


def test_detection_synthetic_pipeline(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "synth", n_train=3, n_val=1, n_test=1, seed=1)
    data_cfg = {
        "patch_size": [32, 32, 32],
        "hu_window": [-1000, 400],
        "normalize": True,
        "default_radius_mm": 6.0,
        "min_confidence": "low",
        "build_missing_masks": True,
        "patches_per_series": 2,
        "positive_fraction": 1.0,
        "augmentation": {"train": {"flip_prob": 0.5}},
    }
    datasets = make_detection_datasets(root, data_cfg)
    assert "train" in datasets and len(datasets["train"]) >= 1
    item = datasets["train"][0]
    assert item["image"].shape == (1, 32, 32, 32)
    assert item["mask"].shape == (1, 32, 32, 32)
    assert float(item["mask"].max()) >= 0.0

    # One tiny train step
    model = UNet3D(in_channels=1, base_channels=8, depth=3)
    loss_fn = DiceBCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = item["image"].unsqueeze(0)
    y = item["mask"].unsqueeze(0)
    opt.zero_grad()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
