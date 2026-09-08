"""Unit tests for model shapes, labels, and metrics."""

from __future__ import annotations

import numpy as np
import torch

from pulmoscan.classification.model import build_model
from pulmoscan.data.io import crop_voi, window_normalize
from pulmoscan.data.labels import _label_from_score
from pulmoscan.data.synthetic import generate_synthetic_dataset
from pulmoscan.evaluation.metrics import compute_binary_metrics
from pulmoscan.data.dataset import make_datasets
from pulmoscan.common.utils import load_yaml


def test_label_rule():
    assert _label_from_score(4.0, 4.0, 2.0) == 1
    assert _label_from_score(1.5, 4.0, 2.0) == 0
    assert _label_from_score(3.0, 4.0, 2.0) is None


def test_crop_and_window():
    vol = np.zeros((64, 64, 64), dtype=np.float32)
    vol[20:30, 20:30, 20:30] = 100
    voi = crop_voi(vol, (25, 25, 25), (32, 32, 32))
    assert voi.shape == (32, 32, 32)
    wn = window_normalize(voi, (-1000, 400), True)
    assert wn.min() >= 0 and wn.max() <= 1


def test_model_forward_and_attention():
    cfg = {
        "in_channels": 1,
        "backbone": {"base_channels": 16, "out_channels": 64},
        "swin": {
            "embed_dim": 64,
            "depths": [1, 1],
            "num_heads": [2, 4],
            "window_size": [2, 2, 2],
            "dropout": 0.0,
            "drop_path": 0.0,
        },
        "head": {"hidden_dim": 32, "dropout": 0.0},
    }
    model = build_model(cfg)
    x = torch.randn(2, 1, 48, 48, 48)
    p = model(x)
    assert p.shape == (2,)
    assert torch.all((p >= 0) & (p <= 1))
    logits = model(x, return_logits=True)
    assert logits.shape == (2,)
    attn = model.attention_map(x)
    assert attn.ndim == 4 and attn.shape[0] == 2


def test_metrics():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    m = compute_binary_metrics(y, p)
    assert 0.0 <= m["auc"] <= 1.0
    assert "sensitivity" in m


def test_synthetic_dataset_pipeline(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "synth", n_train=4, n_val=2, n_test=2, seed=0)
    data_cfg = {
        "voi_size": [32, 32, 32],
        "hu_window": [-1000, 400],
        "normalize": True,
        "labeling": {"malignant_min": 4.0, "benign_max": 2.0, "min_confidence": "medium"},
        "augmentation": {"train": {"flip_prob": 0.5}},
        "num_workers": 0,
    }
    datasets = make_datasets(root, data_cfg)
    assert "train" in datasets and len(datasets["train"]) >= 1
    item = datasets["train"][0]
    assert item["image"].shape[0] == 1
    assert item["label"] in (0.0, 1.0)
