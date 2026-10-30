"""Smoke tests for review pipeline demo helpers."""

from __future__ import annotations

import numpy as np

from pulmoscan.data.synthetic import generate_synthetic_dataset
from pulmoscan.inference.demo_pipeline import list_demo_cases, run_case, load_stage2
from pulmoscan.common.utils import get_device
from pathlib import Path


def test_list_and_run_case(tmp_path):
    root = generate_synthetic_dataset(tmp_path / "synth", n_train=2, n_val=1, n_test=1, seed=3)
    cases = list_demo_cases(root)
    assert len(cases) >= 1
    device = get_device("cpu")
    # Force untrained path so the test does not pick up a repo checkpoint
    model, cfg, _, untrained = load_stage2(str(tmp_path / "missing.pt"), device)
    assert untrained is True
    out = Path(tmp_path / "out")
    result = run_case(
        cases[0],
        data_root=root,
        model=model,
        model_cfg=cfg,
        device=device,
        untrained=untrained,
        out_dir=out,
    )
    assert "P(malignant)" in result["md"]
    assert result["ct"].ndim == 3 and result["overlay"].shape == result["ct"].shape
    assert 0.0 <= result["meta"]["prob"] <= 1.0
    assert Path(result["meta"]["figure"]).is_file()


def test_viz_panels():
    from pulmoscan.inference.viz import render_simple_panels

    vol = np.zeros((32, 32, 32), dtype=np.float32) - 800
    lung = np.zeros((32, 32, 32), dtype=np.uint8)
    nod = np.zeros((32, 32, 32), dtype=np.uint8)
    lung[8:24, 8:24, 8:24] = 1
    nod[14:18, 14:18, 14:18] = 1
    vol[nod > 0] = -100
    ct, ov, zoom = render_simple_panels(vol, lung, nod, slice_z=16)
    assert ct.shape[2] == 3 and ov.shape[2] == 3 and zoom.ndim == 3
