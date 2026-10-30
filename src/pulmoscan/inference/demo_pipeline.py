"""Review demo: CT → lung + nodule segmentation view → malignancy prediction.

Uses existing processed/synthetic data (volumes, nodule masks, optional lung masks)
and a Stage-2 hybrid CNN–Swin checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch

from pulmoscan.classification.model import build_model
from pulmoscan.common.utils import ensure_dir, get_device, load_yaml
from pulmoscan.data.io import crop_voi, load_nifti, patient_to_voxel_zyx, window_normalize
from pulmoscan.data.labels import load_json, _strip_processed_prefix
from pulmoscan.detection.masks import resolve_mask_path, ensure_consensus_mask
from pulmoscan.inference.viz import (
    estimate_lung_mask,
    nodule_centroid_zyx,
    render_pipeline_figure,
    render_simple_panels,
)
from pulmoscan.xai.attention import save_attention_overlay, upsample_attention

DISCLAIMER = "Research use only. Not a medical diagnosis."


def _find_checkpoint(explicit: str | None) -> Path | None:
    """Resolve Stage-2 weights.

    If ``explicit`` is provided, only that path is used (no silent fallback).
    Otherwise search env + common artifact locations.
    """
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    env = os.environ.get("PULMOSCAN_CHECKPOINT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("./artifacts/checkpoints_4gb/best.pt"),
            Path("./artifacts/checkpoints/best.pt"),
            Path("./best.pt"),
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_stage2(checkpoint: str | Path | None, device: torch.device):
    path = _find_checkpoint(str(checkpoint) if checkpoint else None)
    if path is None:
        # Untrained weights — demo still runs; scores are not meaningful
        cfg = load_yaml("configs/model/hybrid_cnn_swin_4gb.yaml")
        model = build_model(cfg).to(device)
        model.eval()
        return model, cfg, None, True
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_cfg") or load_yaml("configs/model/hybrid_cnn_swin_4gb.yaml")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, path, False


def list_demo_cases(data_root: str | Path) -> list[dict[str, Any]]:
    root = Path(data_root)
    manifest_path = root / "manifests" / "dataset_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = load_json(manifest_path)
    cases: list[dict[str, Any]] = []
    for item in manifest.get("series", []):
        series_uid = str(item["series_uid"])
        patient_id = str(item.get("patient_id", series_uid))
        split = str(item.get("split", ""))
        vol_rel = _strip_processed_prefix(item.get("volume") or f"volumes/{series_uid}.nii.gz")
        phys_rel = _strip_processed_prefix(
            item.get("physical_nodules") or f"metadata/physical_nodules/{series_uid}.json"
        )
        vol_path = root / vol_rel
        if not vol_path.is_file():
            continue
        phys_path = root / phys_rel
        n_nodules = 0
        label_hint = ""
        if phys_path.is_file():
            phys = load_json(phys_path)
            nods = phys.get("physical_nodules", [])
            n_nodules = len(nods)
            if nods:
                m = nods[0].get("malignancy_mean")
                if m is not None:
                    label_hint = "malignant" if float(m) >= 4 else "benign" if float(m) <= 2 else "indeterminate"
        cases.append(
            {
                "label": f"{patient_id}  |  {series_uid}  |  {split}  |  nodules={n_nodules}"
                + (f"  |  {label_hint}" if label_hint else ""),
                "series_uid": series_uid,
                "patient_id": patient_id,
                "split": split,
                "volume_path": str(vol_path),
                "physical_path": str(phys_path) if phys_path.is_file() else "",
                "n_nodules": n_nodules,
            }
        )
    return cases


def _load_nodule_mask(root: Path, series_uid: str, vol_shape: tuple[int, ...], vol_rel: str) -> np.ndarray:
    path = resolve_mask_path(root, series_uid)
    if not path.is_file():
        try:
            path = ensure_consensus_mask(root, series_uid, vol_rel)
        except Exception:  # noqa: BLE001
            return np.zeros(vol_shape, dtype=np.uint8)
    mask, _ = load_nifti(path, mmap=False)
    if mask.shape != vol_shape:
        return np.zeros(vol_shape, dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def _load_lung_mask(root: Path, series_uid: str, volume: np.ndarray) -> np.ndarray:
    for cand in (
        root / "lung_masks" / f"{series_uid}.nii.gz",
        root / "lung_masks" / series_uid / "lung.nii.gz",
    ):
        if cand.is_file():
            m, _ = load_nifti(cand, mmap=False)
            if m.shape == volume.shape:
                return (m > 0).astype(np.uint8)
    return estimate_lung_mask(volume)


def _pick_nodule(phys_path: str) -> dict[str, Any] | None:
    if not phys_path or not Path(phys_path).is_file():
        return None
    phys = load_json(Path(phys_path))
    nods = phys.get("physical_nodules", [])
    if not nods:
        return None
    # Prefer high confidence
    def rank(n: dict[str, Any]) -> int:
        return {"high": 2, "medium": 1, "low": 0}.get(str(n.get("confidence", "low")).lower(), 0)

    return sorted(nods, key=rank, reverse=True)[0]


def run_case(
    case: dict[str, Any],
    *,
    data_root: Path,
    model: torch.nn.Module,
    model_cfg: dict[str, Any],
    device: torch.device,
    untrained: bool,
    out_dir: Path,
    slice_offset: int = 0,
) -> dict[str, Any]:
    series_uid = case["series_uid"]
    vol, affine = load_nifti(case["volume_path"], mmap=True)
    vol = np.asanyarray(vol)
    lung = _load_lung_mask(data_root, series_uid, vol)
    vol_rel = f"volumes/{series_uid}.nii.gz"
    nodule_mask = _load_nodule_mask(data_root, series_uid, vol.shape, vol_rel)

    nod = _pick_nodule(case.get("physical_path", ""))
    truth = None
    malignancy_mean = None
    if nod is not None:
        malignancy_mean = nod.get("malignancy_mean")
        if malignancy_mean is not None:
            m = float(malignancy_mean)
            truth = "malignant" if m >= 4 else "benign" if m <= 2 else "indeterminate"
        centroid = nod.get("representative_centroid_patient_mm") or nod.get("centroid_patient_mm")
        if centroid:
            center_zyx = patient_to_voxel_zyx(centroid, affine)
        else:
            center_zyx = np.array(nodule_centroid_zyx(nodule_mask), dtype=np.float32)
    else:
        center_zyx = np.array(nodule_centroid_zyx(nodule_mask), dtype=np.float32)

    cz = int(round(float(center_zyx[0]))) + int(slice_offset)
    cz = int(np.clip(cz, 0, vol.shape[0] - 1))

    voi_size = tuple(model_cfg.get("voi_size", [32, 32, 32]))
    if len(voi_size) != 3:
        voi_size = (32, 32, 32)
    hu_window = (-1000.0, 400.0)
    voi_hu = crop_voi(vol, center_zyx, voi_size, fill=hu_window[0])
    voi = window_normalize(voi_hu, hu_window, True)

    x = torch.from_numpy(np.ascontiguousarray(voi)).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = float(model(x).item())
        attn = model.attention_map(x)
        attn_up = upsample_attention(attn, tuple(voi.shape)).cpu().numpy()[0]

    tier = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
    pred = "malignant" if prob >= 0.5 else "benign"

    fig_path = out_dir / f"{series_uid}_pipeline.png"
    rgb, _ = render_pipeline_figure(
        vol,
        lung,
        nodule_mask,
        slice_z=cz,
        title=f"{case['patient_id']}  {series_uid}",
        prob=prob,
        truth_label=truth,
        out_path=fig_path,
        hu_window=hu_window,
    )
    ct_img, overlay_img, zoom_img = render_simple_panels(
        vol, lung, nodule_mask, slice_z=cz, hu_window=hu_window
    )
    attn_path = out_dir / f"{series_uid}_attention.png"
    save_attention_overlay(
        voi,
        attn_up,
        attn_path,
        title=f"P(malignant)={prob:.3f}",
    )

    warn = ""
    if untrained:
        warn = (
            "\n\n⚠️ **No Stage-2 checkpoint found** — scores use randomly initialized weights. "
            "Point `--checkpoint` at your Colab `best.pt` for real predictions.\n"
        )

    md = (
        f"### Case `{case['patient_id']}`\n"
        f"- **Series:** `{series_uid}`\n"
        f"- **Split:** `{case.get('split', '')}`\n"
        f"- **Nodule center (zyx voxels):** "
        f"({center_zyx[0]:.1f}, {center_zyx[1]:.1f}, {center_zyx[2]:.1f})\n"
        f"- **Shown axial slice:** `{cz}`\n\n"
        f"### Malignancy prediction (Stage 2)\n"
        f"- **P(malignant):** **{prob:.3f}**\n"
        f"- **Risk tier:** **{tier}**\n"
        f"- **Predicted class (≥0.5):** **{pred}**\n"
    )
    if truth is not None:
        md += f"- **LIDC label (from annotations):** **{truth}**"
        if malignancy_mean is not None:
            md += f" (mean malignancy={float(malignancy_mean):.2f})"
        md += "\n"
    md += (
        f"\n### What you are seeing\n"
        f"1. **Input CT** — axial slice through the nodule\n"
        f"2. **Segmentation** — teal = lung field, red = segmented nodule\n"
        f"3. **Close-up** — zoomed nodule region\n"
        f"4. **Attention** — Stage-2 focus map on the VOI\n"
        f"\n*{DISCLAIMER}*"
        f"{warn}"
    )

    meta = {
        "series_uid": series_uid,
        "patient_id": case["patient_id"],
        "prob": prob,
        "tier": tier,
        "pred": pred,
        "truth": truth,
        "slice_z": cz,
        "figure": str(fig_path),
        "attention": str(attn_path),
    }
    return {
        "md": md,
        "figure_rgb": rgb,
        "ct": ct_img,
        "overlay": overlay_img,
        "zoom": zoom_img,
        "attention": str(attn_path),
        "meta": meta,
    }


def build_app(
    data_root: str | Path,
    checkpoint: str | None = None,
    device_pref: str = "auto",
) -> gr.Blocks:
    data_root = Path(data_root)
    device = get_device(device_pref)
    model, model_cfg, ckpt_path, untrained = load_stage2(checkpoint, device)
    out_dir = ensure_dir("./artifacts/demo_pipeline")
    cases = list_demo_cases(data_root)
    case_labels = [c["label"] for c in cases] or ["(no cases found — check --data-root)"]
    by_label = {c["label"]: c for c in cases}

    ckpt_note = f"Stage-2 checkpoint: `{ckpt_path}`" if ckpt_path else "Stage-2 checkpoint: **missing** (random weights)"
    with gr.Blocks(title="PulmoScan AI — Pipeline Demo") as demo:
        gr.Markdown(
            "# PulmoScan AI — Review Demo\n"
            "Select a CT case → see **lung + nodule segmentation** → get **malignancy risk**.\n\n"
            f"Data root: `{data_root.resolve()}`  \n"
            f"{ckpt_note}  \n"
            f"*{DISCLAIMER}*"
        )
        with gr.Row():
            case_dd = gr.Dropdown(choices=case_labels, value=case_labels[0], label="CT case")
            slice_off = gr.Slider(-10, 10, value=0, step=1, label="Axial slice offset")
            btn = gr.Button("Run pipeline", variant="primary")

        out_md = gr.Markdown()
        with gr.Row():
            img_ct = gr.Image(label="1 · Input CT", type="numpy")
            img_overlay = gr.Image(label="2 · Lung + nodule segmentation", type="numpy")
            img_zoom = gr.Image(label="3 · Nodule close-up", type="numpy")
        with gr.Row():
            img_fig = gr.Image(label="Combined figure", type="numpy")
            img_attn = gr.Image(label="4 · Stage-2 attention on VOI", type="filepath")

        def on_run(label: str, offset: float):
            if label not in by_label:
                return (
                    "No cases available under the data root.",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            result = run_case(
                by_label[label],
                data_root=data_root,
                model=model,
                model_cfg=model_cfg,
                device=device,
                untrained=untrained,
                out_dir=out_dir,
                slice_offset=int(offset),
            )
            with (out_dir / "last_result.json").open("w", encoding="utf-8") as f:
                json.dump(result["meta"], f, indent=2)
            return (
                result["md"],
                result["ct"],
                result["overlay"],
                result["zoom"],
                result["figure_rgb"],
                result["attention"],
            )

        btn.click(
            on_run,
            inputs=[case_dd, slice_off],
            outputs=[out_md, img_ct, img_overlay, img_zoom, img_fig, img_attn],
        )
        # Auto-run first case on load for review convenience
        demo.load(
            on_run,
            inputs=[case_dd, slice_off],
            outputs=[out_md, img_ct, img_overlay, img_zoom, img_fig, img_attn],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="PulmoScan full pipeline review demo")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PULMOSCAN_DATA_ROOT", "./data/synthetic"),
        help="Processed or synthetic data root",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("PULMOSCAN_CHECKPOINT", ""),
        help="Stage-2 best.pt (optional)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PULMOSCAN_DEMO_PORT", "7867")))
    parser.add_argument("--device", default=os.environ.get("PULMOSCAN_DEVICE", "auto"))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not (data_root / "manifests" / "dataset_manifest.json").is_file():
        raise SystemExit(
            f"No dataset manifest under {data_root}. "
            "Run: python -m pulmoscan.data.synthetic --out ./data/synthetic"
        )

    ckpt = args.checkpoint or None
    app = build_app(data_root, ckpt, args.device)
    print(f"Launching pipeline demo on http://127.0.0.1:{args.port}")
    app.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
