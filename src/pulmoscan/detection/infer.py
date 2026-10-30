"""Stage-1 sliding-window inference → nodule candidates → Stage-2 VOI records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from pulmoscan.common.utils import ensure_dir, get_device, load_yaml
from pulmoscan.data.io import crop_voi, load_nifti, window_normalize
from pulmoscan.detection.candidates import (
    candidates_to_stage2_records,
    probability_to_candidates,
)
from pulmoscan.detection.dataset import build_series_samples
from pulmoscan.detection.metrics import froc_points, match_detections_to_gt
from pulmoscan.detection.unet3d import build_unet


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    *,
    patch_size: tuple[int, int, int],
    overlap: float,
    hu_window: tuple[float, float],
    normalize: bool,
    device: torch.device,
) -> np.ndarray:
    """Gaussian-weighted sliding-window inference over a full CT volume."""
    d, h, w = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
    stride = (
        max(1, int(d * (1.0 - overlap))),
        max(1, int(h * (1.0 - overlap))),
        max(1, int(w * (1.0 - overlap))),
    )
    # Precompute Gaussian weight for blending
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, d, dtype=np.float32),
        np.linspace(-1, 1, h, dtype=np.float32),
        np.linspace(-1, 1, w, dtype=np.float32),
        indexing="ij",
    )
    sigma = 0.5
    weight = np.exp(-0.5 * (zz**2 + yy**2 + xx**2) / (sigma**2)).astype(np.float32)

    out = np.zeros(volume.shape, dtype=np.float32)
    acc = np.zeros(volume.shape, dtype=np.float32)

    z_starts = list(range(0, max(volume.shape[0] - d, 0) + 1, stride[0]))
    y_starts = list(range(0, max(volume.shape[1] - h, 0) + 1, stride[1]))
    x_starts = list(range(0, max(volume.shape[2] - w, 0) + 1, stride[2]))
    if not z_starts:
        z_starts = [0]
    if not y_starts:
        y_starts = [0]
    if not x_starts:
        x_starts = [0]
    # Ensure last window covers the end
    if z_starts[-1] + d < volume.shape[0]:
        z_starts.append(max(volume.shape[0] - d, 0))
    if y_starts[-1] + h < volume.shape[1]:
        y_starts.append(max(volume.shape[1] - h, 0))
    if x_starts[-1] + w < volume.shape[2]:
        x_starts.append(max(volume.shape[2] - w, 0))

    model.eval()
    fill = float(hu_window[0])
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                center = (z0 + d / 2.0, y0 + h / 2.0, x0 + w / 2.0)
                patch = crop_voi(volume, center, (d, h, w), fill=fill)
                patch = window_normalize(patch, hu_window, normalize)
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(patch))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                logits = model(tensor)
                prob = torch.sigmoid(logits).float().cpu().numpy()[0, 0]

                z1, y1, x1 = z0 + d, y0 + h, x0 + w
                # Handle crops that hung off the border (crop_voi pads)
                dst_z0, dst_y0, dst_x0 = max(z0, 0), max(y0, 0), max(x0, 0)
                dst_z1 = min(z1, volume.shape[0])
                dst_y1 = min(y1, volume.shape[1])
                dst_x1 = min(x1, volume.shape[2])
                src_z0, src_y0, src_x0 = dst_z0 - z0, dst_y0 - y0, dst_x0 - x0
                src_z1 = src_z0 + (dst_z1 - dst_z0)
                src_y1 = src_y0 + (dst_y1 - dst_y0)
                src_x1 = src_x0 + (dst_x1 - dst_x0)
                if dst_z1 <= dst_z0 or dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
                    continue
                wgt = weight[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
                out[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] += (
                    prob[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1] * wgt
                )
                acc[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] += wgt

    acc = np.maximum(acc, 1e-6)
    return out / acc


def detect_series(
    model: torch.nn.Module,
    volume_path: str | Path,
    *,
    patch_size: tuple[int, int, int],
    overlap: float,
    hu_window: tuple[float, float],
    normalize: bool,
    threshold: float,
    min_voxels: int,
    max_candidates: int,
    device: torch.device,
    series_uid: str = "",
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Run detection on one volume. Returns (stage2_records, prob_map, affine)."""
    vol, affine = load_nifti(volume_path, mmap=True)
    # Work on a contiguous float32 view for windowing speed on small volumes;
    # for large volumes mmap + crop_voi already copies patches.
    prob = sliding_window_predict(
        model,
        vol,
        patch_size=patch_size,
        overlap=overlap,
        hu_window=hu_window,
        normalize=normalize,
        device=device,
    )
    cands = probability_to_candidates(
        prob,
        threshold=threshold,
        min_voxels=min_voxels,
        max_candidates=max_candidates,
        series_uid=series_uid,
    )
    return cands, prob, affine


def run_detection(
    data_root: str | Path,
    checkpoint: str | Path,
    *,
    split: str | None = "test",
    data_cfg_path: str | Path = "configs/data/detection_4gb.yaml",
    device_pref: str = "auto",
    output_dir: str | Path = "./artifacts/detections",
    threshold: float | None = None,
    save_prob_maps: bool = False,
    evaluate_froc: bool = True,
) -> dict[str, Any]:
    data_cfg = load_yaml(data_cfg_path)
    device = get_device(device_pref)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_cfg") or load_yaml("configs/model/unet3d_4gb.yaml")
    model = build_unet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    patch_size = tuple(data_cfg.get("patch_size", model_cfg.get("patch_size", [64, 64, 64])))
    overlap = float(data_cfg.get("inference", {}).get("overlap", 0.5))
    hu_window = tuple(data_cfg.get("hu_window", [-1000, 400]))
    normalize = bool(data_cfg.get("normalize", True))
    thr = float(
        threshold
        if threshold is not None
        else data_cfg.get("inference", {}).get("threshold", 0.5)
    )
    min_voxels = int(data_cfg.get("inference", {}).get("min_voxels", 8))
    max_cands = int(data_cfg.get("inference", {}).get("max_candidates", 50))

    samples = build_series_samples(
        data_root,
        min_confidence=str(data_cfg.get("min_confidence", "low")),
        default_radius_mm=float(data_cfg.get("default_radius_mm", 6.0)),
        build_missing_masks=bool(data_cfg.get("build_missing_masks", True)),
    )
    if split:
        samples = [s for s in samples if s.split == split]
    if not samples:
        raise RuntimeError(f"No series for split={split} under {data_root}")

    out_dir = ensure_dir(output_dir)
    det_dir = ensure_dir(out_dir / "candidates")
    all_rows: list[dict[str, Any]] = []
    per_scan_froc: list[dict[str, Any]] = []
    match_stats = {"tp": 0, "fp": 0, "fn": 0}

    for sample in tqdm(samples, desc="detect"):
        cands, prob, affine = detect_series(
            model,
            sample.volume_path,
            patch_size=patch_size,
            overlap=overlap,
            hu_window=hu_window,
            normalize=normalize,
            threshold=thr,
            min_voxels=min_voxels,
            max_candidates=max_cands,
            device=device,
            series_uid=sample.series_uid,
        )
        rows = candidates_to_stage2_records(
            cands,
            series_uid=sample.series_uid,
            patient_id=sample.patient_id,
            affine=affine,
            split=sample.split,
        )
        all_rows.extend(rows)

        series_out = {
            "series_uid": sample.series_uid,
            "patient_id": sample.patient_id,
            "split": sample.split,
            "threshold": thr,
            "n_candidates": len(rows),
            "candidates": rows,
        }
        with (det_dir / f"{sample.series_uid}.json").open("w", encoding="utf-8") as f:
            json.dump(series_out, f, indent=2)

        if save_prob_maps:
            import nibabel as nib

            prob_dir = ensure_dir(out_dir / "prob_maps")
            data = np.transpose(prob.astype(np.float32), (2, 1, 0))
            nib.save(
                nib.Nifti1Image(data, affine),
                str(prob_dir / f"{sample.series_uid}.nii.gz"),
            )

        if evaluate_froc:
            preds = [
                {"centroid_zyx": r["centroid_zyx"], "score": r["detection_score"]}
                for r in rows
            ]
            per_scan_froc.append(
                {
                    "gt": sample.centroids_zyx,
                    "preds": preds,
                    "spacing_zyx": (1.0, 1.0, 1.0),
                    "match_distance_mm": float(
                        data_cfg.get("inference", {}).get("match_distance_mm", 15.0)
                    ),
                }
            )
            tp, fp, fn = match_detections_to_gt(
                [r["centroid_zyx"] for r in rows],
                sample.centroids_zyx,
                match_distance_mm=float(
                    data_cfg.get("inference", {}).get("match_distance_mm", 15.0)
                ),
            )
            match_stats["tp"] += tp
            match_stats["fp"] += fp
            match_stats["fn"] += fn

    manifest = {
        "n_series": len(samples),
        "n_candidates": len(all_rows),
        "threshold": thr,
        "split": split,
        "checkpoint": str(checkpoint),
        "candidates": all_rows,
        "match": match_stats,
    }
    if evaluate_froc and per_scan_froc:
        manifest["froc"] = froc_points(per_scan_froc)

    with (out_dir / "detections_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also write a Stage-2-friendly physical_nodules-like tree for optional re-scoring
    phys_out = ensure_dir(out_dir / "detected_nodules")
    by_series: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_series.setdefault(row["series_uid"], []).append(row)
    for series_uid, nods in by_series.items():
        doc = {
            "series_uid": series_uid,
            "physical_nodules": [
                {
                    "physical_id": n["physical_id"],
                    "confidence": "detection",
                    "representative_centroid_patient_mm": n["centroid_patient_mm"],
                    "malignancy_mean": None,
                    "detection_score": n["detection_score"],
                    "source": "stage1_unet",
                    "observations": [],
                }
                for n in nods
            ],
            "ambiguous_observations": [],
            "unmatched_observations": [],
        }
        with (phys_out / f"{series_uid}.json").open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 detect nodules → VOI candidates")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--data-config", default="configs/data/detection_4gb.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="./artifacts/detections")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save-prob-maps", action="store_true")
    parser.add_argument("--no-froc", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("PULMOSCAN_DATA_ROOT", "./data/processed")
    split = None if args.split == "all" else args.split
    result = run_detection(
        data_root,
        args.checkpoint,
        split=split,
        data_cfg_path=args.data_config,
        device_pref=args.device,
        output_dir=args.output_dir,
        threshold=args.threshold,
        save_prob_maps=args.save_prob_maps,
        evaluate_froc=not args.no_froc,
    )
    summary = {
        "n_series": result["n_series"],
        "n_candidates": result["n_candidates"],
        "match": result.get("match"),
        "froc": result.get("froc"),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
