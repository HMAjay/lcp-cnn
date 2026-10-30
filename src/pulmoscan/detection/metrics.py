"""Segmentation and detection metrics (Dice, FROC helpers)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def soft_dice(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float | None = 0.5,
    smooth: float = 1.0,
) -> float:
    """Dice on flattened arrays. If threshold is set, binarize probs first."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    if threshold is not None:
        p = (p >= float(threshold)).astype(np.float64)
    inter = float((p * t).sum())
    return float((2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))


def match_detections_to_gt(
    pred_centroids_zyx: Sequence[Sequence[float]],
    gt_centroids_zyx: Sequence[Sequence[float]],
    *,
    match_distance_mm: float = 15.0,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[int, int, int]:
    """Greedy match predictions to GT. Returns (tp, fp, fn)."""
    preds = [np.asarray(c, dtype=np.float64) for c in pred_centroids_zyx]
    gts = [np.asarray(c, dtype=np.float64) for c in gt_centroids_zyx]
    spacing = np.asarray(spacing_zyx, dtype=np.float64)
    matched_gt: set[int] = set()
    tp = 0
    for p in preds:
        best_j = -1
        best_d = float("inf")
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            d = float(np.linalg.norm((p - g) * spacing))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_d <= match_distance_mm:
            matched_gt.add(best_j)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def froc_points(
    per_scan: Sequence[dict],
    *,
    score_key: str = "score",
    fp_targets: Sequence[float] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> dict[str, float]:
    """Compute FROC-style sensitivities at target FP/scan rates.

    Each entry in ``per_scan``:
      {"gt": [[z,y,x], ...], "preds": [{"centroid_zyx": [...], "score": float}, ...]}
    Matching uses 15 mm (caller should pre-filter or we match by descending score).
    """
    # Collect all predictions with scan index
    all_preds: list[tuple[float, int, list[float]]] = []
    n_gt = 0
    n_scans = len(per_scan)
    for si, scan in enumerate(per_scan):
        n_gt += len(scan.get("gt", []))
        for pred in scan.get("preds", []):
            all_preds.append(
                (
                    float(pred.get(score_key, pred.get("prob", 0.0))),
                    si,
                    list(pred["centroid_zyx"]),
                )
            )
    all_preds.sort(key=lambda x: x[0], reverse=True)

    # Greedy cumulative matching
    matched: dict[int, set[int]] = {i: set() for i in range(n_scans)}
    tp_cum = 0
    fp_cum = 0
    curve: list[tuple[float, float]] = [(0.0, 0.0)]  # (fp/scan, sens)

    for score, si, centroid in all_preds:
        gts = per_scan[si].get("gt", [])
        spacing = tuple(per_scan[si].get("spacing_zyx", (1.0, 1.0, 1.0)))
        best_j = -1
        best_d = float("inf")
        sp = np.asarray(spacing, dtype=np.float64)
        p = np.asarray(centroid, dtype=np.float64)
        for j, g in enumerate(gts):
            if j in matched[si]:
                continue
            d = float(np.linalg.norm((p - np.asarray(g, dtype=np.float64)) * sp))
            if d < best_d:
                best_d = d
                best_j = j
        match_mm = float(per_scan[si].get("match_distance_mm", 15.0))
        if best_j >= 0 and best_d <= match_mm:
            matched[si].add(best_j)
            tp_cum += 1
        else:
            fp_cum += 1
        fp_rate = fp_cum / max(n_scans, 1)
        sens = tp_cum / max(n_gt, 1)
        curve.append((fp_rate, sens))
        _ = score  # silence unused

    # Interpolate sensitivity at target FP rates
    out: dict[str, float] = {"n_scans": float(n_scans), "n_gt": float(n_gt)}
    fps = np.asarray([c[0] for c in curve], dtype=np.float64)
    sens = np.asarray([c[1] for c in curve], dtype=np.float64)
    sens_at: list[float] = []
    for t in fp_targets:
        # last point with fp_rate <= t
        idx = np.where(fps <= float(t))[0]
        if idx.size == 0:
            s = 0.0
        else:
            s = float(sens[idx[-1]])
        key = f"sens@fp{t:g}"
        out[key] = s
        sens_at.append(s)
    out["froc_mean"] = float(np.mean(sens_at)) if sens_at else 0.0
    return out
