"""Slice visualization: CT + lung field + nodule highlight for the review demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage

# Clinical-ish lung window so parenchyma is visible (not washed out)
LUNG_WINDOW = (-1000.0, -200.0)
SOFT_WINDOW = (-1000.0, 400.0)


def estimate_lung_mask(volume_hu: np.ndarray) -> np.ndarray:
    """Fallback lung mask from HU when no lung_masks/ file exists."""
    air = (volume_hu > -1000) & (volume_hu < -300)
    labeled, n = ndimage.label(air)
    if n == 0:
        return np.zeros(volume_hu.shape, dtype=np.uint8)
    sizes = ndimage.sum(air, labeled, index=range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = np.zeros(volume_hu.shape, dtype=bool)
    for idx in order[:2]:
        keep |= labeled == (idx + 1)
    out = ndimage.binary_fill_holes(keep)
    return out.astype(np.uint8)


def window_display(volume: np.ndarray, hu_window: tuple[float, float] = SOFT_WINDOW) -> np.ndarray:
    lo, hi = float(hu_window[0]), float(hu_window[1])
    x = np.clip(volume, lo, hi)
    return ((x - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def nodule_centroid_zyx(nodule_mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(nodule_mask > 0)
    if coords.size == 0:
        z = nodule_mask.shape[0] // 2
        return z, nodule_mask.shape[1] // 2, nodule_mask.shape[2] // 2
    c = coords.mean(axis=0)
    return int(round(c[0])), int(round(c[1])), int(round(c[2]))


def _to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def _compose_overlay(
    ct: np.ndarray,
    lung: np.ndarray,
    nod: np.ndarray,
    *,
    lung_alpha: float = 0.35,
    nod_alpha: float = 0.75,
) -> np.ndarray:
    """Gray CT + bright teal lung wash + solid red nodule."""
    base = np.stack([ct, ct, ct], axis=-1).astype(np.float32)
    overlay = base.copy()
    lung_bool = lung > 0
    nod_bool = nod > 0
    # Bright teal so the lung field is unmistakable
    teal = np.array([0.05, 0.92, 0.78], dtype=np.float32)
    red = np.array([1.0, 0.12, 0.10], dtype=np.float32)
    overlay[lung_bool] = (1.0 - lung_alpha) * overlay[lung_bool] + lung_alpha * teal
    overlay[nod_bool] = (1.0 - nod_alpha) * overlay[nod_bool] + nod_alpha * red
    return np.clip(overlay, 0.0, 1.0)


def _lung_only_panel(ct: np.ndarray, lung: np.ndarray) -> np.ndarray:
    """Dedicated lung panel: CT under light teal + thick cyan outline."""
    base = np.stack([ct, ct, ct], axis=-1).astype(np.float32)
    out = base.copy()
    lung_bool = lung > 0
    teal = np.array([0.05, 0.92, 0.78], dtype=np.float32)
    out[lung_bool] = 0.55 * out[lung_bool] + 0.45 * teal
    # Emphasize boundary with a drawn edge band
    edge = lung_bool ^ ndimage.binary_erosion(lung_bool, iterations=2)
    out[edge] = np.array([0.2, 1.0, 0.95], dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def render_pipeline_figure(
    volume: np.ndarray,
    lung_mask: np.ndarray,
    nodule_mask: np.ndarray,
    *,
    slice_z: int | None = None,
    title: str = "PulmoScan demo",
    prob: float | None = None,
    truth_label: str | None = None,
    out_path: str | Path | None = None,
    hu_window: tuple[float, float] = LUNG_WINDOW,
) -> tuple[np.ndarray, Path | None]:
    """Render 4-panel figure: CT | lung field | lung+nodule | nodule zoom."""
    disp = window_display(volume, hu_window)
    cz, cy, cx = nodule_centroid_zyx(nodule_mask)
    if slice_z is None:
        slice_z = cz
    slice_z = int(np.clip(slice_z, 0, volume.shape[0] - 1))

    ct = disp[slice_z]
    lung = (lung_mask[slice_z] > 0).astype(np.float32)
    nod = (nodule_mask[slice_z] > 0).astype(np.float32)
    lung_panel = _lung_only_panel(ct, lung)
    overlay = _compose_overlay(ct, lung, nod)

    r = 24
    y0 = int(np.clip(cy - r, 0, max(overlay.shape[0] - 8, 0)))
    y1 = int(np.clip(cy + r, y0 + 8, overlay.shape[0]))
    x0 = int(np.clip(cx - r, 0, max(overlay.shape[1] - 8, 0)))
    x1 = int(np.clip(cx + r, x0 + 8, overlay.shape[1]))
    zoom = overlay[y0:y1, x0:x1].copy()
    if zoom.size == 0:
        zoom = overlay.copy()

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
    axes[0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("1 · CT (lung window)")
    axes[0].axhline(cy, color="yellow", lw=0.6, alpha=0.5)
    axes[0].axvline(cx, color="yellow", lw=0.6, alpha=0.5)

    axes[1].imshow(lung_panel)
    axes[1].set_title("2 · Lung field")
    if lung.any():
        axes[1].contour(lung, levels=[0.5], colors=["#00FFE0"], linewidths=1.6)

    axes[2].imshow(overlay)
    axes[2].set_title("3 · Lung + nodule")
    if lung.any():
        axes[2].contour(lung, levels=[0.5], colors=["#00FFE0"], linewidths=1.2)
    if nod.any():
        axes[2].contour(nod, levels=[0.5], colors=["#FF3030"], linewidths=1.6)

    axes[3].imshow(zoom)
    axes[3].set_title("4 · Nodule close-up")
    for ax in axes:
        ax.axis("off")

    legend = [
        Patch(facecolor=(0.05, 0.92, 0.78), edgecolor="none", label="Lung field"),
        Patch(facecolor=(1.0, 0.12, 0.10), edgecolor="none", label="Nodule"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False)

    subtitle = title
    if prob is not None:
        tier = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
        subtitle += f"  |  P(malignant)={prob:.3f} ({tier})"
    if truth_label:
        subtitle += f"  |  label={truth_label}"
    fig.suptitle(subtitle, fontsize=12, y=1.02)
    fig.tight_layout()

    saved: Path | None = None
    if out_path is not None:
        saved = Path(out_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, dpi=150, bbox_inches="tight", facecolor="white")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb, saved


def render_simple_panels(
    volume: np.ndarray,
    lung_mask: np.ndarray,
    nodule_mask: np.ndarray,
    *,
    slice_z: int | None = None,
    hu_window: tuple[float, float] = LUNG_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return four RGB panels: CT, lung-only, lung+nodule, nodule zoom."""
    disp = window_display(volume, hu_window)
    cz, cy, cx = nodule_centroid_zyx(nodule_mask)
    if slice_z is None:
        slice_z = cz
    slice_z = int(np.clip(slice_z, 0, volume.shape[0] - 1))
    ct = disp[slice_z]
    lung = (lung_mask[slice_z] > 0).astype(np.float32)
    nod = (nodule_mask[slice_z] > 0).astype(np.float32)

    ct_rgb = np.stack([ct, ct, ct], axis=-1)
    lung_panel = _lung_only_panel(ct, lung)
    overlay = _compose_overlay(ct, lung, nod)

    r = 24
    y0 = int(np.clip(cy - r, 0, max(overlay.shape[0] - 8, 0)))
    y1 = int(np.clip(cy + r, y0 + 8, overlay.shape[0]))
    x0 = int(np.clip(cx - r, 0, max(overlay.shape[1] - 8, 0)))
    x1 = int(np.clip(cx + r, x0 + 8, overlay.shape[1]))
    zoom = overlay[y0:y1, x0:x1].copy()
    if zoom.size == 0:
        zoom = overlay.copy()

    return _to_u8(ct_rgb), _to_u8(lung_panel), _to_u8(overlay), _to_u8(zoom)
