"""Slice visualization: CT + lung tint + nodule highlight for the review demo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage


def estimate_lung_mask(volume_hu: np.ndarray) -> np.ndarray:
    """Fallback lung mask from HU when no lung_masks/ file exists."""
    air = (volume_hu > -1000) & (volume_hu < -300)
    # Keep largest connected air components (rough lungs)
    labeled, n = ndimage.label(air)
    if n == 0:
        return np.zeros(volume_hu.shape, dtype=np.uint8)
    sizes = ndimage.sum(air, labeled, index=range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = np.zeros(volume_hu.shape, dtype=bool)
    for idx in order[:2]:
        keep |= labeled == (idx + 1)
    # Fill holes per slice lightly
    out = ndimage.binary_fill_holes(keep)
    return out.astype(np.uint8)


def window_display(volume: np.ndarray, hu_window: tuple[float, float] = (-1000.0, 400.0)) -> np.ndarray:
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
    hu_window: tuple[float, float] = (-1000.0, 400.0),
) -> tuple[np.ndarray, Path | None]:
    """Render a 3-panel figure: CT | lung+nodule overlay | VOI zoom.

    Returns (RGB uint8 HWC array, optional saved path).
    """
    disp = window_display(volume, hu_window)
    cz, cy, cx = nodule_centroid_zyx(nodule_mask)
    if slice_z is None:
        slice_z = cz
    slice_z = int(np.clip(slice_z, 0, volume.shape[0] - 1))

    ct = disp[slice_z]
    lung = (lung_mask[slice_z] > 0).astype(np.float32)
    nod = (nodule_mask[slice_z] > 0).astype(np.float32)

    # RGB overlay: gray CT, green lung wash, red nodule
    base = np.stack([ct, ct, ct], axis=-1)
    overlay = base.copy()
    # lung tint (teal)
    lung_alpha = 0.28
    overlay[lung > 0] = (
        (1 - lung_alpha) * overlay[lung > 0]
        + lung_alpha * np.array([0.15, 0.75, 0.65], dtype=np.float32)
    )
    # nodule solid red-ish
    nod_alpha = 0.65
    overlay[nod > 0] = (
        (1 - nod_alpha) * overlay[nod > 0]
        + nod_alpha * np.array([0.95, 0.15, 0.12], dtype=np.float32)
    )

    # VOI zoom around nodule on the 2D axial overlay (H, W, 3)
    r = 24
    y0 = int(np.clip(cy - r, 0, max(overlay.shape[0] - 8, 0)))
    y1 = int(np.clip(cy + r, y0 + 8, overlay.shape[0]))
    x0 = int(np.clip(cx - r, 0, max(overlay.shape[1] - 8, 0)))
    x1 = int(np.clip(cx + r, x0 + 8, overlay.shape[1]))
    zoom = overlay[y0:y1, x0:x1].copy()
    if zoom.size == 0:
        zoom = overlay.copy()

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    axes[0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input CT (axial)")
    axes[0].axhline(cy, color="yellow", lw=0.6, alpha=0.5)
    axes[0].axvline(cx, color="yellow", lw=0.6, alpha=0.5)

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("Lung (teal) + nodule (red)")
    axes[1].contour(lung, levels=[0.5], colors=["#7CFFEE"], linewidths=0.8, alpha=0.9)
    if nod.any():
        axes[1].contour(nod, levels=[0.5], colors=["#FF4D4D"], linewidths=1.2)

    axes[2].imshow(np.clip(zoom, 0, 1))
    axes[2].set_title("Nodule close-up")
    for ax in axes:
        ax.axis("off")

    legend = [
        Patch(facecolor=(0.15, 0.75, 0.65), edgecolor="none", label="Lung field"),
        Patch(facecolor=(0.95, 0.15, 0.12), edgecolor="none", label="Nodule"),
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
    w, h = fig.canvas.get_width_height()
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
    hu_window: tuple[float, float] = (-1000.0, 400.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three RGB images for Gradio galleries: CT, overlay, zoom."""
    disp = window_display(volume, hu_window)
    cz, cy, cx = nodule_centroid_zyx(nodule_mask)
    if slice_z is None:
        slice_z = cz
    slice_z = int(np.clip(slice_z, 0, volume.shape[0] - 1))
    ct = disp[slice_z]
    lung = lung_mask[slice_z] > 0
    nod = nodule_mask[slice_z] > 0

    ct_rgb = np.stack([ct, ct, ct], axis=-1)
    overlay = ct_rgb.copy()
    overlay[lung] = (1 - 0.28) * overlay[lung] + 0.28 * np.array([0.15, 0.75, 0.65])
    overlay[nod] = (1 - 0.65) * overlay[nod] + 0.65 * np.array([0.95, 0.15, 0.12])

    r = 24
    y0 = int(np.clip(cy - r, 0, max(volume.shape[1] - 8, 0)))
    y1 = int(np.clip(cy + r, y0 + 8, volume.shape[1]))
    x0 = int(np.clip(cx - r, 0, max(volume.shape[2] - 8, 0)))
    x1 = int(np.clip(cx + r, x0 + 8, volume.shape[2]))
    zoom = overlay[y0:y1, x0:x1].copy()
    if zoom.size == 0:
        zoom = overlay.copy()

    def to_u8(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    return to_u8(ct_rgb), to_u8(overlay), to_u8(zoom)
