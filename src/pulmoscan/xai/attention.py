"""Attention-map helpers for hybrid CNN–Swin."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def upsample_attention(
    attn: torch.Tensor,
    target_size: tuple[int, int, int],
) -> torch.Tensor:
    """Upsample (B, D', H', W') attention to VOI size."""
    if attn.ndim == 3:
        attn = attn.unsqueeze(0)
    attn = attn.unsqueeze(1)  # B,1,D,H,W
    up = F.interpolate(attn.float(), size=target_size, mode="trilinear", align_corners=False)
    return up.squeeze(1)


def save_attention_overlay(
    voi: np.ndarray,
    attn: np.ndarray,
    out_path: str | Path,
    title: str = "Attention overlay",
) -> Path:
    """Save mid-slice CT + attention overlay PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = voi.shape[0] // 2
    img = voi[z]
    heat = attn[z] if attn.ndim == 3 else attn
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("VOI mid-slice")
    axes[1].imshow(heat, cmap="magma")
    axes[1].set_title("Attention")
    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(heat, cmap="magma", alpha=0.45)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
