"""Dice + BCE loss for sparse nodule segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Soft Dice + BCE-with-logits. Works with logits (B,1,D,H,W) and binary targets."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.smooth = float(smooth)
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dtype != logits.dtype:
            targets = targets.to(dtype=logits.dtype)
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice = dice.mean()
        return self.dice_weight * dice + self.bce_weight * bce


def build_seg_loss(
    name: str = "dice_bce",
    *,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    pos_weight: float | None = None,
) -> nn.Module:
    name = (name or "dice_bce").lower()
    if name not in {"dice_bce", "dice+bce", "dice_bce_loss"}:
        raise ValueError(f"Unsupported Stage-1 loss: {name}")
    return DiceBCELoss(
        dice_weight=dice_weight,
        bce_weight=bce_weight,
        pos_weight=pos_weight,
    )
