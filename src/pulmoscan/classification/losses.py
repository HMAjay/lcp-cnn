"""Loss functions for imbalanced malignancy classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()


def build_loss(name: str = "weighted_bce", *, pos_weight: float | None = None, gamma: float = 2.0) -> nn.Module:
    name = name.lower()
    if name == "focal":
        return FocalLossWithLogits(gamma=gamma, pos_weight=pos_weight)
    return WeightedBCEWithLogits(pos_weight=pos_weight)
