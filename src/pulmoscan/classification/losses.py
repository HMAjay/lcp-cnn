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


class TGHybridLoss(nn.Module):
    """Gated malignancy BCE + concept regression + gate BCE + abstain BCE + expert split."""

    def __init__(
        self,
        *,
        pos_weight: float | None = None,
        lambda_mal: float = 1.0,
        lambda_concept: float = 0.25,
        lambda_gate: float = 0.25,
        lambda_expert: float = 0.35,
        lambda_abstain: float = 0.2,
    ) -> None:
        super().__init__()
        self.bce = WeightedBCEWithLogits(pos_weight=pos_weight)
        self.lambda_mal = lambda_mal
        self.lambda_concept = lambda_concept
        self.lambda_gate = lambda_gate
        self.lambda_expert = lambda_expert
        self.lambda_abstain = lambda_abstain

    def _weighted_bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.bce.pos_weight is not None:
            pw = torch.as_tensor(self.bce.pos_weight, device=logits.device, dtype=logits.dtype)
            # pos_weight applied as in BCE-with-logits: multiply positive terms
            per = per * torch.where(targets > 0.5, pw, torch.ones_like(per))
        if weight is None:
            return per.mean()
        denom = weight.sum().clamp_min(1.0)
        return (per * weight).sum() / denom

    def forward(self, aux: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        y = batch["label"]
        mal_w = batch.get("malignancy_weight")
        if mal_w is None:
            mal_w = torch.ones_like(y)
        sub = batch.get("subsolid", torch.zeros_like(y))
        uncertain = batch.get("uncertain", torch.zeros_like(y))
        concepts = batch.get("concepts")
        attr_mask = batch.get("attr_mask")

        loss_mal = self._weighted_bce(aux["malignancy_logits"], y, mal_w)
        loss_gate = self._weighted_bce(aux["gate_logits"], sub, attr_mask[:, 0] if attr_mask is not None else None)
        loss_solid = self._weighted_bce(aux["solid_logits"], y, mal_w * (1.0 - sub))
        loss_sub = self._weighted_bce(aux["subsolid_logits"], y, mal_w * sub)
        loss_abs = F.binary_cross_entropy_with_logits(aux["abstain_logits"], uncertain)

        loss_c = aux["malignancy_logits"].new_zeros(())
        if concepts is not None and attr_mask is not None:
            pred = aux["concepts"]
            # Concepts are LIDC 1–5 scores; no activation so MSE can fit the range.
            se = (pred - concepts) ** 2
            denom = attr_mask.sum().clamp_min(1.0)
            loss_c = (se * attr_mask).sum() / denom

        return (
            self.lambda_mal * loss_mal
            + self.lambda_concept * loss_c
            + self.lambda_gate * loss_gate
            + self.lambda_expert * (loss_solid + loss_sub)
            + self.lambda_abstain * loss_abs
        )


def build_loss(
    name: str = "weighted_bce",
    *,
    pos_weight: float | None = None,
    gamma: float = 2.0,
    **tg_kwargs: float,
) -> nn.Module:
    name = name.lower()
    if name in {"tg_hybrid", "tg-hybrid"}:
        return TGHybridLoss(pos_weight=pos_weight, **tg_kwargs)
    if name == "focal":
        return FocalLossWithLogits(gamma=gamma, pos_weight=pos_weight)
    return WeightedBCEWithLogits(pos_weight=pos_weight)
