"""Hybrid 3D CNN + Swin Transformer malignancy classifier."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pulmoscan.classification.backbone_cnn import ResNet3D18Backbone
from pulmoscan.classification.swin3d import SwinEncoder3D


class HybridCNNSwin(nn.Module):
    """
    VOI (B,1,D,H,W)
      → 3D CNN feature map (B,C,D',H',W')
      → Swin3D encoder + CLS attention pool
      → sigmoid malignancy probability
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = cfg or {}
        in_ch = int(cfg.get("in_channels", 1))
        backbone_cfg = cfg.get("backbone", {})
        swin_cfg = cfg.get("swin", {})
        head_cfg = cfg.get("head", {})

        self.backbone = ResNet3D18Backbone(
            in_channels=in_ch,
            base_channels=int(backbone_cfg.get("base_channels", 32)),
            out_channels=int(backbone_cfg.get("out_channels", 256)),
        )
        window = swin_cfg.get("window_size", [2, 2, 2])
        self.swin = SwinEncoder3D(
            in_channels=self.backbone.out_channels,
            embed_dim=int(swin_cfg.get("embed_dim", 256)),
            depths=list(swin_cfg.get("depths", [2, 2])),
            num_heads=list(swin_cfg.get("num_heads", [4, 8])),
            window_size=(int(window[0]), int(window[1]), int(window[2])),
            mlp_ratio=float(swin_cfg.get("mlp_ratio", 4.0)),
            dropout=float(swin_cfg.get("dropout", 0.1)),
            attn_dropout=float(swin_cfg.get("attn_dropout", 0.0)),
            drop_path=float(swin_cfg.get("drop_path", 0.1)),
        )
        hidden = int(head_cfg.get("hidden_dim", 128))
        dropout = float(head_cfg.get("dropout", 0.3))
        self.head = nn.Sequential(
            nn.LayerNorm(self.swin.out_dim),
            nn.Linear(self.swin.out_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.last_tokens: torch.Tensor | None = None
        self.last_feat: torch.Tensor | None = None
        self.last_cls: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        self.last_feat = feat
        cls, tokens = self.swin(feat)
        self.last_tokens = tokens
        self.last_cls = cls
        return cls, tokens, feat

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        cls, _tokens, _feat = self.encode(x)
        logits = self.head(cls).squeeze(-1)
        if return_aux:
            return {"malignancy_logits": logits, "prob": torch.sigmoid(logits)}
        if return_logits:
            return logits
        return torch.sigmoid(logits)

    @torch.no_grad()
    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return coarse spatial attention (B, D', H', W') from CLS→token weights."""
        was_training = self.training
        self.eval()
        feat = self.backbone(x)
        _, tokens = self.swin(feat)
        # Recompute CLS attention explicitly
        b, n, c = tokens.shape
        cls = self.swin.cls_token.expand(b, -1, -1)
        _, attn = self.swin.cls_attn(cls, tokens, tokens, need_weights=True, average_attn_weights=True)
        # attn shape: (B, N) or (B, 1, N)
        if attn.ndim == 3:
            attn = attn.squeeze(1)
        d, h, w = feat.shape[-3:]
        # After possible patch merging, token grid may be smaller than feat
        # Infer cubic-ish reshape from n
        grid = self._infer_grid(n, d, h, w)
        attn_map = attn.view(b, *grid)
        if was_training:
            self.train()
        return attn_map

    @staticmethod
    def _infer_grid(n: int, d: int, h: int, w: int) -> tuple[int, int, int]:
        # Prefer exact feat grid if matches; else downsample by 2 each merge
        if d * h * w == n:
            return d, h, w
        for scale in (2, 4, 8):
            dd, hh, ww = max(d // scale, 1), max(h // scale, 1), max(w // scale, 1)
            if dd * hh * ww == n:
                return dd, hh, ww
        # Fallback: factorize n into near-cube
        cube = int(round(n ** (1 / 3)))
        for zd in range(max(cube - 2, 1), cube + 3):
            if n % zd:
                continue
            rem = n // zd
            side = int(round(rem**0.5))
            for yh in range(max(side - 2, 1), side + 3):
                if rem % yh == 0:
                    return zd, yh, rem // yh
        return n, 1, 1


class TGHybridCNNSwin(HybridCNNSwin):
    """Texture-gated dual-expert head on the shared CNN–Swin CLS embedding.

    Extra vs the baseline hybrid:
      - concept head (texture, spiculation, margin, sphericity)
      - gate P(subsolid)
      - solid / subsolid malignancy experts mixed by the gate
      - abstain head for indeterminate (LIDC score ~3) nodules
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        super().__init__(cfg)
        cfg = cfg or {}
        tg = cfg.get("tg_hybrid_head", cfg.get("head", {}))
        hidden = int(tg.get("hidden_dim", cfg.get("head", {}).get("hidden_dim", 128)))
        dropout = float(tg.get("dropout", cfg.get("head", {}).get("dropout", 0.3)))
        dim = self.swin.out_dim

        def _mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )

        # Reuse baseline MLP as the solid expert so ablations stay comparable.
        self.solid_head = self.head
        self.subsolid_head = _mlp(1)
        self.gate_head = _mlp(1)
        self.concept_head = _mlp(4)
        self.abstain_head = _mlp(1)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        cls, _tokens, _feat = self.encode(x)
        solid_logits = self.solid_head(cls).squeeze(-1)
        subsolid_logits = self.subsolid_head(cls).squeeze(-1)
        gate_logits = self.gate_head(cls).squeeze(-1)
        gate = torch.sigmoid(gate_logits)
        # Mix expert probabilities, then convert back to logits for BCE-with-logits.
        mixed_prob = (1.0 - gate) * torch.sigmoid(solid_logits) + gate * torch.sigmoid(subsolid_logits)
        mixed_prob = mixed_prob.clamp(1e-4, 1.0 - 1e-4)
        malignancy_logits = torch.logit(mixed_prob)
        concepts = self.concept_head(cls)
        abstain_logits = self.abstain_head(cls).squeeze(-1)
        if return_aux:
            return {
                "malignancy_logits": malignancy_logits,
                "solid_logits": solid_logits,
                "subsolid_logits": subsolid_logits,
                "gate_logits": gate_logits,
                "concepts": concepts,
                "abstain_logits": abstain_logits,
                "prob": mixed_prob,
                "gate": gate,
            }
        if return_logits:
            return malignancy_logits
        return mixed_prob


def build_model(cfg: dict[str, Any] | None = None) -> HybridCNNSwin:
    cfg = cfg or {}
    if bool(cfg.get("tg_hybrid", False)):
        return TGHybridCNNSwin(cfg)
    return HybridCNNSwin(cfg)
