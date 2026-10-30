"""Lightweight 3D U-Net for nodule segmentation (pure PyTorch, no MONAI)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd sizes cause 1-voxel mismatch after upsample
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        if dz or dy or dx:
            x = nn.functional.pad(
                x,
                [
                    dx // 2,
                    dx - dx // 2,
                    dy // 2,
                    dy - dy // 2,
                    dz // 2,
                    dz - dz // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """Encoder–decoder 3D U-Net → per-voxel logits (B, 1, D, H, W)."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.depth = depth
        chs = [base_channels * (2**i) for i in range(depth)]
        self.stem = DoubleConv3D(in_channels, chs[0])
        self.downs = nn.ModuleList(
            [Down3D(chs[i], chs[i + 1]) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList(
            [
                Up3D(chs[i + 1], chs[i], chs[i])
                for i in range(depth - 2, -1, -1)
            ]
        )
        self.head = nn.Conv3d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = self.stem(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # skips[-1] is bottleneck; ups consume skips[-2], skips[-3], ...
        x = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            x = up(x, skip)
        return self.head(x)


def build_unet(cfg: dict[str, Any]) -> UNet3D:
    return UNet3D(
        in_channels=int(cfg.get("in_channels", 1)),
        base_channels=int(cfg.get("base_channels", 16)),
        depth=int(cfg.get("depth", 4)),
    )
