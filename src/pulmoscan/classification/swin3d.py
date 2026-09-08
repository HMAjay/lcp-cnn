"""Compact 3D Swin Transformer encoder operating on CNN feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _to_3tuple(x: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(x, tuple):
        return (int(x[0]), int(x[1]), int(x[2]))
    return (int(x), int(x), int(x))


def window_partition(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    """(B, D, H, W, C) -> (num_windows*B, Wd*Wh*Ww, C)."""
    wd, wh, ww = window_size
    b, d, h, w, c = x.shape
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, wd * wh * ww, c)
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    d: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """(num_windows*B, Wd*Wh*Ww, C) -> (B, D, H, W, C)."""
    wd, wh, ww = window_size
    b = int(windows.shape[0] / ((d // wd) * (h // wh) * (w // ww)))
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(b, d, h, w, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class WindowAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int, int],
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Relative position bias
        wd, wh, ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += wd - 1
        rel[:, :, 1] += wh - 1
        rel[:, :, 2] += ww - 1
        rel[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[:, :, 1] *= 2 * ww - 1
        rel_index = rel.sum(-1)
        self.register_buffer("relative_position_index", rel_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B_, N, C)
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int, int] = (2, 2, 2),
        shift_size: tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, attn_dropout, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W, C)
        b, d, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        wd, wh, ww = self.window_size
        # pad to multiples of window
        pad_d = (wd - d % wd) % wd
        pad_h = (wh - h % wh) % wh
        pad_w = (ww - w % ww) % ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, dp, hp, wp, _ = x.shape

        sd, sh, sw = self.shift_size
        if any(self.shift_size):
            shifted = torch.roll(x, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
        else:
            shifted = x

        windows = window_partition(shifted, self.window_size)
        attn_windows = self.attn(windows)
        shifted = window_reverse(attn_windows, self.window_size, dp, hp, wp)

        if any(self.shift_size):
            x = torch.roll(shifted, shifts=(sd, sh, sw), dims=(1, 2, 3))
        else:
            x = shifted

        if pad_d or pad_h or pad_w:
            x = x[:, :d, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W, C)
        b, d, h, w, c = x.shape
        pad_d, pad_h, pad_w = d % 2, h % 2, w % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            _, d, h, w, _ = x.shape
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.reduction(self.norm(x))
        return x


class SwinEncoder3D(nn.Module):
    """Swin stages over CNN feature tokens; returns CLS-pooled features + last attention."""

    def __init__(
        self,
        in_channels: int = 256,
        embed_dim: int = 256,
        depths: list[int] | tuple[int, ...] = (2, 2),
        num_heads: list[int] | tuple[int, ...] = (4, 8),
        window_size: tuple[int, int, int] = (2, 2, 2),
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = _to_3tuple(window_size)
        self.proj = nn.Linear(in_channels, embed_dim) if in_channels != embed_dim else nn.Identity()
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        dim = embed_dim
        dp_idx = 0
        for i, depth in enumerate(depths):
            blocks = nn.ModuleList()
            for j in range(depth):
                shift = (0, 0, 0) if j % 2 == 0 else tuple(w // 2 for w in self.window_size)
                blocks.append(
                    SwinTransformerBlock3D(
                        dim=dim,
                        num_heads=num_heads[i],
                        window_size=self.window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        drop_path=dpr[dp_idx],
                    )
                )
                dp_idx += 1
            downsample = PatchMerging3D(dim) if i < len(depths) - 1 else None
            self.layers.append(nn.ModuleDict({"blocks": blocks, "downsample": downsample or nn.Identity()}))
            if i < len(depths) - 1:
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_attn = nn.MultiheadAttention(dim, num_heads=num_heads[-1], batch_first=True, dropout=attn_dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        feat: (B, C, D, H, W)
        returns: (cls_features (B, C_out), spatial_tokens (B, N, C_out))
        """
        x = rearrange(feat, "b c d h w -> b d h w c")
        x = self.proj(x)
        x = self.pos_drop(x)

        for stage in self.layers:
            for blk in stage["blocks"]:
                x = blk(x)
            if not isinstance(stage["downsample"], nn.Identity):
                x = stage["downsample"](x)

        x = self.norm(x)
        tokens = rearrange(x, "b d h w c -> b (d h w) c")
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        cls_out, attn = self.cls_attn(cls, tokens, tokens, need_weights=True, average_attn_weights=True)
        # attn: (B, 1, N) when average_attn_weights=True in recent torch — actually (B, N) 
        return cls_out.squeeze(1), tokens
