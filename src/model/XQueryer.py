"""Lightweight XQueryer model.

Folder: src/model
Purpose: neural-network modules and dataset code for the lightweight
XQueryer pipeline.
Author: Dr. Bin Cao, https://bin-cao.github.io/
"""

from __future__ import annotations

import torch.nn.functional as F
import torch
from torch import nn


class FFTFilterBank(nn.Module):
    """Parameter-free FFT branch that creates robust multi-band XRD views."""

    def __init__(self, keep_ratios: tuple[float, ...] = (0.35, 0.6, 0.85)):
        super().__init__()
        self.keep_ratios = keep_ratios

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1)
        bands = [x]
        freq_len = spectrum.shape[-1]
        for ratio in self.keep_ratios:
            cutoff = max(2, min(freq_len, int(freq_len * ratio)))
            filtered = spectrum.clone()
            filtered[..., cutoff:] = 0
            bands.append(torch.fft.irfft(filtered, n=x.shape[-1], dim=-1))
        return torch.cat(bands, dim=1)


class SeparableConvBlock(nn.Module):
    """Depthwise-separable residual convolution for compact peak feature learning."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x + residual)
        return self.dropout(x)


class CNNPeakEncoder(nn.Module):
    """Multi-scale CNN backbone for local peak shape and shift features."""

    def __init__(self, in_channels: int, base_channels: int, dropout: float):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            SeparableConvBlock(base_channels, base_channels, kernel_size=17, stride=2, dropout=dropout),
            SeparableConvBlock(base_channels, base_channels * 2, kernel_size=15, stride=2, dropout=dropout),
            SeparableConvBlock(base_channels * 2, base_channels * 2, kernel_size=11, stride=2, dropout=dropout),
            SeparableConvBlock(base_channels * 2, base_channels * 3, kernel_size=9, stride=2, dropout=dropout),
            SeparableConvBlock(base_channels * 3, base_channels * 3, kernel_size=7, stride=2, dropout=dropout),
        )
        self.out_channels = base_channels * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class TokenProjector(nn.Module):
    """Compress dense CNN features to a short token sequence for attention."""

    def __init__(self, in_channels: int, attn_dim: int, num_tokens: int, dropout: float):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, attn_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).transpose(1, 2).contiguous()
        return self.proj(x) + self.pos_embed


class CrossAttentionBlock(nn.Module):
    """Residual element-guided cross attention with query self-attention."""

    def __init__(self, attn_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(attn_dim)
        self.norm_kv = nn.LayerNorm(attn_dim)
        self.norm_self = nn.LayerNorm(attn_dim)
        self.norm_ffn = nn.LayerNorm(attn_dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(attn_dim, attn_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim * 4, attn_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(queries)
        kv = self.norm_kv(tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        queries = queries + self.drop(attn_out)
        self_out, _ = self.self_attn(self.norm_self(queries), self.norm_self(queries), self.norm_self(queries), need_weights=False)
        queries = queries + self.drop(self_out)
        return queries + self.ffn(self.norm_ffn(queries))


class CosineClassifier(nn.Module):
    """Normalized classifier with temperature scaling for large-class retrieval."""

    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(scale))))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(x, weight) * self.log_scale.exp().clamp(1.0, 100.0)


class Xmodel(nn.Module):
    """XQueryer-compatible FFT + CNN + CrossAttention + classifier model.

    Compared with the original XQueryer, this version keeps the same conceptual
    framework but replaces the huge element-to-sequence expansion with a small
    set of element-conditioned queries. CNN features are compressed to tokens
    before attention, preserving global peak matching while reducing memory.
    """

    def __init__(
        self,
        input_length: int = 3500,
        num_classes: int = 100315,
        base_channels: int = 64,
        attn_dim: int = 192,
        num_heads: int = 6,
        num_tokens: int = 96,
        num_queries: int = 4,
        num_attn_layers: int = 2,
        element_dim: int = 92,
        dropout: float = 0.15,
        classifier: str = "cosine",
        **_: object,
    ):
        super().__init__()
        self.input_length = input_length
        self.num_classes = num_classes
        self.element_dim = element_dim
        self.classifier_type = classifier

        self.fft = FFTFilterBank()
        self.cnn = CNNPeakEncoder(in_channels=4, base_channels=base_channels, dropout=dropout)
        self.tokens = TokenProjector(self.cnn.out_channels, attn_dim, num_tokens, dropout)

        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, attn_dim))
        self.element_encoder = nn.Sequential(
            nn.Linear(element_dim, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, num_queries * attn_dim),
        )
        self.attention_layers = nn.ModuleList(
            [CrossAttentionBlock(attn_dim, num_heads, dropout) for _ in range(num_attn_layers)]
        )
        fusion_dim = attn_dim * num_queries + self.cnn.out_channels * 2 + 8
        self.neck = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        if classifier == "cosine":
            self.cls_head = CosineClassifier(256, num_classes)
        elif classifier == "linear":
            self.cls_head = nn.Linear(256, num_classes)
        else:
            raise ValueError("classifier must be 'cosine' or 'linear'")
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _prepare_element(self, elem: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if elem is None:
            return torch.zeros((batch_size, self.element_dim), dtype=dtype, device=device)
        elem = elem.to(device=device, dtype=dtype)
        if elem.shape[1] == self.element_dim:
            return elem
        fixed = torch.zeros((elem.shape[0], self.element_dim), dtype=dtype, device=device)
        fixed[:, : min(elem.shape[1], self.element_dim)] = elem[:, : self.element_dim]
        return fixed

    def forward(self, x: torch.Tensor, elem: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)
        x = x[:, : self.input_length].float()
        x = x / x.amax(dim=1, keepdim=True).clamp_min(1e-6)
        x = x.unsqueeze(1)

        fft_views = self.fft(x)
        cnn_features = self.cnn(fft_views)
        xrd_tokens = self.tokens(cnn_features)
        cnn_global = torch.cat([cnn_features.mean(dim=-1), cnn_features.amax(dim=-1)], dim=1)
        fft_global = torch.cat([fft_views.mean(dim=-1), fft_views.std(dim=-1)], dim=1)

        elem = self._prepare_element(elem, x.shape[0], x.device, x.dtype)
        element_queries = self.element_encoder(elem).view(x.shape[0], self.query_embed.shape[1], -1)
        queries = self.query_embed.expand(x.shape[0], -1, -1) + element_queries
        for layer in self.attention_layers:
            queries = layer(queries, xrd_tokens)
        fused = torch.cat([queries.flatten(1), cnn_global, fft_global], dim=1)
        return self.cls_head(self.neck(fused))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
