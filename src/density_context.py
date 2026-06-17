"""Optional density-context MLP for NSF posterior conditioning."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DensityContextNet(nn.Module):
    """
    Projects pooled set features (+ optional coverage summaries) into an NSF
    conditioning vector.  Kept separate from the point-prediction path.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
