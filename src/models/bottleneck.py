"""
Optional encoder bottleneck: compress pooled encoder features before the head.

Three-layer MLP: encoder_dim → h1 → h2 → latent_dim
"""

from __future__ import annotations

import torch.nn as nn


class EncoderBottleneck(nn.Module):
    """Map transformer / DeepSets output to a smaller latent representation."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or latent_dim >= in_dim:
            raise ValueError(
                f"bottleneck latent dim must satisfy 0 < {latent_dim} < {in_dim}"
            )
        h1 = max(latent_dim * 2, in_dim // 2)
        h2 = max(latent_dim * 2, (in_dim + latent_dim) // 2)
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, latent_dim),
        )

    def forward(self, x):
        return self.net(x)
