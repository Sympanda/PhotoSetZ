"""
Set Transformer encoder for photometric redshift estimation.

Architecture
------------
  Token embedding MLP:  token_dim → embed_dim
  Transformer encoder:  n_layers × (multi-head self-attention + FFN)
  PMA pooling:          Pooling by Multi-head Attention (Lee et al. 2019)
                        with k=1 seed vector → (B, embed_dim)

The PMA output is then passed to a prediction head.

References
----------
  Lee et al. (2019) "Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks", ICML.
"""

from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    dims = [in_dim] + hidden_dims
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class PMA(nn.Module):
    """
    Pooling by Multi-head Attention (Lee et al. 2019).

    Learns k seed vectors S ∈ R^(k × d) and attends over the encoded tokens:

        PMA_k(Z) = MultiHeadAttn(S, rFF(Z), rFF(Z))

    For photoz we use k=1, producing a single summary vector.
    """

    def __init__(self, embed_dim: int, n_heads: int, n_seeds: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        Z: torch.Tensor,                 # (B, N, embed_dim)
        key_padding_mask: torch.Tensor,  # (B, N) True=padding
    ) -> torch.Tensor:                   # (B, n_seeds, embed_dim)
        B = Z.size(0)
        S = self.seeds.expand(B, -1, -1)              # (B, n_seeds, embed_dim)
        Z_ff = self.ff(Z)                              # rFF applied to values/keys
        out, _ = self.attn(S, Z_ff, Z_ff, key_padding_mask=key_padding_mask)
        return self.ln(out)                            # (B, n_seeds, embed_dim)


class SetTransformer(nn.Module):
    """
    Parameters
    ----------
    token_dim : int
        Dimension of each input token.
    embed_dim : int
        Internal transformer dimension (d_model).
    n_heads : int
        Number of attention heads.
    n_attn_layers : int
        Number of transformer encoder layers.
    ffn_dim : int
        Feed-forward network hidden dimension inside each transformer layer.
        Defaults to 4 * embed_dim.
    n_pma_seeds : int
        Number of PMA seed vectors (almost always 1 for a single summary).
    dropout : float
        Dropout probability in attention and FFN.
    pre_embed_hidden : list[int]
        Hidden dims for the initial token-embedding MLP before the transformer.
    """

    def __init__(
        self,
        token_dim: int = 4,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        ffn_dim: int | None = None,
        n_pma_seeds: int = 1,
        dropout: float = 0.1,
        pre_embed_hidden: List[int] = (64,),
    ) -> None:
        super().__init__()

        ffn_dim = ffn_dim or 4 * embed_dim

        # Token embedding: raw features → embed_dim
        self.embed = _build_mlp(token_dim, list(pre_embed_hidden), embed_dim, dropout=dropout)

        # Transformer encoder layers
        # Pre-LN (norm_first=True) gives more stable training gradients but
        # disables PyTorch's nested-tensor fast path — the resulting warning is
        # expected and harmless.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        # Pooling by Multi-head Attention
        self.pma = PMA(embed_dim, n_heads, n_seeds=n_pma_seeds, dropout=dropout)

        self._embed_dim = embed_dim
        self._n_pma_seeds = n_pma_seeds

    @property
    def output_dim(self) -> int:
        return self._embed_dim * self._n_pma_seeds

    def forward(
        self,
        tokens: torch.Tensor,           # (B, N, token_dim)
        key_padding_mask: torch.Tensor, # (B, N) True=padding
    ) -> torch.Tensor:                  # (B, embed_dim)
        # Embed each token independently
        h = self.embed(tokens)          # (B, N, embed_dim)

        # Transformer encoder with masked self-attention
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)  # (B, N, embed_dim)

        # Pool to a single vector via PMA
        pooled = self.pma(h, key_padding_mask)   # (B, n_seeds, embed_dim)

        # Flatten seed dimension
        B = pooled.size(0)
        return pooled.view(B, -1)                # (B, embed_dim * n_seeds)
