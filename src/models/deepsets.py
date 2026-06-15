"""
DeepSets encoder for photometric redshift estimation.

Architecture
------------
  φ (per-token MLP):   token_dim → hidden_dim → ... → latent_dim
  Pool:                masked {mean | sum | max | attention}-pooling over tokens
  ρ (aggregation MLP): latent_dim → hidden_dim → ... → embed_dim

The output embedding `embed_dim` is then passed to a prediction head.

Reference: Zaheer et al. (2017) "Deep Sets", NeurIPS.
"""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


PoolType = Literal["mean", "sum", "max", "attention"]

_ACTIVATIONS = {
    "gelu":       nn.GELU,
    "relu":       nn.ReLU,
    "silu":       nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh":       nn.Tanh,
}


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


def _build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: nn.Module = None,
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    activation = activation or nn.GELU()
    layers: List[nn.Module] = []
    dims = [in_dim] + hidden_dims
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), activation, nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-1], out_dim))
    if final_activation:
        layers.append(activation)
    return nn.Sequential(*layers)


class AttentionPool(nn.Module):
    """Learned attention pooling over the token dimension."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(latent_dim, 1)

    def forward(
        self,
        h: torch.Tensor,           # (B, N, latent_dim)
        key_padding_mask: torch.Tensor,  # (B, N) True=padding
    ) -> torch.Tensor:             # (B, latent_dim)
        # Compute attention scores; mask out padding positions
        scores = self.score(h).squeeze(-1)          # (B, N)
        scores = scores.masked_fill(key_padding_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)     # (B, N)
        return (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, latent_dim)


class DeepSets(nn.Module):
    """
    Parameters
    ----------
    token_dim : int
        Dimension of each input token (default 4).
    phi_hidden : list[int]
        Hidden layer widths for the per-token network φ.
    latent_dim : int
        Output dimension of φ (also the pooling dimension).
    rho_hidden : list[int]
        Hidden layer widths for the aggregation network ρ.
    embed_dim : int
        Output dimension of ρ, passed to the prediction head.
    pooling : str
        One of 'mean', 'sum', 'max', 'attention'.
    dropout : float
        Dropout probability applied inside both MLPs.
    """

    def __init__(
        self,
        token_dim: int = 4,
        phi_hidden: List[int] = (128, 128),
        latent_dim: int = 128,
        rho_hidden: List[int] = (256, 128),
        embed_dim: int = 128,
        pooling: PoolType = "mean",
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.pooling_type = pooling
        act = _get_activation(activation)

        self.phi = _build_mlp(
            token_dim, list(phi_hidden), latent_dim,
            activation=act, dropout=dropout, final_activation=True,
        )
        self.rho = _build_mlp(
            latent_dim, list(rho_hidden), embed_dim,
            activation=act, dropout=dropout,
        )

        if pooling == "attention":
            self.attn_pool = AttentionPool(latent_dim)
        else:
            self.attn_pool = None

    @property
    def output_dim(self) -> int:
        return self.rho[-1].out_features  # type: ignore[index]

    def forward(
        self,
        tokens: torch.Tensor,           # (B, N, token_dim)
        key_padding_mask: torch.Tensor, # (B, N) True=padding
    ) -> torch.Tensor:                  # (B, embed_dim)
        # Per-token encoding
        h = self.phi(tokens)            # (B, N, latent_dim)

        # Masked pooling
        if self.pooling_type == "attention":
            pooled = self.attn_pool(h, key_padding_mask)          # (B, latent_dim)
        else:
            # Zero out padding positions before pooling
            valid_mask = ~key_padding_mask                         # (B, N) True=valid
            h = h * valid_mask.unsqueeze(-1).float()

            n_valid = valid_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
            n_valid = n_valid.clamp(min=1.0)

            if self.pooling_type == "mean":
                pooled = h.sum(dim=1) / n_valid                    # (B, latent_dim)
            elif self.pooling_type == "sum":
                pooled = h.sum(dim=1)
            elif self.pooling_type == "max":
                h = h.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
                pooled = h.max(dim=1).values
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return self.rho(pooled)          # (B, embed_dim)
