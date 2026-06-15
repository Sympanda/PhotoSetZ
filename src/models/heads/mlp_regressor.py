"""
MLP Regressor head — the simplest possible prediction head.

Predicts a single scalar redshift via a small MLP and optimises
with Huber loss (smooth L1), which is less sensitive to outliers
than MSE while still being differentiable everywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {
    "gelu":       nn.GELU,
    "relu":       nn.ReLU,
    "silu":       nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh":       nn.Tanh,
}


class MLPRegressor(nn.Module):
    """
    Parameters
    ----------
    embed_dim : int
        Dimension of the encoder output.
    hidden_dims : list[int]
        Hidden layer widths.
    dropout : float
        Dropout probability.
    huber_delta : float
        Delta parameter for Huber (smooth L1) loss.
        Setting to 1.0 gives standard Huber; larger values approach MSE.
    activation : str
        Non-linearity: gelu | relu | silu | leaky_relu | tanh.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: List[int] = (64, 32),
        dropout: float = 0.1,
        huber_delta: float = 0.5,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta

        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(_ACTIVATIONS)}")

        layers: List[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_cls(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        embedding: torch.Tensor,          # (B, embed_dim)
        z_true: Optional[torch.Tensor] = None,  # (B,) — needed for loss
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        z_pred = self.net(embedding).squeeze(-1)  # (B,)

        result: Dict[str, torch.Tensor] = {"z_pred": z_pred}

        if z_true is not None:
            result["loss"] = F.huber_loss(z_pred, z_true, delta=self.huber_delta)

        return result
