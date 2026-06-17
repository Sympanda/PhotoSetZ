"""Helpers for encoder / bottleneck / head input dimensions."""

from __future__ import annotations

from typing import Optional

from src.config import Config, ModelConfig


def resolve_bottleneck_dim(model_cfg: ModelConfig) -> Optional[int]:
    """
    Return latent bottleneck dimension, or None when disabled.

    Config values
    -------------
    bottleneck: false  → disabled (default, backwards compatible)
    bottleneck: 64     → 3-layer MLP to 64-d latent
    """
    b = model_cfg.bottleneck
    if isinstance(b, bool):
        return None if not b else _raise_bool_bottleneck()
    if isinstance(b, int):
        if b <= 0:
            return None
        return b
    return None


def _raise_bool_bottleneck() -> None:
    raise ValueError(
        "bottleneck: true is invalid — set a latent dimension, e.g. bottleneck: 64"
    )


def representation_dim(cfg: Config, encoder_output_dim: int) -> int:
    """Dimension fed to heads / density context (after optional bottleneck)."""
    latent = resolve_bottleneck_dim(cfg.model)
    return latent if latent is not None else encoder_output_dim
