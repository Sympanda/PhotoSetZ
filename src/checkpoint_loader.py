"""
Build DeepSetZ models compatible with saved checkpoints (old and new layouts).

When a run's config.yaml predates newer fields (e.g. use_coverage), dataclass
defaults can disagree with the weights on disk.  This module inspects the
state dict and aligns config flags before building the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.config import Config
from src.coverage_summary import COVERAGE_SUMMARY_DIM
from src.model_dims import resolve_bottleneck_dim, representation_dim
from src.train import DeepSetZ, build_encoder, build_model


def _first_head_trunk_in_features(state_dict: dict) -> Optional[int]:
    """Input dim of the first head MLP layer (trunk or net)."""
    best: Optional[int] = None
    for key, tensor in state_dict.items():
        if tensor.ndim != 2:
            continue
        if key.endswith("trunk.0.weight") or key.endswith("net.0.weight"):
            return int(tensor.shape[1])
        if key.startswith("head.") and key.endswith(".0.weight") and best is None:
            best = int(tensor.shape[1])
    return best


def _infer_token_dim(state_dict: dict, cfg: Config) -> None:
    for key, tensor in state_dict.items():
        if tensor.ndim != 2:
            continue
        if "pre_embed.0.weight" in key or "phi.0.weight" in key:
            cfg.model.token_dim = int(tensor.shape[1])
            return


def _infer_bottleneck_latent(state_dict: dict) -> Optional[int]:
    """Latent dim from the last bottleneck Linear weight (out_features)."""
    last_out: Optional[int] = None
    for key, tensor in state_dict.items():
        if key.startswith("bottleneck.") and key.endswith(".weight") and tensor.ndim == 2:
            last_out = int(tensor.shape[0])
    return last_out


def align_cfg_to_checkpoint(
    cfg: Config,
    state_dict: dict,
    head_type: str | None = None,
) -> Config:
    """
    Adjust cfg.model layout flags so build_model matches checkpoint shapes.

    Mutates *cfg* in place and returns it.
    """
    ht = head_type or cfg.head.type
    _infer_token_dim(state_dict, cfg)

    bottleneck_latent = _infer_bottleneck_latent(state_dict)
    if bottleneck_latent is not None:
        cfg.model.bottleneck = bottleneck_latent
    else:
        cfg.model.bottleneck = False

    enc_dim = build_encoder(cfg).output_dim
    repr_dim = representation_dim(cfg, enc_dim)
    has_density_net = any(k.startswith("density_context_net.") for k in state_dict)
    head_in = _first_head_trunk_in_features(state_dict)

    if has_density_net:
        cfg.model.density_context_branch = True
        cfg.model.use_coverage_summary = True
        cfg.model.use_coverage = False
        return cfg

    cfg.model.density_context_branch = False
    cfg.model.use_coverage_summary = False

    if head_in is None:
        return cfg

    delta = head_in - repr_dim
    if delta == 0:
        cfg.model.use_coverage = False
    elif delta == 1:
        cfg.model.use_coverage = True
    elif delta == COVERAGE_SUMMARY_DIM and ht == "nsf":
        cfg.model.use_coverage = False
        cfg.model.use_coverage_summary = True
    elif delta == enc_dim + COVERAGE_SUMMARY_DIM:
        # Unlikely mis-parse; keep config as-is.
        pass
    else:
        # Best effort: prefer legacy scalar coverage when close.
        if delta == 1:
            cfg.model.use_coverage = True
        elif delta == 0:
            cfg.model.use_coverage = False

    return cfg


def load_model_from_checkpoint(
    cfg: Config,
    checkpoint_path: str | Path,
    *,
    head_type: str | None = None,
    device: torch.device | None = None,
    n_total_filters: int | None = None,
    strict: bool = True,
):
    """
    Build a model, align config to checkpoint shapes, and load weights.

    Returns ``DeepSetZ`` or ``SBIStage2Model`` depending on checkpoint type.
    """
    ckpt = Path(checkpoint_path)
    try:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)

    ht = head_type or cfg.head.type
    is_sbi = ht == "sbi_npe" or any(k.startswith("sbi_head.") for k in state_dict)

    if is_sbi:
        from src.sbi_training import build_sbi_stage2_model
        cfg.head.type = "sbi_npe"
        dev = device or torch.device("cpu")
        model = build_sbi_stage2_model(cfg, dev, n_total_filters or 1, stage1_state=None)
        model.load_state_dict(state_dict, strict=strict)
        if device is not None:
            model = model.to(device)
        return model

    align_cfg_to_checkpoint(cfg, state_dict, head_type=head_type)
    model = build_model(
        cfg,
        device=device,
        head_type=head_type,
        n_total_filters=n_total_filters,
    )
    model.load_state_dict(state_dict, strict=strict)
    return model
