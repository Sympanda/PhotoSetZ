"""Helpers for split-training stages and encoder checkpoint transfer."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.config import Config, StageTrainingConfig
from src.run_artifacts import CKPT_POINT


def apply_stage_overrides(
    cfg: Config,
    stage: StageTrainingConfig,
    *,
    default_head: Optional[str] = None,
) -> Config:
    """Deep-copy *cfg* and apply per-stage training / head overrides."""
    c = copy.deepcopy(cfg)
    if stage.head:
        c.head.type = stage.head
    elif default_head:
        c.head.type = default_head

    for attr in (
        "epochs", "lr", "warmup_epochs", "early_stop_patience",
        "early_stop_min_epoch", "fisher_lambda", "spread_lambda", "huber_lambda",
    ):
        val = getattr(stage, attr)
        if val is not None:
            setattr(c.training, attr, val)
    if stage.use_coverage is not None:
        c.model.use_coverage = stage.use_coverage
    return c


def freeze_module(mod: nn.Module) -> None:
    for p in mod.parameters():
        p.requires_grad = False


def unfreeze_module(mod: nn.Module) -> None:
    for p in mod.parameters():
        p.requires_grad = True


def load_encoder_weights(model: nn.Module, checkpoint: Path) -> None:
    """Load encoder.* tensors from a full DeepSetZ state dict."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    enc_state = {
        k.replace("encoder.", "", 1): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    if not enc_state:
        raise ValueError(f"No encoder weights found in {checkpoint}")
    model.encoder.load_state_dict(enc_state)


def encoder_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.encoder.parameters())


def resolve_stage1_checkpoint(path: str | Path, root: Path) -> Path:
    """Resolve and validate a stage-1 encoder checkpoint path."""
    ckpt = Path(path)
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"stage1_checkpoint not found: {ckpt}")
    return ckpt


def seed_stage1_artifacts(out_dir: Path, source_ckpt: Path) -> None:
    """
    Copy stage-1 artefacts into a new run dir so the notebook can load
    Point (MLP) alongside a freshly trained stage 2.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_ckpt, out_dir / CKPT_POINT)

    run_dir = source_ckpt.parent
    for name in (
        "history_stage1.json",
        "test_metrics_point.json",
        "subset_metrics_stage1.json",
        "predictions_stage1.npz",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    src_plots = run_dir / "plots" / "stage1"
    if src_plots.is_dir():
        dst = out_dir / "plots" / "stage1"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src_plots.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
