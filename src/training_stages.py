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

# Training fields a stage block may override (None → inherit top-level training.*).
STAGE_TRAINING_OVERRIDE_FIELDS = (
    "epochs", "lr", "weight_decay", "lr_scheduler", "warmup_epochs",
    "clip_grad_norm", "batch_size", "early_stop_patience", "early_stop_min_epoch",
    "fisher_lambda", "spread_lambda", "huber_lambda", "huber_delta",
    "val_dropout", "full_filter_epochs", "dropout_resume_lr_mult",
)


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

    for attr in STAGE_TRAINING_OVERRIDE_FIELDS:
        val = getattr(stage, attr)
        if val is not None:
            setattr(c.training, attr, val)
    if stage.use_coverage is not None:
        c.model.use_coverage = stage.use_coverage
    return c


def format_stage_training_summary(tc) -> str:
    """One-line summary of effective training hyperparameters for a stage."""
    parts = [
        f"lr={tc.lr:.2e}",
        f"wd={tc.weight_decay:.2e}",
        f"sched={tc.lr_scheduler}",
        f"bs={tc.batch_size}",
    ]
    if tc.warmup_epochs:
        parts.append(f"warmup={tc.warmup_epochs}")
    if tc.clip_grad_norm > 0:
        parts.append(f"clip={tc.clip_grad_norm}")
    return "  ".join(parts)


def freeze_module(mod: nn.Module) -> None:
    for p in mod.parameters():
        p.requires_grad = False


def unfreeze_module(mod: nn.Module) -> None:
    for p in mod.parameters():
        p.requires_grad = True


def load_encoder_weights(model: nn.Module, checkpoint: Path) -> None:
    """Load encoder.* and optional bottleneck.* from a full DeepSetZ checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    enc_state = {
        k.replace("encoder.", "", 1): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    if not enc_state:
        raise ValueError(f"No encoder weights found in {checkpoint}")
    model.encoder.load_state_dict(enc_state)

    bottle_state = {
        k.replace("bottleneck.", "", 1): v
        for k, v in state.items()
        if k.startswith("bottleneck.")
    }
    if bottle_state:
        if getattr(model, "bottleneck", None) is None:
            raise ValueError(
                f"Checkpoint {checkpoint.name} contains bottleneck weights but "
                "the model was built with bottleneck disabled."
            )
        model.bottleneck.load_state_dict(bottle_state)


def encoder_param_count(model: nn.Module) -> int:
    n = sum(p.numel() for p in model.encoder.parameters())
    if getattr(model, "bottleneck", None) is not None:
        n += sum(p.numel() for p in model.bottleneck.parameters())
    return n


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
