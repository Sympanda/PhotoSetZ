"""Build and evaluate SBI/NPE stage-2 models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.model_dims import representation_dim
from src.models.heads.mlp_regressor import MLPRegressor
from src.models.heads.sbi_context import SBIContextBuilder
from src.models.heads.sbi_npe import SBINPEHead
from src.models.sbi_stage2 import SBIStage2Model
from src.training_stages import apply_stage_overrides


def build_sbi_stage2_model(
    cfg: Config,
    device: torch.device,
    n_total_filters: int,
    stage1_state: dict | None = None,
) -> SBIStage2Model:
    """Build frozen compressor + trainable SBI/NPE head for stage 2."""
    from src.train import build_model

    sc = cfg.head.sbi_npe
    cfg_point = apply_stage_overrides(cfg, cfg.training.stage1, default_head="mlp_regressor")

    base = build_model(
        cfg_point, device=device, head_type="mlp_regressor", n_total_filters=n_total_filters,
    )
    if stage1_state is not None:
        base.load_state_dict(stage1_state, strict=False)

    if not isinstance(base.head, MLPRegressor):
        raise TypeError("Stage 1 must use mlp_regressor for SBI/NPE pipeline.")

    repr_dim = representation_dim(cfg_point, base.encoder.output_dim)
    cp = sc.context_projection
    ctx_builder = SBIContextBuilder(
        context_mode=sc.context_mode,
        repr_dim=repr_dim,
        include_point_prediction=sc.include_point_prediction,
        include_coverage_summary=sc.include_coverage_summary,
        context_projection_enabled=cp.enabled,
        projection_hidden=cp.hidden_dims,
        projection_dropout=cp.dropout,
        projection_layer_norm=cp.layer_norm,
        context_dim=sc.context_dim,
        n_total_filters=n_total_filters,
        include_errors=cfg.data.include_errors,
        add_detection_flags=cfg.data.add_detection_flags,
    ).to(device)

    sbi_head = SBINPEHead(
        context_dim=ctx_builder.context_dim,
        density_estimator=sc.density_estimator,
        hidden_features=sc.hidden_features,
        num_transforms=sc.num_transforms,
        num_bins=sc.num_bins,
        target_transform_mode=sc.target_transform,
        y_min=sc.y_min,
        y_max=sc.y_max,
        eps=sc.eps,
        z_max_eval=sc.grid_eval.z_max_eval,
        n_grid=sc.grid_eval.n_grid,
    ).to(device)
    sbi_head.ensure_flow(device)

    model = SBIStage2Model(
        encoder=base.encoder,
        point_head=base.head,
        sbi_head=sbi_head,
        context_builder=ctx_builder,
        bottleneck=base.bottleneck,
        use_coverage=cfg_point.model.use_coverage,
        n_total_filters=n_total_filters,
        log_target=cfg.data.log_target,
    ).to(device)

    if sc.freeze_encoder:
        model.freeze_compressor()
    return model


def sbi_stage2_optimizer(model: SBIStage2Model, cfg: Config):
    """AdamW over trainable SBI/context parameters only."""
    from torch.optim import AdamW
    sc = cfg.head.sbi_npe
    tc = cfg.training
    lr = sc.lr if sc.lr is not None else tc.lr
    wd = sc.weight_decay if sc.weight_decay is not None else tc.weight_decay
    return AdamW(model.trainable_parameters(), lr=lr, weight_decay=wd)


@torch.no_grad()
def collect_sbi_prob_outputs(
    model: SBIStage2Model,
    loader: DataLoader,
    device: torch.device,
    log_target: bool = False,
    temperature: float = 1.0,
) -> dict:
    """Collect grid-based posterior summaries for SBI/NPE stage-2 model."""
    model.eval()
    head = model.sbi_head
    y_max = head.target_transform.y_max
    z_max_phys = float(np.expm1(y_max)) if head.target_transform.mode.startswith("log1p") else y_max
    z_grid = head.z_grid(device, z_min=0.0, z_max=z_max_phys)

    all_z_true, all_mean, all_med, all_mode, all_pit = [], [], [], [], []

    bar = tqdm(loader, desc="  └─ collecting SBI posteriors", unit="batch", leave=False)
    for tokens, mask, z_true in bar:
        tokens, mask = tokens.to(device), mask.to(device)
        out = model(tokens, mask)
        h = model._encode(tokens, mask)
        ctx = model.context_builder(h, tokens, mask, out["z_pred"])

        _, pdf = head.evaluate_grid(z_grid, ctx, temperature=temperature)
        ests = head.point_estimates_from_grid(z_grid, pdf)
        z_phys = torch.expm1(z_true.to(device)) if log_target else z_true.to(device)
        pit = head.pit_values(z_grid, pdf, z_phys)

        all_z_true.append(z_true.cpu())
        all_mean.append(ests["z_mean"].cpu())
        all_med.append(ests["z_median"].cpu())
        all_mode.append(ests["z_mode"].cpu())
        all_pit.append(pit.cpu())

    def _to_real(t: torch.Tensor) -> np.ndarray:
        arr = t.numpy()
        return np.expm1(arr) if log_target else arr

    z_true_raw = torch.cat(all_z_true)
    return {
        "z_true": _to_real(z_true_raw),
        "z_mean": torch.cat(all_mean).numpy(),
        "z_median": torch.cat(all_med).numpy(),
        "z_mode": torch.cat(all_mode).numpy(),
        "pit": torch.cat(all_pit).numpy(),
        "head_type": "SBI_NPE",
    }


def load_stage1_state_dict(checkpoint: Path) -> dict:
    return torch.load(checkpoint, map_location="cpu", weights_only=False)
