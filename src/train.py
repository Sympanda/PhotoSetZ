"""
Training script for DeepSetZ.

Usage
-----
    python src/train.py configs/deepsets.yaml
    python src/train.py configs/set_transformer.yaml --run_name my_run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.platform_fix  # noqa: F401  — before torch on macOS

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import Config, load_config, save_config
from src.dataset import GalaxyDataset, DropoutConfig, collate_fn
from src.data_options import resolve_data_options, dataset_kwargs_from_config, compute_token_dim
from src.coverage_summary import compute_coverage_summary, COVERAGE_SUMMARY_DIM
from src.density_context import DensityContextNet
from src.model_dims import resolve_bottleneck_dim, representation_dim
from src.models.bottleneck import EncoderBottleneck
from src.models.deepsets import DeepSets
from src.models.set_transformer import SetTransformer
from src.models.heads import HEAD_REGISTRY
from src.evaluate import compute_metrics
from src.plot import (
    generate_all_plots,
    evaluate_survey_subsets,
    plot_calibration,
    plot_calibration_comparison,
    plot_delta_z,
    plot_scatter,
)
from src.dataloader_utils import resolve_num_workers, shutdown_dataloaders
from src.calibration import run_post_hoc_calibration
from src.run_artifacts import (
    CKPT_END_TO_END, CKPT_POINT, CKPT_POSTERIOR, CKPT_SBI_NPE,
    ROLE_END_TO_END, ROLE_POSTERIOR,
    post_hoc_sigma_scale,
)
from src.training_stages import (
    apply_stage_overrides,
    format_stage_training_summary,
    freeze_module,
    load_encoder_weights,
    resolve_stage1_checkpoint,
    seed_stage1_artifacts,
)


# ------------------------------------------------------------------
# Model factory
# ------------------------------------------------------------------

def build_encoder(cfg: Config) -> nn.Module:
    mc = cfg.model
    if mc.type == "deepsets":
        dc = mc.deepsets
        return DeepSets(
            token_dim   = mc.token_dim,
            phi_hidden  = dc.phi_hidden,
            latent_dim  = dc.latent_dim,
            rho_hidden  = dc.rho_hidden,
            embed_dim   = dc.embed_dim,
            pooling     = dc.pooling,
            dropout     = dc.dropout,
            activation  = dc.activation,
        )
    elif mc.type == "set_transformer":
        sc = mc.set_transformer
        return SetTransformer(
            token_dim         = mc.token_dim,
            embed_dim         = sc.embed_dim,
            n_heads           = sc.n_heads,
            n_attn_layers     = sc.n_attn_layers,
            ffn_dim           = sc.ffn_dim,
            n_pma_seeds       = sc.n_pma_seeds,
            dropout           = sc.dropout,
            pre_embed_hidden  = sc.pre_embed_hidden,
        )
    else:
        raise ValueError(f"Unknown model type: {mc.type}")


def _nsf_uses_density_context(cfg: Config, head_type: str | None = None) -> bool:
    ht = head_type or cfg.head.type
    if ht != "nsf":
        return False
    mc = cfg.model
    return mc.density_context_branch or mc.use_coverage_summary


def _point_head_extra_dim(cfg: Config) -> int:
    return 1 if cfg.model.use_coverage else 0


def _density_context_input_dim(cfg: Config, encoder_dim: int) -> int:
    d = encoder_dim
    if cfg.model.use_coverage_summary:
        d += COVERAGE_SUMMARY_DIM
    return d


def build_head(
    cfg: Config,
    embed_dim: int,
    head_type: str | None = None,
    *,
    for_density_context: bool = False,
) -> nn.Module:
    hc = cfg.head
    head_type = head_type or hc.type
    head_cls = HEAD_REGISTRY[head_type]

    if for_density_context and head_type == "nsf":
        if cfg.model.density_context_branch:
            in_dim = embed_dim  # output of DensityContextNet
        else:
            in_dim = _density_context_input_dim(cfg, embed_dim)
    else:
        in_dim = embed_dim + _point_head_extra_dim(cfg)

    if head_type == "mlp_regressor":
        c = hc.mlp_regressor
        return head_cls(embed_dim=in_dim, hidden_dims=c.hidden_dims,
                        dropout=c.dropout, huber_delta=c.huber_delta,
                        activation=c.activation)
    elif head_type == "binned_pdf":
        c = hc.binned_pdf
        return head_cls(embed_dim=in_dim, n_bins=c.n_bins,
                        z_min=c.z_min, z_max=c.z_max,
                        hidden_dims=c.hidden_dims, dropout=c.dropout,
                        activation=c.activation)
    elif head_type == "mdn":
        c = hc.mdn
        return head_cls(embed_dim=in_dim, n_components=c.n_components,
                        hidden_dims=c.hidden_dims, dropout=c.dropout,
                        sigma_min=c.sigma_min, activation=c.activation)
    elif head_type == "nsf":
        c = hc.nsf
        return head_cls(embed_dim=in_dim, n_bins=c.n_bins,
                        z_min=c.z_min, z_max=c.z_max,
                        hidden_dims=c.hidden_dims, dropout=c.dropout,
                        activation=c.activation, deriv_min=c.deriv_min)
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def build_model(
    cfg: Config,
    device: torch.device | None = None,
    head_type: str | None = None,
    n_total_filters: int | None = None,
) -> DeepSetZ:
    """Build encoder + head with config-consistent input dims and training flags."""
    data_opts = resolve_data_options(cfg.data)
    expected_td = data_opts.token_dim
    if cfg.model.token_dim != expected_td:
        print(
            f"  [info] auto-setting model.token_dim {cfg.model.token_dim} → {expected_td} "
            f"(include_errors={data_opts.include_errors}, "
            f"add_detection_flags={data_opts.add_detection_flags})"
        )
        cfg.model.token_dim = expected_td
    elif cfg.data.include_errors and cfg.model.token_dim == 4:
        print("  [info] include_errors=True → auto-setting model.token_dim to 5")
        cfg.model.token_dim = 5

    encoder = build_encoder(cfg)
    enc_dim = encoder.output_dim
    latent_dim = resolve_bottleneck_dim(cfg.model)
    repr_dim = representation_dim(cfg, enc_dim)
    ht = head_type or cfg.head.type
    uses_density = _nsf_uses_density_context(cfg, ht)

    bottleneck = None
    if latent_dim is not None:
        bottleneck = EncoderBottleneck(
            enc_dim, latent_dim, dropout=cfg.model.bottleneck_dropout,
        )
        print(f"  Encoder bottleneck: {enc_dim} → {latent_dim} (3-layer MLP)")

    density_context_net = None
    if uses_density and cfg.model.density_context_branch:
        ctx_in = _density_context_input_dim(cfg, repr_dim)
        density_context_net = DensityContextNet(
            in_dim=ctx_in,
            out_dim=repr_dim,
            hidden_dims=cfg.model.density_context_hidden,
            dropout=cfg.model.density_context_dropout,
        )

    head = build_head(
        cfg, repr_dim, head_type=ht,
        for_density_context=uses_density,
    )
    tc = cfg.training
    model = DeepSetZ(
        encoder,
        head,
        bottleneck=bottleneck,
        use_coverage=cfg.model.use_coverage,
        use_coverage_summary=cfg.model.use_coverage_summary,
        density_context_branch=cfg.model.density_context_branch,
        density_context_net=density_context_net,
        n_total_filters=n_total_filters or 1,
        include_errors=data_opts.include_errors,
        add_detection_flags=data_opts.add_detection_flags,
        fisher_lambda=tc.fisher_lambda,
        spread_lambda=tc.spread_lambda,
        huber_lambda=tc.huber_lambda,
        huber_delta=tc.huber_delta,
    )
    if device is not None:
        model = model.to(device)
    return model


# ------------------------------------------------------------------
# Full model wrapper
# ------------------------------------------------------------------

class DeepSetZ(nn.Module):
    """Encoder + prediction head."""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        bottleneck: nn.Module | None = None,
        use_coverage: bool = False,
        use_coverage_summary: bool = False,
        density_context_branch: bool = False,
        density_context_net: nn.Module | None = None,
        n_total_filters: int = 1,
        include_errors: bool = False,
        add_detection_flags: bool = False,
        fisher_lambda: float = 0.0,
        spread_lambda: float = 0.0,
        huber_lambda: float = 0.0,
        huber_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder                = encoder
        self.bottleneck             = bottleneck
        self.head                   = head
        self.use_coverage          = use_coverage
        self.use_coverage_summary  = use_coverage_summary
        self.density_context_branch = density_context_branch
        self.density_context_net   = density_context_net
        self.n_total_filters       = max(int(n_total_filters), 1)
        self.include_errors        = include_errors
        self.add_detection_flags   = add_detection_flags
        self.fisher_lambda         = fisher_lambda
        self.spread_lambda         = spread_lambda
        self.huber_lambda          = huber_lambda
        self.huber_delta           = huber_delta

    def _coverage_scalar(self, key_padding_mask: torch.Tensor) -> torch.Tensor:
        n_active = (~key_padding_mask).float().sum(dim=-1, keepdim=True)
        return n_active / self.n_total_filters

    def _nsf_density_context(
        self,
        h_set: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts = [h_set]
        if self.use_coverage_summary:
            h_cov = compute_coverage_summary(
                tokens,
                key_padding_mask,
                n_total_filters=self.n_total_filters,
                include_errors=self.include_errors,
                add_detection_flags=self.add_detection_flags,
            )
            parts.append(h_cov)
        ctx_in = torch.cat(parts, dim=-1)
        if self.density_context_branch and self.density_context_net is not None:
            return self.density_context_net(ctx_in)
        return ctx_in

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        z_true: torch.Tensor | None = None,
    ) -> dict:
        h_set = self.encoder(tokens, key_padding_mask)
        if self.bottleneck is not None:
            h_set = self.bottleneck(h_set)

        from src.models.heads.mlp_regressor import MLPRegressor
        from src.models.heads.nsf import NeuralSplineFlow

        uses_nsf_density = (
            isinstance(self.head, NeuralSplineFlow)
            and (self.density_context_branch or self.use_coverage_summary)
        )

        if uses_nsf_density:
            head_embedding = self._nsf_density_context(h_set, tokens, key_padding_mask)
        else:
            head_embedding = h_set
            if self.use_coverage:
                head_embedding = torch.cat(
                    [head_embedding, self._coverage_scalar(key_padding_mask)],
                    dim=-1,
                )

        head_kwargs: dict = {"z_true": z_true}
        if not isinstance(self.head, MLPRegressor):
            head_kwargs.update(
                fisher_lambda=self.fisher_lambda,
                spread_lambda=self.spread_lambda,
                huber_lambda=self.huber_lambda,
                huber_delta=self.huber_delta,
            )
        return self.head(head_embedding, **head_kwargs)


# ------------------------------------------------------------------
# Training utilities
# ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: DeepSetZ,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    epoch: int,
    n_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0

    bar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{n_epochs} [train]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for tokens, mask, z_true in bar:
        tokens = tokens.to(device)
        mask   = mask.to(device)
        z_true = z_true.to(device)

        optimiser.zero_grad()
        out  = model(tokens, mask, z_true=z_true)
        loss = out["loss"]
        loss.backward()

        if cfg.training.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)

        optimiser.step()
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss/max(bar.n,1):.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: DeepSetZ,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval",
    return_preds: bool = False,
    log_target: bool = False,
) -> dict | tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate model on *loader*.

    When log_target=True the dataset returns log(1+z) targets and the model
    predicts in that space.  We convert both predictions and targets back to
    real z with expm1 **before** computing metrics and returning arrays, so
    all downstream code (plots, printed tables) always operates in real z.
    """
    model.eval()
    z_preds, z_trues = [], []
    total_loss = 0.0

    bar = tqdm(loader, desc=f"  └─ {desc}", unit="batch", leave=False, dynamic_ncols=True)
    for tokens, mask, z_true in bar:
        tokens = tokens.to(device)
        mask   = mask.to(device)
        z_true = z_true.to(device)

        out = model(tokens, mask, z_true=z_true)
        if "loss" in out:
            total_loss += out["loss"].item()
        z_preds.append(out["z_pred"].cpu())
        z_trues.append(z_true.cpu())

    z_preds = torch.cat(z_preds).numpy()
    z_trues = torch.cat(z_trues).numpy()

    # Convert back to real redshift space before metrics / storage
    if log_target:
        z_preds = np.expm1(z_preds)
        z_trues = np.expm1(z_trues)

    metrics = compute_metrics(z_preds, z_trues)
    nll = total_loss / max(len(loader), 1)
    metrics["loss"] = nll   # backward-compatible alias
    metrics["nll"]  = nll

    if return_preds:
        return metrics, z_preds, z_trues
    return metrics


@torch.no_grad()
def collect_prob_outputs(
    model: "DeepSetZ",
    loader: DataLoader,
    device: torch.device,
    log_target: bool = False,
    sigma_scale: float = 1.0,
) -> dict | None:
    """
    Collect probabilistic point estimates and PIT values for MDN / BinnedPDF heads.
    Returns None for MLP regressor (no distribution to summarise).

    The returned dict (all arrays in *real* redshift space except pit) has keys:
        z_true    : (N,) spectroscopic redshifts
        z_mean    : (N,) mean of the predicted distribution
        z_median  : (N,) median of the predicted distribution
        z_mode    : (N,) mode  of the predicted distribution
        pit       : (N,) Probability Integral Transform values ∈ [0, 1]
        head_type : str  ("MDN", "BinnedPDF", "NSF", or "SBI_NPE")
    """
    from src.models.sbi_stage2 import SBIStage2Model
    if isinstance(model, SBIStage2Model):
        from src.sbi_training import collect_sbi_prob_outputs
        return collect_sbi_prob_outputs(
            model, loader, device, log_target=log_target, temperature=sigma_scale,
        )

    from src.models.heads.mdn        import MDN
    from src.models.heads.binned_pdf import BinnedPDF
    from src.models.heads.nsf        import NeuralSplineFlow

    head = model.head
    if isinstance(head, MDN):
        head_type = "MDN"
    elif isinstance(head, BinnedPDF):
        head_type = "BinnedPDF"
    elif isinstance(head, NeuralSplineFlow):
        head_type = "NSF"
    else:
        return None

    model.eval()

    all_z_raw:   list = []
    all_pi:      list = []
    all_mu:      list = []
    all_sigma:   list = []
    all_probs:   list = []
    all_widths:  list = []
    all_heights: list = []
    all_derivs:  list = []

    from src.calibration import apply_sigma_scale_to_out

    bar = tqdm(loader, desc="  └─ collecting posteriors", unit="batch",
               leave=False, dynamic_ncols=True)
    for tokens, mask, z_true in bar:
        out = apply_sigma_scale_to_out(
            model(tokens.to(device), mask.to(device)), head, sigma_scale,
        )
        all_z_raw.append(z_true.cpu())
        if head_type == "MDN":
            all_pi.append(out["pi"].cpu())
            all_mu.append(out["mu"].cpu())
            all_sigma.append(out["sigma"].cpu())
        elif head_type == "BinnedPDF":
            all_probs.append(out["probs"].cpu())
        else:  # NSF
            all_widths.append(out["widths"].cpu())
            all_heights.append(out["heights"].cpu())
            all_derivs.append(out["derivs"].cpu())

    z_true_raw = torch.cat(all_z_raw)   # (N,)  — log(1+z) if log_target

    if head_type == "MDN":
        pi    = torch.cat(all_pi)
        mu    = torch.cat(all_mu)
        sigma = torch.cat(all_sigma)
        ests  = head.point_estimates_from_params(pi, mu, sigma)
        pit   = head.pit_values(pi, mu, sigma, z_true_raw).numpy()
    elif head_type == "BinnedPDF":
        probs = torch.cat(all_probs)
        ests  = head.point_estimates_from_probs(probs)
        pit   = head.pit_values(probs, z_true_raw).numpy()
    else:  # NSF
        widths  = torch.cat(all_widths)
        heights = torch.cat(all_heights)
        derivs  = torch.cat(all_derivs)
        ests    = head.point_estimates_from_params(widths, heights, derivs)
        from src.calibration import pit_nsf_temperature
        if sigma_scale != 1.0:
            pit = pit_nsf_temperature(
                head, widths, heights, derivs, z_true_raw, temperature=sigma_scale,
            )
        else:
            pit = head.pit_values(widths, heights, derivs, z_true_raw).numpy()

    def _to_real(t: torch.Tensor) -> np.ndarray:
        return np.expm1(t.numpy()) if log_target else t.numpy()

    return {
        "z_true":    _to_real(z_true_raw),
        "z_mean":    _to_real(ests["z_mean"]),
        "z_median":  _to_real(ests["z_median"]),
        "z_mode":    _to_real(ests["z_mode"]),
        "pit":       pit,
        "head_type": head_type,
    }


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(cfg: Config) -> None:
    set_seed(cfg.training.seed)
    device = get_device()
    print(f"Device: {device}")

    # Resolve paths relative to repo root
    root = Path(__file__).parent.parent
    train_path = root / cfg.data.train_path
    test_path  = root / cfg.data.test_path
    res_dir    = root / cfg.data.res_dir
    out_dir    = root / cfg.output_dir / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    print("\nLoading datasets …")
    dc = cfg.training.dropout
    train_dropout = DropoutConfig(
        p_complete               = dc.p_complete,
        p_preset                 = dc.p_preset,
        p_survey_drop            = dc.p_survey_drop,
        p_aggressive             = dc.p_aggressive,
        aggressive_rate_low      = dc.aggressive_rate_low,
        aggressive_rate_high     = dc.aggressive_rate_high,
        max_surveys_to_drop      = dc.max_surveys_to_drop,
        p_filter_after_survey    = dc.p_filter_after_survey,
        filter_rate_after_survey = dc.filter_rate_after_survey,
        min_filters              = dc.min_filters,
    )
    active         = cfg.data.active_surveys or None
    log_target     = cfg.data.log_target
    data_kw = dataset_kwargs_from_config(cfg.data)

    train_ds = GalaxyDataset(
        parquet_path=train_path,
        res_dir=res_dir,
        target_col=cfg.data.target_col,
        dropout_cfg=train_dropout,
        active_surveys=active,
        log_target=log_target,
        **data_kw,
    )
    test_ds = GalaxyDataset(
        parquet_path=test_path,
        res_dir=res_dir,
        target_col=cfg.data.target_col,
        dropout_cfg=None,
        active_surveys=active,
        log_target=log_target,
        **data_kw,
    )

    val_ds_full = GalaxyDataset(
        parquet_path=train_path,
        res_dir=res_dir,
        target_col=cfg.data.target_col,
        dropout_cfg=None,
        active_surveys=active,
        log_target=log_target,
        **data_kw,
    )

    val_ds_drop = GalaxyDataset(
        parquet_path=train_path,
        res_dir=res_dir,
        target_col=cfg.data.target_col,
        dropout_cfg=cfg.training.dropout,
        active_surveys=active,
        log_target=log_target,
        **data_kw,
    )

    # Deterministic 90/10 index split
    n_total = len(train_ds)
    n_val   = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    rng = torch.Generator().manual_seed(cfg.training.seed)
    all_idx = torch.randperm(n_total, generator=rng).tolist()
    train_idx, val_idx = all_idx[:n_train], all_idx[n_train:]

    from torch.utils.data import Subset
    train_sub    = Subset(train_ds,     train_idx)
    val_sub_full = Subset(val_ds_full,  val_idx)
    val_sub_drop = Subset(val_ds_drop,  val_idx)

    tc = cfg.training
    print(f"  Train: {n_train:,}  Val: {n_val:,}  Test: {len(test_ds):,}")
    if tc.val_dropout:
        print(f"  Val dropout: enabled  |  fisher_λ={tc.fisher_lambda}  spread_λ={tc.spread_lambda}")
    print(f"  Filters in registry: {train_ds.n_filters}")

    # Dropout distribution sanity check
    drop_stats = train_ds.sample_dropout_stats(n_samples=5_000)
    print(f"  Dropout stats (5k samples): "
          f"mean={drop_stats['mean_filters']:.1f} filters, "
          f"min={drop_stats['min_filters']}, "
          f"max={drop_stats['max_filters']}, "
          f"complete={drop_stats['pct_complete']*100:.0f}%, "
          f"≤3={drop_stats['pct_3_or_fewer']*100:.0f}%")

    save_config(cfg, out_dir / "config.yaml")
    print(f"\nOutputs → {out_dir}")

    if tc.stage1_checkpoint and not tc.split_training:
        raise ValueError("stage1_checkpoint requires split_training: true")
    if _is_sbi_head(cfg.head.type) and not tc.split_training:
        raise ValueError(
            "head.type: sbi_npe requires split_training: true (stage 1 MLP → stage 2 SBI/NPE)."
        )

    if tc.split_training:
        _train_split_pipeline(
            cfg, tc, device, log_target, out_dir,
            train_sub, val_sub_full, val_sub_drop,
            train_ds, test_ds, train_dropout,
        )
    else:
        train_loader, val_loader, val_loader_drop, test_loader = _build_dataloaders(
            tc, device, train_sub, val_sub_full, val_sub_drop, test_ds,
        )
        try:
            _train_end_to_end(
                cfg, tc, device, log_target, out_dir,
                train_loader, val_loader, val_loader_drop, test_loader,
                train_ds, test_ds, train_dropout,
            )
        finally:
            shutdown_dataloaders(train_loader, val_loader, val_loader_drop, test_loader)


def _build_dataloaders(tc, device, train_sub, val_sub_full, val_sub_drop, test_ds):
    """Build train/val/test loaders from stage-effective training config."""
    nw  = resolve_num_workers(tc.num_workers)
    pin = device.type == "cuda"
    bs  = tc.batch_size
    train_loader = DataLoader(
        train_sub, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=nw, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_sub_full, batch_size=bs * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=nw,
    )
    val_loader_drop = None
    if tc.val_dropout:
        val_loader_drop = DataLoader(
            val_sub_drop, batch_size=bs * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=nw,
        )
    test_loader = DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=nw,
    )
    return train_loader, val_loader, val_loader_drop, test_loader


def _make_optimiser_and_scheduler(model: DeepSetZ, tc):
    optimiser = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=tc.lr,
        weight_decay=tc.weight_decay,
    )
    if tc.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimiser, T_max=max(tc.epochs - tc.warmup_epochs, 1))
    elif tc.lr_scheduler == "step":
        scheduler = StepLR(optimiser, step_size=30, gamma=0.1)
    else:
        scheduler = None
    return optimiser, scheduler


def _is_prob_head(head_type: str) -> bool:
    return head_type in ("mdn", "nsf", "binned_pdf", "sbi_npe")


def _is_sbi_head(head_type: str) -> bool:
    return head_type == "sbi_npe"


def _train_end_to_end(
    cfg, tc, device, log_target, out_dir,
    train_loader, val_loader, val_loader_drop, test_loader,
    train_ds, test_ds, train_dropout,
) -> None:
    print("\nBuilding model …")
    model = build_model(cfg, device=device, n_total_filters=train_ds.n_filters)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Mode: end-to-end  |  Encoder: {cfg.model.type}  |  Head: {cfg.head.type}")
    print(f"  Parameters: {n_params:,}  |  Device: {device}")

    optimiser, scheduler = _make_optimiser_and_scheduler(model, tc)
    print(f"\n{'='*65}")
    print(f"  Training  {cfg.run_name}  for {tc.epochs} epochs")
    print(f"{'='*65}\n")

    _run_training_loop(
        model, train_loader, val_loader, val_loader_drop, test_loader,
        optimiser, scheduler, device, cfg, tc, log_target, out_dir,
        train_ds, test_ds, train_dropout,
        best_ckpt_name=CKPT_END_TO_END,
        history_name="history.json",
        test_metrics_name="test_metrics.json",
        run_post_hoc=_is_prob_head(cfg.head.type),
        post_hoc_role=ROLE_END_TO_END,
        post_hoc_final_plots=_is_prob_head(cfg.head.type)
            and cfg.training.post_hoc_calibration.enabled,
        stage_title="",
    )


def _train_split_pipeline(
    cfg, tc, device, log_target, out_dir,
    train_sub, val_sub_full, val_sub_drop,
    train_ds, test_ds, train_dropout,
) -> None:
    """Stage 1: encoder + MLP.  Stage 2: frozen encoder + PDF head."""
    root = Path(__file__).parent.parent
    posterior_head = tc.stage2.head or cfg.head.type

    # ── Stage 1: point map (or load from existing checkpoint) ────────
    if tc.stage1_checkpoint:
        stage1_ckpt = resolve_stage1_checkpoint(tc.stage1_checkpoint, root)
        print(f"\nSkipping stage 1 — loading encoder from {stage1_ckpt}")
        seed_stage1_artifacts(out_dir, stage1_ckpt)
        print(f"  Copied stage-1 artefacts → {out_dir / CKPT_POINT}")
    else:
        cfg1 = apply_stage_overrides(cfg, tc.stage1, default_head="mlp_regressor")
        tc1  = cfg1.training
        print("\nBuilding stage-1 model (encoder + point head) …")
        model1 = build_model(
            cfg1, device=device, head_type=cfg1.head.type,
            n_total_filters=train_ds.n_filters,
        )
        n1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
        repr_dim = representation_dim(cfg1, model1.encoder.output_dim)
        head_in = repr_dim + 1 if cfg1.model.use_coverage else repr_dim
        cov = "on (latent+1)" if cfg1.model.use_coverage else "off"
        bn = f"  bottleneck: {repr_dim}d" if model1.bottleneck is not None else ""
        print(f"  Head: {cfg1.head.type}  |  Trainable params: {n1:,}  |  coverage: {cov}  |  head in: {head_in}{bn}")
        print(f"  {format_stage_training_summary(tc1)}")

        opt1, sched1 = _make_optimiser_and_scheduler(model1, tc1)
        train_loader, val_loader, val_loader_drop, test_loader = _build_dataloaders(
            tc1, device, train_sub, val_sub_full, val_sub_drop, test_ds,
        )
        print(f"\n{'='*65}")
        print(f"  Stage 1 — point  ({cfg.run_name})  for {tc1.epochs} epochs")
        print(f"{'='*65}\n")

        try:
            _run_training_loop(
                model1, train_loader, val_loader, val_loader_drop, test_loader,
                opt1, sched1, device, cfg1, tc1, log_target, out_dir,
                train_ds, test_ds, train_dropout,
                best_ckpt_name=CKPT_POINT,
                history_name="history_stage1.json",
                test_metrics_name="test_metrics_point.json",
                subset_metrics_name="subset_metrics_stage1.json",
                predictions_name="predictions_stage1.npz",
                plots_subdir="stage1",
                stage_label="Stage 1 — Point (MLP)",
                run_post_hoc=False,
                stage_title="[stage 1] ",
            )
        finally:
            shutdown_dataloaders(train_loader, val_loader, val_loader_drop, test_loader)
        stage1_ckpt = out_dir / CKPT_POINT

    # ── Stage 2: posterior on frozen encoder ─────────────────────────
    cfg2 = apply_stage_overrides(cfg, tc.stage2, default_head=posterior_head)
    tc2  = cfg2.training
    stage2_is_sbi = _is_sbi_head(cfg2.head.type)

    if stage2_is_sbi:
        from src.sbi_training import build_sbi_stage2_model, load_stage1_state_dict, sbi_stage2_optimizer
        print("\nBuilding stage-2 model (frozen compressor + SBI/NPE) …")
        stage1_state = load_stage1_state_dict(stage1_ckpt)
        model2 = build_sbi_stage2_model(
            cfg2, device, train_ds.n_filters, stage1_state=stage1_state,
        )
        n2 = sum(p.numel() for p in model2.trainable_parameters())
        print(f"  Head: sbi_npe  |  Trainable params: {n2:,}  |  context: {cfg2.head.sbi_npe.context_mode}")
        print(f"  {format_stage_training_summary(tc2)}")
        opt2 = sbi_stage2_optimizer(model2, cfg2)
        sched2 = CosineAnnealingLR(opt2, T_max=max(tc2.epochs - tc2.warmup_epochs, 1))
        best_ckpt = CKPT_SBI_NPE
    else:
        print("\nBuilding stage-2 model (encoder + posterior head) …")
        model2 = build_model(
            cfg2, device=device, head_type=cfg2.head.type,
            n_total_filters=train_ds.n_filters,
        )
        load_encoder_weights(model2, stage1_ckpt)
        if tc.stage2.freeze_encoder:
            freeze_module(model2.encoder)
            if model2.bottleneck is not None:
                freeze_module(model2.bottleneck)
            print(f"  Encoder: frozen (loaded from {stage1_ckpt.name})"
                  + (" + bottleneck" if model2.bottleneck is not None else ""))
        else:
            print(f"  Encoder: loaded from {stage1_ckpt.name} (trainable)")
        n2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
        repr_dim = representation_dim(cfg2, model2.encoder.output_dim)
        head_in = repr_dim + 1 if cfg2.model.use_coverage else repr_dim
        cov = "on (latent+1)" if cfg2.model.use_coverage else "off"
        bn = f"  bottleneck: {repr_dim}d" if model2.bottleneck is not None else ""
        print(f"  Head: {cfg2.head.type}  |  Trainable params: {n2:,}  |  coverage: {cov}  |  head in: {head_in}{bn}")
        print(f"  {format_stage_training_summary(tc2)}")
        opt2, sched2 = _make_optimiser_and_scheduler(model2, tc2)
        best_ckpt = CKPT_POSTERIOR

    train_loader, val_loader, val_loader_drop, test_loader = _build_dataloaders(
        tc2, device, train_sub, val_sub_full, val_sub_drop, test_ds,
    )
    print(f"\n{'='*65}")
    print(f"  Stage 2 — posterior  ({cfg.run_name})  for {tc2.epochs} epochs")
    print(f"{'='*65}\n")

    try:
        _run_training_loop(
            model2, train_loader, val_loader, val_loader_drop, test_loader,
            opt2, sched2, device, cfg2, tc2, log_target, out_dir,
            train_ds, test_ds, train_dropout,
            best_ckpt_name=best_ckpt,
            history_name="history.json",
            test_metrics_name="test_metrics_posterior.json",
            subset_metrics_name="subset_metrics_posterior.json",
            predictions_name="predictions_posterior.npz",
            plots_subdir="stage2",
            stage_label="Stage 2 — Posterior (SBI/NPE)" if stage2_is_sbi else "Stage 2 — Posterior",
            run_post_hoc=_is_prob_head(cfg2.head.type) and (
                not stage2_is_sbi or cfg2.head.sbi_npe.calibration.grid_temperature_scaling
            ),
            post_hoc_role=ROLE_POSTERIOR,
            post_hoc_final_plots=True,
            stage_title="[stage 2] ",
        )
    finally:
        shutdown_dataloaders(train_loader, val_loader, val_loader_drop, test_loader)


def _generate_post_hoc_final_plots(
    model,
    test_loader,
    device,
    out_dir: Path,
    run_name: str,
    prob_outputs_raw: dict,
    sigma_scale: float,
    log_target: bool,
) -> None:
    """Write plots/final/ with post-hoc-calibrated posterior diagnostics."""
    scaled = collect_prob_outputs(
        model, test_loader, device,
        log_target=log_target, sigma_scale=sigma_scale,
    )
    if scaled is None:
        return

    plots_dir = out_dir / "plots" / "final"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating post-hoc final plots → {plots_dir}")

    z_true   = prob_outputs_raw["z_true"]
    z_mean   = prob_outputs_raw["z_mean"]
    head_type = prob_outputs_raw["head_type"]
    label    = "Stage 2 + post-hoc σ"
    title    = f"DeepSetZ — {run_name} — {label}"

    plot_scatter(
        z_true, z_mean, plots_dir,
        title=f"{title} (mean)",
        filename="scatter_mean.png",
    )
    plot_delta_z(
        z_true, z_mean, plots_dir,
        title=f"Δz (mean) — {run_name} — {label}",
        filename="delta_z_mean.png",
    )
    plot_calibration(
        scaled["pit"], plots_dir,
        title=f"Calibration — {run_name} — {label} ({head_type})",
        filename="calibration.png",
    )
    plot_calibration_comparison(
        prob_outputs_raw["pit"], scaled["pit"], plots_dir,
        title=f"Calibration — {run_name} — {label} ({head_type})",
        filename="calibration_comparison.png",
        sigma_scale=sigma_scale,
    )
    print("  Done.\n")


def _run_nsf_context_diagnostic(model: DeepSetZ, loader: DataLoader, device: torch.device) -> None:
    """Verify NSF log_prob changes when conditioning context changes."""
    from src.models.heads.nsf import NeuralSplineFlow, _rqs_forward

    if not isinstance(model.head, NeuralSplineFlow):
        return
    head = model.head
    model.eval()
    contexts, z_list = [], []
    with torch.no_grad():
        for tokens, mask, z_true in loader:
            tokens = tokens.to(device)
            mask = mask.to(device)
            h_set = model.encoder(tokens, mask)
            if model.bottleneck is not None:
                h_set = model.bottleneck(h_set)
            if model.density_context_branch or model.use_coverage_summary:
                ctx = model._nsf_density_context(h_set, tokens, mask)
            elif model.use_coverage:
                ctx = torch.cat([h_set, model._coverage_scalar(mask)], dim=-1)
            else:
                ctx = h_set
            contexts.append(ctx)
            z_list.append(z_true.to(device))
            if sum(c.size(0) for c in contexts) >= 64:
                break
    h = torch.cat(contexts, dim=0)[:64]
    y = torch.cat(z_list, dim=0)[:64]
    if h.size(0) < 32:
        print("  [nsf-diag] Skipped — fewer than 32 validation samples.")
        return
    w1, ht1, d1 = head._spline_params(h[:32])
    w2, ht2, d2 = head._spline_params(h[32:64])
    same_y = y[:32]
    _, lp1 = _rqs_forward(same_y, w1, ht1, d1, head.z_min, head.z_max)
    _, lp2 = _rqs_forward(same_y, w2, ht2, d2, head.z_min, head.z_max)
    diff = (lp1 - lp2).abs().mean().item()
    print(f"\n  [nsf-diag] context sensitivity |log p diff|: {diff:.6f}")
    if diff < 1e-5:
        print("  [nsf-diag] WARNING: NSF appears insensitive to context.")


def _load_best_checkpoint_for_eval(
    model: DeepSetZ,
    out_dir: Path,
    device: torch.device,
    best_ckpt_name: str,
    save_best: bool,
) -> Path | None:
    """Reload the best validation checkpoint before final test evaluation."""
    if not save_best:
        print("\nEvaluating final in-memory model (save_best=False).")
        return None

    candidate_paths = [
        out_dir / best_ckpt_name,
        out_dir / CKPT_SBI_NPE,
        out_dir / CKPT_POSTERIOR,
        out_dir / CKPT_END_TO_END,
    ]
    seen: set[Path] = set()
    best_path = None
    for p in candidate_paths:
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            best_path = p
            break

    if best_path is not None:
        try:
            state = torch.load(best_path, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        print(f"\nLoaded best checkpoint for final evaluation: {best_path.name}")
        return best_path

    print(
        "\nWARNING: save_best=True but no best checkpoint found; "
        "evaluating final in-memory model."
    )
    return None


def _run_training_loop(
    model, train_loader, val_loader, val_loader_drop, test_loader,
    optimiser, scheduler, device, cfg, tc, log_target, out_dir,
    train_ds, test_ds, train_dropout: DropoutConfig,
    *,
    best_ckpt_name: str = CKPT_END_TO_END,
    history_name: str = "history.json",
    test_metrics_name: str = "test_metrics.json",
    subset_metrics_name: str = "subset_metrics.json",
    predictions_name: str = "predictions.npz",
    plots_subdir: str | None = None,
    stage_label: str = "",
    run_post_hoc: bool = False,
    post_hoc_role: str = ROLE_END_TO_END,
    post_hoc_final_plots: bool = False,
    stage_title: str = "",
) -> None:
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    early_stop_min_epoch = (
        tc.early_stop_min_epoch if tc.early_stop_min_epoch > 0 else tc.full_filter_epochs
    )
    if tc.full_filter_epochs > 0:
        print(
            f"  Curriculum: full filters for epochs 1–{tc.full_filter_epochs}, "
            f"then normal dropout (LR ×{tc.dropout_resume_lr_mult} at transition)"
        )
        if early_stop_min_epoch > 0:
            print(f"  Early stopping active after epoch {early_stop_min_epoch}")

    for epoch in range(1, tc.epochs + 1):
        t0 = time.time()

        # Filter-dropout curriculum: learn full-map representation first
        if tc.full_filter_epochs > 0 and epoch <= tc.full_filter_epochs:
            train_ds.set_dropout_cfg(None)
        else:
            train_ds.set_dropout_cfg(train_dropout)

        # Warmup: linearly scale LR from 0 to target
        if epoch <= tc.warmup_epochs:
            lr = tc.lr * (epoch / max(tc.warmup_epochs, 1))
            if (
                tc.full_filter_epochs > 0
                and epoch == tc.full_filter_epochs + 1
                and tc.dropout_resume_lr_mult != 1.0
            ):
                lr *= tc.dropout_resume_lr_mult
                print(
                    f"  [filter-dropout] Dropout resumed — LR × {tc.dropout_resume_lr_mult:.2f} "
                    f"→ {lr:.2e}"
                )
            for pg in optimiser.param_groups:
                pg["lr"] = lr
        elif (
            tc.full_filter_epochs > 0
            and epoch == tc.full_filter_epochs + 1
            and tc.dropout_resume_lr_mult != 1.0
        ):
            for pg in optimiser.param_groups:
                pg["lr"] *= tc.dropout_resume_lr_mult
            print(
                f"  [filter-dropout] Dropout resumed — LR × {tc.dropout_resume_lr_mult:.2f} "
                f"→ {optimiser.param_groups[0]['lr']:.2e}"
            )

        train_loss = train_one_epoch(
            model, train_loader, optimiser, device, cfg, epoch, tc.epochs
        )

        if epoch > tc.warmup_epochs and scheduler is not None:
            scheduler.step()

        current_lr = optimiser.param_groups[0]["lr"]
        row: dict = {"epoch": epoch, "train_nll": train_loss, "train_loss": train_loss, "lr": current_lr}

        if epoch % tc.val_every == 0:
            # Pass 1: clean val (no dropout) — used for early stopping
            val_metrics = evaluate(model, val_loader, device, desc="validating",
                                   log_target=log_target)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            row["val_nll"] = val_metrics["nll"]

            # Pass 2: val with dropout — same conditions as training
            # Seed is fixed per epoch for a stable, comparable signal
            if val_loader_drop is not None:
                torch.manual_seed(tc.seed + epoch)
                drop_metrics = evaluate(model, val_loader_drop, device,
                                        desc="val(drop)", log_target=log_target)
                torch.manual_seed(tc.seed + epoch)  # reset for training reproducibility
                row.update({f"val_drop_{k}": v for k, v in drop_metrics.items()})
                row["val_drop_nll"] = drop_metrics["nll"]
                drop_str = f"  val_drop_nll={drop_metrics['nll']:.4f}"
            else:
                drop_str = ""

            improved = val_metrics["nll"] < best_val_loss
            in_min_epoch_phase = (
                early_stop_min_epoch > 0 and epoch <= early_stop_min_epoch
            )
            if improved:
                flag = " ✓ best"
            elif in_min_epoch_phase:
                flag = f" [early-stop disabled until epoch {early_stop_min_epoch}]"
            else:
                flag = f" (patience {patience_counter+1}/{tc.early_stop_patience})"

            print(
                f"{stage_title}Epoch {epoch:03d}/{tc.epochs}  "
                f"train_nll={train_loss:.4f}  "
                f"val_nll={val_metrics['nll']:.4f}"
                f"{drop_str}  "
                f"σ_NMAD={val_metrics['sigma_nmad']:.4f}  "
                f"bias={val_metrics['bias']:+.4f}  "
                f"outlier={val_metrics['outlier_rate']*100:.1f}%  "
                f"lr={current_lr:.2e}  "
                f"({time.time()-t0:.1f}s)"
                f"{flag}"
            )

            if improved:
                best_val_loss = val_metrics["nll"]
                patience_counter = 0
                if tc.save_best:
                    torch.save(model.state_dict(), out_dir / best_ckpt_name)
            elif epoch > early_stop_min_epoch:
                patience_counter += 1

            if (
                tc.early_stop_patience > 0
                and epoch > early_stop_min_epoch
                and patience_counter >= tc.early_stop_patience
            ):
                print(f"\nEarly stopping — no improvement for {tc.early_stop_patience} epochs.")
                break

        history.append(row)

    # Final checkpoint (last epoch weights — kept for debugging)
    torch.save(model.state_dict(), out_dir / "final_model.pt")

    # Reload best validation weights before test evaluation and plots
    _load_best_checkpoint_for_eval(
        model, out_dir, device, best_ckpt_name, tc.save_best,
    )

    if tc.run_nsf_context_diagnostic:
        _run_nsf_context_diagnostic(model, val_loader, device)

    # ── Test-set evaluation ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Test set evaluation (full filter set)")
    print(f"{'='*65}")
    test_metrics, z_pred_np, z_true_np = evaluate(
        model, test_loader, device, desc="test set",
        return_preds=True, log_target=log_target,
    )
    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*32}")
    for k, v in test_metrics.items():
        print(f"  {k:<20} {v:>10.4f}")
    print()

    # ── Probabilistic outputs (MDN / BinnedPDF only) ────────────────
    prob_outputs = collect_prob_outputs(
        model, test_loader, device, log_target=log_target,
    )

    # Save raw predictions
    if prob_outputs is not None:
        np.savez(
            out_dir / predictions_name,
            z_true   = z_true_np,
            z_pred   = z_pred_np,
            z_mean   = prob_outputs["z_mean"],
            z_median = prob_outputs["z_median"],
            z_mode   = prob_outputs["z_mode"],
            pit      = prob_outputs["pit"],
        )
        print(f"\n  Probabilistic estimates saved → {predictions_name}  [{prob_outputs['head_type']}]")
        for est in ("z_mean", "z_median", "z_mode"):
            m = compute_metrics(prob_outputs[est], z_true_np)
            print(f"    {est:<10}  σ_NMAD={m['sigma_nmad']:.4f}  "
                  f"bias={m['bias']:+.4f}  outlier={m['outlier_rate']*100:.1f}%  "
                  f"RMSE={m['rmse']:.4f}")
    else:
        np.savez(out_dir / predictions_name, z_true=z_true_np, z_pred=z_pred_np)

    # ── Survey-subset breakdown ─────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Survey-subset evaluation")
    print(f"{'='*65}")
    subset_results = evaluate_survey_subsets(
        model, test_ds, device,
        batch_size  = tc.batch_size * 2,
        num_workers = resolve_num_workers(tc.num_workers),
        log_target  = log_target,
    )
    print(f"\n  {'Subset':<35} {'n_filt':>6} {'σ_NMAD':>8} {'bias':>8} {'outlier%':>9}")
    print(f"  {'-'*70}")
    for label, m in sorted(subset_results.items(), key=lambda x: (x[1]["n_filters"], x[0])):
        print(f"  {label:<35} {m['n_filters']:>6d} "
              f"{m['sigma_nmad']:>8.4f} {m['bias']:>+8.4f} "
              f"{m['outlier_rate']*100:>8.1f}%")

    # ── Save artefacts ──────────────────────────────────────────────
    with open(out_dir / history_name, "w") as fh:
        json.dump(history, fh, indent=2)
    with open(out_dir / test_metrics_name, "w") as fh:
        json.dump(test_metrics, fh, indent=2)

    if run_post_hoc and cfg.training.post_hoc_calibration.enabled:
        print(f"\n{'='*65}")
        print("  Post-hoc σ calibration (val split)")
        print(f"{'='*65}")
        run_post_hoc_calibration(
            model, val_loader, device, out_dir, cfg,
            checkpoint_role=post_hoc_role,
        )
        if post_hoc_final_plots and prob_outputs is not None:
            sigma_scale = post_hoc_sigma_scale(out_dir)
            if sigma_scale is not None:
                prob_outputs["z_true"] = z_true_np
                _generate_post_hoc_final_plots(
                    model, test_loader, device, out_dir, cfg.run_name,
                    prob_outputs, sigma_scale, log_target,
                )
    with open(out_dir / subset_metrics_name, "w") as fh:
        json.dump(subset_results, fh, indent=2)

    # ── Plots ───────────────────────────────────────────────────────
    generate_all_plots(
        history        = history,
        z_true         = z_true_np,
        z_pred         = z_pred_np,
        subset_results = subset_results,
        out_dir        = out_dir,
        run_name       = cfg.run_name,
        prob_outputs   = prob_outputs,
        plots_subdir   = plots_subdir,
        stage_label    = stage_label,
    )

    print(f"Outputs saved to {out_dir}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepSetZ photometric redshift model")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--run_name", default=None, help="Override run name")
    parser.add_argument(
        "--stage1_checkpoint", default=None,
        help="Skip stage 1; load encoder from this checkpoint (best_point.pt or best_model.pt)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.stage1_checkpoint:
        cfg.training.stage1_checkpoint = args.stage1_checkpoint
        cfg.training.split_training = True

    train(cfg)


if __name__ == "__main__":
    main()
