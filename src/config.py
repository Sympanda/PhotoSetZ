"""
Configuration dataclasses for DeepSetZ.

Configs are loaded from YAML files via `load_config`.  The YAML structure
mirrors the dataclass hierarchy, so fields not present in the file fall back
to their Python default values.

Example
-------
    cfg = load_config("configs/deepsets.yaml")
    print(cfg.model.type)
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Union


# ------------------------------------------------------------------
# Model configs
# ------------------------------------------------------------------

@dataclass
class DeepSetsConfig:
    phi_hidden:   List[int] = field(default_factory=lambda: [128, 128])
    latent_dim:   int  = 128
    rho_hidden:   List[int] = field(default_factory=lambda: [256, 128])
    embed_dim:    int  = 128
    pooling:      str  = "mean"  # mean | sum | max | attention
    dropout:      float = 0.1
    activation:   str  = "gelu"  # gelu | relu | silu | leaky_relu | tanh


@dataclass
class SetTransformerConfig:
    embed_dim:          int  = 128
    n_heads:            int  = 4
    n_attn_layers:      int  = 2
    ffn_dim:            Optional[int] = None  # defaults to 4 * embed_dim
    n_pma_seeds:        int  = 1
    dropout:            float = 0.1
    pre_embed_hidden:   List[int] = field(default_factory=lambda: [64])


@dataclass
class ModelConfig:
    type:           str  = "deepsets"   # deepsets | set_transformer
    token_dim:      int  = 4
    deepsets:       DeepSetsConfig       = field(default_factory=DeepSetsConfig)
    set_transformer: SetTransformerConfig = field(default_factory=SetTransformerConfig)
    # Append n_active/n_total_filters as a scalar to the pooled encoder embedding
    # before the head (latent dim + 1). Legacy coverage conditioning.
    use_coverage:   bool = True
    # Fixed-size wavelength/coverage summaries (see coverage_summary.py).
    use_coverage_summary: bool = False
    # Separate MLP context path for NSF density estimation.
    density_context_branch: bool = False
    density_context_hidden: List[int] = field(default_factory=lambda: [256, 128])
    density_context_dropout: float = 0.05
    # Encoder bottleneck: false = disabled; int = latent dim (e.g. 64).
    bottleneck: Union[bool, int] = False
    bottleneck_dropout: float = 0.1


# ------------------------------------------------------------------
# Head configs
# ------------------------------------------------------------------

@dataclass
class MLPRegressorConfig:
    hidden_dims:  List[int] = field(default_factory=lambda: [64, 32])
    dropout:      float = 0.1
    huber_delta:  float = 0.5
    activation:   str  = "gelu"  # gelu | relu | silu | leaky_relu | tanh


@dataclass
class BinnedPDFConfig:
    n_bins:      int   = 100
    z_min:       float = 0.0
    z_max:       float = 3.0
    hidden_dims: List[int] = field(default_factory=lambda: [64])
    dropout:     float = 0.1
    activation:  str   = "gelu"  # gelu | relu | silu | leaky_relu | tanh


@dataclass
class MDNConfig:
    n_components: int  = 5
    hidden_dims:  List[int] = field(default_factory=lambda: [64])
    dropout:      float = 0.1
    sigma_min:    float = 0.01
    activation:   str  = "gelu"  # gelu | relu | silu | leaky_relu | tanh


@dataclass
class NSFConfig:
    """Neural Spline Flow head.

    NOTE: if log_target=True, set z_max to log(1+z_true_max).
    E.g. for z ∈ [0, 3]: z_max ≈ 1.39.  For z ∈ [0, 2.5]: z_max ≈ 1.32.
    """
    n_bins:      int   = 32
    z_min:       float = 0.0
    z_max:       float = 3.0           # use ~1.39 when log_target=True
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout:     float = 0.1
    activation:  str   = "gelu"
    deriv_min:   float = 1e-3
    # NSF post-hoc calibration on evaluated grid PDFs (not spline widths).
    use_grid_temperature_scaling: bool = False
    disable_spline_width_posthoc_scaling: bool = True


@dataclass
class SbiNpeContextProjectionConfig:
    enabled:    bool = True
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout:    float = 0.0
    layer_norm: bool = True


@dataclass
class SbiNpeGridEvalConfig:
    n_grid:     int = 512
    z_min_eval: float = 0.0
    z_max_eval: Optional[float] = None


@dataclass
class SbiNpeCalibrationConfig:
    grid_temperature_scaling: bool = False
    temperature_grid: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]
    )


@dataclass
class SbiNpeEnsembleConfig:
    enabled:   bool = False
    n_members: int = 1
    seeds:     List[int] = field(default_factory=list)


@dataclass
class SbiNpeConfig:
    """SBI/NPE conditional density head (requires optional ``sbi`` package)."""
    density_estimator: str = "nsf"          # nsf | maf | mdn
    hidden_features:   int = 64
    num_transforms:    int = 3
    num_bins:          int = 8
    context_mode:      str = "frozen_point" # frozen_point | point_only | latent_only | ensemble_summary
    target_transform:  str = "log1p_logit"  # log1p_logit | log1p | z_logit | z
    y_min:             float = 0.0
    y_max:             float = 1.45
    eps:               float = 1e-5
    train_stage:       int = 2
    freeze_encoder:    bool = True
    include_point_prediction: bool = True
    include_coverage_summary: bool = True
    context_dim:       Optional[int] = None
    context_projection: SbiNpeContextProjectionConfig = field(
        default_factory=SbiNpeContextProjectionConfig
    )
    lr:                Optional[float] = None
    weight_decay:      Optional[float] = None
    batch_size:        Optional[int] = None
    max_epochs:        Optional[int] = None
    early_stopping_metric: str = "val_nll"
    ensemble:          SbiNpeEnsembleConfig = field(default_factory=SbiNpeEnsembleConfig)
    grid_eval:         SbiNpeGridEvalConfig = field(default_factory=SbiNpeGridEvalConfig)
    calibration:       SbiNpeCalibrationConfig = field(default_factory=SbiNpeCalibrationConfig)


@dataclass
class HeadConfig:
    type:           str  = "mlp_regressor"   # mlp_regressor | binned_pdf | mdn | nsf | sbi_npe
    mlp_regressor:  MLPRegressorConfig = field(default_factory=MLPRegressorConfig)
    binned_pdf:     BinnedPDFConfig    = field(default_factory=BinnedPDFConfig)
    mdn:            MDNConfig          = field(default_factory=MDNConfig)
    nsf:            NSFConfig          = field(default_factory=NSFConfig)
    sbi_npe:        SbiNpeConfig       = field(default_factory=SbiNpeConfig)


# ------------------------------------------------------------------
# Dropout config
# ------------------------------------------------------------------

@dataclass
class DropoutConfig:
    """
    Stratified filter dropout strategy.  Probabilities must sum to 1.0.

    Modes
    -----
    complete    : keep all filters
    preset      : use a named realistic survey subset (DECaLS, LSST-only, etc.)
    survey_drop : drop 1–max_surveys_to_drop entire surveys, then optionally thin
    aggressive  : per-filter rate drawn uniformly from [rate_low, rate_high]
    """
    p_complete:               float = 0.15
    p_preset:                 float = 0.25
    p_survey_drop:            float = 0.25
    p_aggressive:             float = 0.35

    aggressive_rate_low:      float = 0.3
    aggressive_rate_high:     float = 0.8

    max_surveys_to_drop:      int   = 2
    p_filter_after_survey:    float = 0.4
    filter_rate_after_survey: float = 0.25

    min_filters:              int   = 3


# ------------------------------------------------------------------
# Split-training stage overrides & post-hoc calibration
# ------------------------------------------------------------------

@dataclass
class StageTrainingConfig:
    """
    Per-stage overrides for split training.
    Fields left at None inherit from the top-level training / head / model config.
    """
    head:                   Optional[str]   = None
    epochs:                 Optional[int]   = None
    lr:                     Optional[float] = None
    weight_decay:           Optional[float] = None
    lr_scheduler:           Optional[str]   = None   # cosine | step | none
    warmup_epochs:          Optional[int]   = None
    clip_grad_norm:         Optional[float] = None
    batch_size:             Optional[int]   = None
    early_stop_patience:    Optional[int]   = None
    early_stop_min_epoch:   Optional[int]   = None
    fisher_lambda:          Optional[float] = None
    spread_lambda:          Optional[float] = None
    huber_lambda:           Optional[float] = None
    huber_delta:            Optional[float] = None
    val_dropout:            Optional[bool]  = None
    full_filter_epochs:     Optional[int]   = None
    dropout_resume_lr_mult: Optional[float] = None
    freeze_encoder:         bool            = False
    use_coverage:           Optional[bool]  = None   # override model.use_coverage for this stage


@dataclass
class PostHocCalibrationConfig:
    """Post-hoc calibration: MDN σ-scale or NSF grid temperature."""
    enabled:    bool  = True
    sigma_min:  float = 0.2
    sigma_max:  float = 1.0
    n_grid:     int   = 80
    # NSF grid temperature T: log_pdf_cal = log_pdf / T
    temperature_min: float = 0.5
    temperature_max: float = 2.0
    n_grid_pdf: int = 256


# ------------------------------------------------------------------
# Training config
# ------------------------------------------------------------------

@dataclass
class TrainingConfig:
    batch_size:          int   = 256
    lr:                  float = 1e-3
    weight_decay:        float = 1e-4
    epochs:              int   = 100
    dropout:             DropoutConfig = field(default_factory=DropoutConfig)
    lr_scheduler:        str   = "cosine"  # cosine | step | none
    warmup_epochs:       int   = 5
    clip_grad_norm:      float = 1.0
    num_workers:         int   = 4    # auto-set to 0 on macOS (see dataloader_utils.py)
    seed:                int   = 42
    log_every:           int   = 100
    val_every:           int   = 1
    save_best:           bool  = True
    early_stop_patience: int   = 15    # 0 = disabled

    # ── Posterior regularisation ──────────────────────────────────────
    # Applied to probabilistic heads (MDN, BinnedPDF, NSF) only.
    # Both default to 0 so all existing runs are unaffected.
    #
    # fisher_lambda  > 0 : penalises sharp/over-confident posteriors
    #                       (standard Fisher information penalty)
    #                       Use when PIT histogram has a U-shape.
    #
    # spread_lambda  > 0 : penalises broad/over-dispersed posteriors
    #                       (weighted variance penalty: E[Σ_k π_k σ_k²])
    #                       Use when PIT histogram has an n-shape (our case).
    fisher_lambda:       float = 0.0
    spread_lambda:       float = 0.0

    # ── Auxiliary point loss (probabilistic heads only) ───────────────
    # loss += huber_lambda * Huber(z_point, z_true)
    # Complements NLL / cross-entropy for σ_NMAD without replacing the
    # probabilistic objective.  Tune spread_lambda separately for PIT calibration.
    huber_lambda:        float = 0.0
    huber_delta:         float = 0.5

    # ── Filter-dropout curriculum ─────────────────────────────────────
    # First full_filter_epochs: train on complete filter sets only (no dropout).
    # At epoch full_filter_epochs + 1, normal dropout resumes and LR is scaled
    # by dropout_resume_lr_mult to ease the transition (1.0 = no change).
    full_filter_epochs:      int   = 0    # 0 = disabled
    dropout_resume_lr_mult:  float = 1.0

    # Early stopping patience only counts after this epoch.
    # Defaults to full_filter_epochs when left at 0.
    early_stop_min_epoch:    int   = 0

    # ── Validation with dropout ───────────────────────────────────────
    # When True, an additional val pass is run each epoch with the same
    # filter dropout as training.  The dropout seed is fixed per epoch
    # (seed = training.seed + epoch) for a stable, comparable signal.
    # Early stopping uses the clean (no-dropout) val loss; both losses
    # are logged and plotted in training_curves.png.
    val_dropout:         bool  = False

    # ── Split training (encoder → point head, then frozen encoder → PDF head) ──
    # When True, stage 1 trains encoder + MLP; stage 2 loads encoder and trains
    # MDN / NSF / BinnedPDF only.  Saves best_point.pt and best_posterior.pt.
    # When False, standard end-to-end training → best_model.pt.
    split_training:       bool  = False
    # Skip stage 1 and load encoder weights from an existing checkpoint
    # (best_point.pt from a split run, or best_model.pt — encoder.* only).
    stage1_checkpoint:    Optional[str] = None
    stage1:               StageTrainingConfig = field(
        default_factory=lambda: StageTrainingConfig(head="mlp_regressor")
    )
    stage2:               StageTrainingConfig = field(
        default_factory=lambda: StageTrainingConfig(freeze_encoder=True, huber_lambda=0.0)
    )
    # Optional NSF context-sensitivity diagnostic after training.
    run_nsf_context_diagnostic: bool = False
    post_hoc_calibration: PostHocCalibrationConfig = field(
        default_factory=PostHocCalibrationConfig
    )


# ------------------------------------------------------------------
# Data config
# ------------------------------------------------------------------

@dataclass
class DataConfig:
    train_path:     str       = "data/ellen/train_175k.parquet"
    test_path:      str       = "data/ellen/test_25k.parquet"
    res_dir:        str       = "data/ellen"
    target_col:     str       = "true_redshift"
    # Restrict to a specific set of surveys or filters.
    # Use survey names ("lsst", "roman", "euclid", "wise") to include whole
    # surveys, or individual column names (e.g. "mag_g_lsst") for fine control.
    # Empty list = use all available filters.
    active_surveys: List[str] = field(default_factory=list)
    # Train in log(1+z) space to reduce tail bias.
    # Predictions and all metrics/plots are always reported in real z.
    log_target:     bool      = False
    # Include per-filter magnitude errors as a 5th token feature.
    include_errors: bool = False
    # SED-like non-detection encoding (default off = legacy drop-NaN behaviour).
    encode_nondetections: bool = False
    nondetection_policy: str = "drop"          # drop | keep_token
    nondetection_mag_fill: float = 30.0
    nondetection_err_fill: float = 1.0
    add_detection_flags: bool = False
    # false: zero-fill missing error cols + warn once (legacy default)
    # true:  raise ValueError if expected error columns are absent
    strict_error_columns: bool = False
    # ── Deprecated aliases (still honoured via resolve_data_options) ──
    allow_missing_error_cols: bool = False
    preserve_nondetections: bool = False


# ------------------------------------------------------------------
# Benchmark baseline config (flat MLP / MDN)
# ------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    enabled:            bool      = False
    model_type:         str       = "flat_mlp"   # flat_mlp | flat_mdn
    subset_name:        str       = ""
    filter_columns:     List[str] = field(default_factory=list)
    hidden_dims:        List[int] = field(default_factory=lambda: [64, 128, 256, 128, 64, 32])
    include_mag_errors: bool      = False
    n_mdn_components:   int       = 5


# ------------------------------------------------------------------
# Top-level config
# ------------------------------------------------------------------

@dataclass
class Config:
    run_name:  str      = "run"
    output_dir: str     = "outputs"
    config_schema_version: str = "1.0"
    data:      DataConfig     = field(default_factory=DataConfig)
    model:     ModelConfig    = field(default_factory=ModelConfig)
    head:      HeadConfig     = field(default_factory=HeadConfig)
    training:  TrainingConfig = field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


# ------------------------------------------------------------------
# YAML loading helpers
# ------------------------------------------------------------------

def _merge(base, update: dict):
    """Recursively merge a dict into a dataclass, returning the dataclass."""
    if update is None:
        return base
    for key, value in update.items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown config key: '{key}'")
        current = getattr(base, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            setattr(base, key, _merge(current, value))
        else:
            setattr(base, key, value)
    return base


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and merge it over the defaults."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config()
    _merge(cfg, data)
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Serialise the config to YAML."""
    with open(path, "w") as fh:
        yaml.dump(asdict(cfg), fh, default_flow_style=False)
