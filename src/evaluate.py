"""
Photometric redshift evaluation metrics.

Standard photoz metrics:

    Δz          = (z_phot - z_spec) / (1 + z_spec)   (normalised residual)
    bias        = median(Δz)
    σ_NMAD      = 1.4826 × median(|Δz - median(Δz)|)  (robust scatter)
    outlier rate = fraction of galaxies with |Δz| > η  (default η = 0.15)
    RMSE        = sqrt(mean(Δz²))

PZDC-specific metrics:
    outlier_rate_pzdc : uses adaptive threshold max(0.06, 3 × σ_IQR)
    cde_loss          : Conditional Density Estimation loss (Izbicki & Lee 2017)

References
----------
  Hildebrandt et al. (2010); Salvato et al. (2019); Izbicki & Lee (2017)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataloader_utils import resolve_num_workers

OUTLIER_THRESHOLD = 0.15


def compute_metrics(
    z_pred: np.ndarray,
    z_true: np.ndarray,
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> Dict[str, float]:
    """
    Compute standard photometric redshift metrics.

    Also computes the PZDC adaptive outlier rate:
        threshold = max(0.06, 3 × σ_IQR)  where σ_IQR = IQR / 1.349

    Parameters
    ----------
    z_pred : (N,) predicted redshifts
    z_true : (N,) spectroscopic redshifts

    Returns
    -------
    dict with keys: bias, sigma_nmad, outlier_rate, outlier_rate_pzdc, rmse, mae
    """
    delta_z = (z_pred - z_true) / (1.0 + z_true)

    bias         = float(np.median(delta_z))
    mad          = float(np.median(np.abs(delta_z - bias)))
    sigma_nmad   = 1.4826 * mad
    outlier_rate = float(np.mean(np.abs(delta_z) > outlier_threshold))
    rmse         = float(np.sqrt(np.mean(delta_z ** 2)))
    mae          = float(np.mean(np.abs(delta_z)))

    # PZDC adaptive threshold: max(0.06, 3 * sigma_IQR)
    q25, q75   = float(np.percentile(delta_z, 25)), float(np.percentile(delta_z, 75))
    sigma_iqr  = (q75 - q25) / 1.349
    pzdc_thr   = max(0.06, 3.0 * sigma_iqr)
    pzdc_out   = float(np.mean(np.abs(delta_z) > pzdc_thr))

    return {
        "bias":              bias,
        "sigma_nmad":        sigma_nmad,
        "outlier_rate":      outlier_rate,        # fixed η=0.15
        "outlier_rate_pzdc": pzdc_out,            # adaptive PZDC threshold
        "pzdc_threshold":    pzdc_thr,
        "rmse":              rmse,
        "mae":               mae,
    }


def compute_cde_loss(
    z_true: np.ndarray,
    probs:  Optional[np.ndarray] = None,  # (N, K) for BinnedPDF
    pi:     Optional[np.ndarray] = None,  # (N, C) for MDN
    mu:     Optional[np.ndarray] = None,  # (N, C) for MDN
    sigma:  Optional[np.ndarray] = None,  # (N, C) for MDN
    bin_centres: Optional[np.ndarray] = None,  # (K,) for BinnedPDF
    z_min: float = 0.0,
    z_max: float = 3.0,
    n_grid: int = 200,
) -> float:
    """
    Conditional Density Estimation (CDE) loss (Izbicki & Lee 2017).

        CDE ≈ (1/N) Σ_i [∫ p̂(z|xᵢ)² dz  -  2 p̂(zᵢ|xᵢ)]

    A lower value indicates better calibration.
    Supports BinnedPDF (probs + bin_centres) and MDN (pi + mu + sigma) heads.

    For BinnedPDF:
        ∫ p̂²dz  ≈  Σ_k (probs_k/Δz)² · Δz  =  Σ_k probs_k² / Δz

    For MDN:
        ∫ (Σ_c π_c N(μ_c,σ_c))² dz  = Σ_j Σ_k π_j π_k N(0|μ_j−μ_k, √(σ_j²+σ_k²))
        (product of two Gaussians identity, analytical)
    """
    if probs is not None and bin_centres is not None:
        # BinnedPDF path
        dz = (z_max - z_min) / probs.shape[1]
        integral_sq = np.sum(probs ** 2, axis=1) / dz   # (N,)

        # Evaluate p̂(z_true) for each galaxy via linear interpolation on the grid
        # Map z_true to bin index
        idx = np.clip(
            ((z_true - z_min) / (z_max - z_min) * len(bin_centres)).astype(int),
            0, len(bin_centres) - 1,
        )
        pdf_at_true = probs[np.arange(len(z_true)), idx] / dz   # (N,)

    elif pi is not None and mu is not None and sigma is not None:
        # MDN path — analytical integral using Gaussian product identity:
        # ∫ N(z|μ_j,σ_j) N(z|μ_k,σ_k) dz = N(0 | μ_j-μ_k, sqrt(σ_j²+σ_k²))
        _LOG_2PI_HALF = 0.5 * np.log(2 * np.pi)

        def _gauss(x, m, s):
            return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

        N, C = pi.shape
        integral_sq = np.zeros(N)
        for j in range(C):
            for k in range(C):
                diff_mu  = mu[:, j] - mu[:, k]
                sigma_jk = np.sqrt(sigma[:, j] ** 2 + sigma[:, k] ** 2)
                integral_sq += pi[:, j] * pi[:, k] * _gauss(diff_mu, 0, sigma_jk)

        # p̂(z_true_i) = Σ_c π_c N(z_true | μ_c, σ_c)
        z_col = z_true[:, None]    # (N, 1)
        pdf_at_true = np.sum(
            pi * _gauss(z_col, mu, sigma),
            axis=1,
        )
    else:
        raise ValueError("Provide either (probs, bin_centres) or (pi, mu, sigma).")

    cde = float(np.mean(integral_sq - 2.0 * pdf_at_true))
    return cde


def compute_pit_metrics(pit_values: np.ndarray) -> Dict[str, float]:
    """
    Compute PIT-based calibration metrics as used by the PZDC:
        - KS statistic vs Uniform[0,1]
        - PIT RMSE vs Uniform CDF
        - PIT KL divergence (histogram-based)
    """
    pit = np.sort(pit_values)
    N   = len(pit)
    uniform_cdf = np.linspace(0, 1, N)

    ks_stat = float(np.max(np.abs(pit - uniform_cdf)))

    pit_rmse = float(np.sqrt(np.mean((pit - uniform_cdf) ** 2)))

    # KL divergence: histogram-based
    n_bins = 20
    hist, _ = np.histogram(pit, bins=n_bins, range=(0, 1), density=True)
    hist    = np.clip(hist, 1e-8, None)
    ideal   = np.ones(n_bins)          # uniform density = 1.0
    kl_div  = float(np.sum(ideal * np.log(ideal / hist)) / n_bins)

    return {"pit_ks": ks_stat, "pit_rmse": pit_rmse, "pit_kl": kl_div}


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    print(f"{prefix}bias           = {metrics['bias']:+.4f}")
    print(f"{prefix}σ_NMAD         = {metrics['sigma_nmad']:.4f}")
    print(f"{prefix}outlier%       = {metrics['outlier_rate']*100:.2f}%  (|Δz|>0.15)")
    print(f"{prefix}outlier% [PZDC]= {metrics['outlier_rate_pzdc']*100:.2f}%  (|Δz|>{metrics.get('pzdc_threshold',0):.3f})")
    print(f"{prefix}RMSE(Δz)       = {metrics['rmse']:.4f}")
    print(f"{prefix}MAE(Δz)        = {metrics['mae']:.4f}")


# ------------------------------------------------------------------
# Standalone evaluation from a saved checkpoint
# ------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
) -> None:
    """Load a saved model and evaluate on the test set."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.config import load_config
    from src.dataset import GalaxyDataset, collate_fn
    from src.train import build_model, get_device

    cfg = load_config(config_path)
    root = Path(checkpoint_path).parent.parent  # assumes outputs/<run>/best_model.pt

    test_path = Path(__file__).parent.parent / cfg.data.test_path
    res_dir   = Path(__file__).parent.parent / cfg.data.res_dir

    test_ds = GalaxyDataset(
        parquet_path   = test_path,
        res_dir        = res_dir,
        target_col     = cfg.data.target_col,
        dropout_cfg    = None,
        active_surveys = cfg.data.active_surveys or None,
        log_target     = cfg.data.log_target,
        include_errors = cfg.data.include_errors,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.training.batch_size * 2,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = resolve_num_workers(cfg.training.num_workers),
    )

    device = get_device()
    model  = build_model(cfg, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    z_preds, z_trues = [], []
    with torch.no_grad():
        for tokens, mask, z_true in test_loader:
            out = model(tokens.to(device), mask.to(device))
            z_preds.append(out["z_pred"].cpu().numpy())
            z_trues.append(z_true.numpy())

    z_preds = np.concatenate(z_preds)
    z_trues = np.concatenate(z_trues)

    if cfg.data.log_target:
        z_preds = np.expm1(z_preds)
        z_trues = np.expm1(z_trues)

    metrics = compute_metrics(z_preds, z_trues)
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved DeepSetZ checkpoint")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument("config",     help="Path to YAML config used for training")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.config)
