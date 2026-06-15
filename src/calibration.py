"""
Post-hoc posterior width calibration for MDN and NSF heads.

Fits a single scale factor s applied to mixture / spline widths:
    σ → s · σ   (MDN)
    widths → s · widths   (NSF)

Point estimates (μ, π) are unchanged; only uncertainty narrows.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config, PostHocCalibrationConfig
from src.models.heads.mdn import MDN
from src.models.heads.nsf import NeuralSplineFlow
from src.run_artifacts import save_post_hoc


def compute_mace(pit: np.ndarray, n_alphas: int = 101) -> float:
    """Mean absolute calibration error on the equal-tails coverage curve."""
    pit = pit[np.isfinite(pit)]
    if len(pit) == 0:
        return float("nan")
    alphas = np.linspace(0.0, 1.0, n_alphas)
    coverage = np.array([
        np.mean((pit >= (1 - a) / 2) & (pit <= (1 + a) / 2))
        for a in alphas
    ])
    return float(np.mean(np.abs(coverage - alphas)))


def compute_ks_uniform(pit: np.ndarray) -> float:
    pit = np.sort(pit[np.isfinite(pit)])
    if len(pit) == 0:
        return float("nan")
    uniform = np.linspace(0, 1, len(pit))
    return float(np.max(np.abs(pit - uniform)))


def _collect_mdn_raw(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return z_true_raw, pi, mu, sigma on CPU."""
    model.eval()
    z_list, pi_list, mu_list, sig_list = [], [], [], []
    with torch.no_grad():
        for tokens, mask, z_true in loader:
            tokens = tokens.to(device)
            mask   = mask.to(device)
            out    = model(tokens, mask)
            z_list.append(z_true.cpu())
            pi_list.append(out["pi"].cpu())
            mu_list.append(out["mu"].cpu())
            sig_list.append(out["sigma"].cpu())
    return (
        torch.cat(z_list),
        torch.cat(pi_list),
        torch.cat(mu_list),
        torch.cat(sig_list),
    )


def _collect_nsf_raw(
    model,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    z_list, w_list, h_list, d_list = [], [], [], []
    with torch.no_grad():
        for tokens, mask, z_true in loader:
            tokens = tokens.to(device)
            mask   = mask.to(device)
            out    = model(tokens, mask)
            z_list.append(z_true.cpu())
            w_list.append(out["widths"].cpu())
            h_list.append(out["heights"].cpu())
            d_list.append(out["derivs"].cpu())
    return (
        torch.cat(z_list),
        torch.cat(w_list),
        torch.cat(h_list),
        torch.cat(d_list),
    )


def pit_mdn(
    head: MDN,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    z_true: torch.Tensor,
    scale: float,
) -> np.ndarray:
    sig = (sigma * scale).clamp(min=head.sigma_min)
    return head.pit_values(pi, mu, sig, z_true).numpy()


def pit_nsf(
    head: NeuralSplineFlow,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivs: torch.Tensor,
    z_true: torch.Tensor,
    scale: float,
) -> np.ndarray:
    w = (widths * scale).clamp(min=1e-6)
    return head.pit_values(w, heights, derivs, z_true).numpy()


def fit_sigma_scale(
    pit_fn,
    scale_min: float = 0.2,
    scale_max: float = 1.0,
    n_grid: int = 80,
) -> Tuple[float, float, float]:
    """
    Grid-search scale s that minimises MACE.

    Returns (best_scale, mace_before, mace_after).
    """
    pit_before = pit_fn(1.0)
    mace_before = compute_mace(pit_before)
    scales = np.linspace(scale_min, scale_max, n_grid)
    best_s, best_mace = 1.0, mace_before
    for s in scales:
        mace = compute_mace(pit_fn(float(s)))
        if mace < best_mace:
            best_mace = mace
            best_s = float(s)
    return best_s, mace_before, best_mace


def run_post_hoc_calibration(
    model,
    val_loader: DataLoader,
    device: torch.device,
    out_dir,
    cfg: Config,
    checkpoint_role: str = "posterior",
) -> Optional[Dict]:
    """
    Fit σ scale on the val loader and write calibration/post_hoc.json.

    Returns the saved payload dict, or None if skipped.
    """
    phc = cfg.training.post_hoc_calibration
    if not phc.enabled:
        return None

    head = model.head
    if not isinstance(head, (MDN, NeuralSplineFlow)):
        print("  [post-hoc] Skipped — head is not MDN or NSF.")
        return None

    head_type = head.__class__.__name__

    if isinstance(head, MDN):
        z_true, pi, mu, sigma = _collect_mdn_raw(model, val_loader, device)

        def _pit(s: float) -> np.ndarray:
            return pit_mdn(head, pi, mu, sigma, z_true, s)

    else:
        z_true, widths, heights, derivs = _collect_nsf_raw(model, val_loader, device)

        def _pit(s: float) -> np.ndarray:
            return pit_nsf(head, widths, heights, derivs, z_true, s)

    best_s, mace_before, mace_after = fit_sigma_scale(
        _pit,
        scale_min=phc.sigma_min,
        scale_max=phc.sigma_max,
        n_grid=phc.n_grid,
    )
    pit_b = _pit(1.0)
    pit_a = _pit(best_s)

    payload = {
        "method":           "sigma_scale",
        "head_type":        head_type,
        "checkpoint_role":  checkpoint_role,
        "sigma_scale":      best_s,
        "mace_before":      mace_before,
        "mace_after":       mace_after,
        "ks_before":        compute_ks_uniform(pit_b),
        "ks_after":         compute_ks_uniform(pit_a),
        "scale_bounds":     [phc.sigma_min, phc.sigma_max],
        "fitted_on":        "val",
    }
    path = save_post_hoc(out_dir, payload)
    print(
        f"  [post-hoc] σ scale = {best_s:.3f}  "
        f"MACE {mace_before:.4f} → {mace_after:.4f}  "
        f"KS {payload['ks_before']:.3f} → {payload['ks_after']:.3f}  "
        f"→ {path.relative_to(out_dir) if hasattr(path, 'relative_to') else path}"
    )
    return payload


def apply_sigma_scale_to_out(out: dict, head, scale: float) -> dict:
    """Return a shallow copy of model forward output with scaled widths."""
    if scale == 1.0 or scale is None:
        return out
    out = dict(out)
    if isinstance(head, MDN):
        out["sigma"] = (out["sigma"] * scale).clamp(min=head.sigma_min)
    elif isinstance(head, NeuralSplineFlow):
        out["widths"] = (out["widths"] * scale).clamp(min=1e-6)
    return out
