"""
Post-hoc posterior width calibration for MDN heads.

Fits a single scale factor s applied to mixture sigmas:
    σ → s · σ   (MDN)

Point estimates (μ, π) are unchanged; only uncertainty narrows.

NOTE: Sigma/width scaling is **not valid** for NSF spline widths (softmax-
normalised bin widths). NSF post-hoc calibration is disabled; use grid-
density temperature scaling instead (not yet implemented).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config, PostHocCalibrationConfig
from src.models.heads.mdn import MDN
from src.models.heads.nsf import NeuralSplineFlow, _rqs_forward
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
    if scale == 1.0:
        return head.pit_values(widths, heights, derivs, z_true).numpy()
    return pit_nsf_temperature(
        head, widths, heights, derivs, z_true, temperature=scale,
    )


def nsf_grid_log_pdf(
    head: NeuralSplineFlow,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivs: torch.Tensor,
    n_grid: int = 256,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Evaluate temperature-scaled log-PDF on a uniform z-grid.

    Returns (z_grid, pdf_normalised, dz).
    """
    B = widths.shape[0]
    device = widths.device
    z_grid = torch.linspace(head.z_min, head.z_max, n_grid, device=device)
    z_rep = z_grid.unsqueeze(0).expand(B, -1).reshape(-1)
    w_rep = widths.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
    h_rep = heights.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
    d_rep = derivs.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
    _, log_pdf = _rqs_forward(z_rep, w_rep, h_rep, d_rep, head.z_min, head.z_max)
    log_pdf = (log_pdf.reshape(B, n_grid) / max(temperature, 1e-6))
    pdf = torch.exp(log_pdf - log_pdf.max(dim=-1, keepdim=True).values)
    dz = (head.z_max - head.z_min) / max(n_grid - 1, 1)
    norm = (pdf.sum(dim=-1) * dz).clamp(min=1e-12)
    pdf_n = pdf / norm.unsqueeze(-1)
    return z_grid, pdf_n, dz


def pit_nsf_temperature(
    head: NeuralSplineFlow,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivs: torch.Tensor,
    z_true: torch.Tensor,
    temperature: float = 1.0,
    n_grid: int = 256,
) -> np.ndarray:
    """PIT from grid-evaluated, temperature-scaled NSF PDF (spline params unchanged)."""
    z_grid, pdf_n, dz = nsf_grid_log_pdf(
        head, widths, heights, derivs, n_grid=n_grid, temperature=temperature,
    )
    B = widths.shape[0]
    z_q = z_true.clamp(head.z_min, head.z_max)
    # CDF via cumulative trapezoidal sum on grid
    cdf_grid = torch.cumsum(pdf_n, dim=-1) * dz
    cdf_grid = cdf_grid.clamp(0.0, 1.0)
    # Linear index for z_query
    idx_f = (z_q - head.z_min) / (head.z_max - head.z_min) * (n_grid - 1)
    idx_lo = idx_f.floor().long().clamp(0, n_grid - 2)
    idx_hi = idx_lo + 1
    w = (idx_f - idx_lo.float()).clamp(0.0, 1.0)
    pit = (1 - w) * cdf_grid[torch.arange(B), idx_lo] + w * cdf_grid[torch.arange(B), idx_hi]
    return pit.numpy()


def fit_temperature(
    pit_fn,
    temp_min: float = 0.5,
    temp_max: float = 2.0,
    n_grid: int = 80,
) -> Tuple[float, float, float]:
    """Grid-search temperature T that minimises MACE."""
    pit_before = pit_fn(1.0)
    mace_before = compute_mace(pit_before)
    temps = np.linspace(temp_min, temp_max, n_grid)
    best_t, best_mace = 1.0, mace_before
    for t in temps:
        mace = compute_mace(pit_fn(float(t)))
        if mace < best_mace:
            best_mace = mace
            best_t = float(t)
    return best_t, mace_before, best_mace


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


def _run_sbi_grid_temperature_calibration(
    model,
    val_loader: DataLoader,
    device: torch.device,
    out_dir,
    cfg: Config,
    checkpoint_role: str = "posterior",
) -> Optional[Dict]:
    """Fit grid temperature T on SBI/NPE validation posteriors."""
    from src.models.sbi_stage2 import SBIStage2Model
    from src.sbi_training import collect_sbi_prob_outputs

    if not isinstance(model, SBIStage2Model):
        return None

    sc = cfg.head.sbi_npe
    if not sc.calibration.grid_temperature_scaling:
        print(
            "  [post-hoc] Skipped — sbi_npe.calibration.grid_temperature_scaling=false."
        )
        return None

    temps = [float(t) for t in sc.calibration.temperature_grid]
    log_target = cfg.data.log_target

    def _pit(temperature: float) -> np.ndarray:
        out = collect_sbi_prob_outputs(
            model, val_loader, device,
            log_target=log_target, temperature=temperature,
        )
        return out["pit"]

    pit_before = _pit(1.0)
    mace_before = compute_mace(pit_before)
    best_t, best_mace = 1.0, mace_before
    for t in temps:
        mace = compute_mace(_pit(t))
        if mace < best_mace:
            best_mace = mace
            best_t = t

    pit_a = _pit(best_t)
    payload = {
        "method":           "grid_temperature",
        "head_type":        "SBI_NPE",
        "checkpoint_role":  checkpoint_role,
        "temperature":      best_t,
        "sigma_scale":      best_t,
        "mace_before":      mace_before,
        "mace_after":       best_mace,
        "ks_before":        compute_ks_uniform(pit_before),
        "ks_after":         compute_ks_uniform(pit_a),
        "temperature_grid": temps,
        "fitted_on":        "val",
    }
    path = save_post_hoc(out_dir, payload)
    print(
        f"  [post-hoc] SBI grid T = {best_t:.3f}  "
        f"MACE {mace_before:.4f} → {best_mace:.4f}  "
        f"KS {payload['ks_before']:.3f} → {payload['ks_after']:.3f}  "
        f"→ {path}"
    )
    return payload


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

    from src.models.sbi_stage2 import SBIStage2Model
    if isinstance(model, SBIStage2Model):
        return _run_sbi_grid_temperature_calibration(
            model, val_loader, device, out_dir, cfg, checkpoint_role,
        )

    head = model.head
    if not isinstance(head, (MDN, NeuralSplineFlow)):
        print("  [post-hoc] Skipped — head is not MDN or NSF.")
        return None

    if isinstance(head, NeuralSplineFlow):
        nsf_cfg = cfg.head.nsf
        if nsf_cfg.disable_spline_width_posthoc_scaling and not nsf_cfg.use_grid_temperature_scaling:
            print(
                "  [post-hoc] Skipped — NSF spline-width scaling disabled. "
                "Enable head.nsf.use_grid_temperature_scaling for grid calibration."
            )
            return None

        if not nsf_cfg.use_grid_temperature_scaling:
            print(
                "  [post-hoc] Skipped — head.nsf.use_grid_temperature_scaling=false."
            )
            return None

        z_true, widths, heights, derivs = _collect_nsf_raw(model, val_loader, device)
        n_pdf = phc.n_grid_pdf

        def _pit(t: float) -> np.ndarray:
            return pit_nsf_temperature(
                head, widths, heights, derivs, z_true,
                temperature=t, n_grid=n_pdf,
            )

        best_t, mace_before, mace_after = fit_temperature(
            _pit,
            temp_min=phc.temperature_min,
            temp_max=phc.temperature_max,
            n_grid=phc.n_grid,
        )
        pit_b = _pit(1.0)
        pit_a = _pit(best_t)
        payload = {
            "method":           "grid_temperature",
            "head_type":        head.__class__.__name__,
            "checkpoint_role":  checkpoint_role,
            "temperature":      best_t,
            "sigma_scale":      best_t,  # alias for notebook loaders expecting sigma_scale
            "mace_before":      mace_before,
            "mace_after":       mace_after,
            "ks_before":        compute_ks_uniform(pit_b),
            "ks_after":         compute_ks_uniform(pit_a),
            "temperature_bounds": [phc.temperature_min, phc.temperature_max],
            "n_grid_pdf":       n_pdf,
            "fitted_on":        "val",
        }
        path = save_post_hoc(out_dir, payload)
        print(
            f"  [post-hoc] NSF grid T = {best_t:.3f}  "
            f"MACE {mace_before:.4f} → {mace_after:.4f}  "
            f"KS {payload['ks_before']:.3f} → {payload['ks_after']:.3f}  "
            f"→ {path}"
        )
        return payload

    head_type = head.__class__.__name__

    if isinstance(head, MDN):
        z_true, pi, mu, sigma = _collect_mdn_raw(model, val_loader, device)

        def _pit(s: float) -> np.ndarray:
            return pit_mdn(head, pi, mu, sigma, z_true, s)

    else:
        raise RuntimeError("Unreachable: NSF handled above.")

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
    """Return a shallow copy of model forward output with scaled MDN sigmas."""
    if scale == 1.0 or scale is None:
        return out
    out = dict(out)
    if isinstance(head, MDN):
        out["sigma"] = (out["sigma"] * scale).clamp(min=head.sigma_min)
    elif isinstance(head, NeuralSplineFlow):
        import warnings
        if scale != 1.0 and getattr(head, "_width_scale_warned", False) is False:
            warnings.warn(
                "NSF inference: use grid temperature scaling (pit_nsf_temperature), "
                "not spline-width mutation.",
                stacklevel=2,
            )
            head._width_scale_warned = True
    return out
