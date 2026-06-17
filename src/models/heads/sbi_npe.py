"""
SBI/NPE-style conditional density head (optional dependency on ``sbi``).

Estimates q(t | context) with ``sbi.neural_nets.posterior_nn``, where ``t`` is an
unconstrained transform of redshift (see ``RedshiftTransform``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.target_transforms import RedshiftTransform

try:
    from sbi.neural_nets import posterior_nn as _posterior_nn
except ImportError:  # pragma: no cover - optional dependency
    _posterior_nn = None


def require_sbi() -> None:
    if _posterior_nn is None:
        raise ImportError(
            "SBI posterior head requested but `sbi` is not installed. "
            "Install with `pip install sbi` or `conda install -c conda-forge sbi`."
        )


class SBINPEHead(nn.Module):
    """
    Conditional density q(t | context) via ``sbi`` NPE-style flows.

    Training loss is NLL in t-space (``-log q(t|context)``).  Grid evaluation
    converts to z-space with the Jacobian for PIT / coverage / qp export.
    """

    head_type = "SBI_NPE"

    def __init__(
        self,
        context_dim: int,
        *,
        density_estimator: str = "nsf",
        hidden_features: int = 64,
        num_transforms: int = 3,
        num_bins: int = 8,
        target_transform_mode: str = "log1p_logit",
        y_min: float = 0.0,
        y_max: float = 1.45,
        eps: float = 1e-5,
        z_max_eval: Optional[float] = None,
        n_grid: int = 512,
    ) -> None:
        super().__init__()
        require_sbi()
        self.context_dim = context_dim
        self.density_estimator = density_estimator
        self.hidden_features = hidden_features
        self.num_transforms = num_transforms
        self.num_bins = num_bins
        self.n_grid = n_grid
        self.target_transform = RedshiftTransform(
            mode=target_transform_mode,
            y_min=y_min,
            y_max=y_max,
            eps=eps,
            z_max=z_max_eval,
        )
        self._flow: Optional[nn.Module] = None
        self._z_grid_cache: Optional[torch.Tensor] = None

    def _build_flow(self, device: torch.device) -> nn.Module:
        builder = _posterior_nn(
            model=self.density_estimator,
            hidden_features=self.hidden_features,
            num_transforms=self.num_transforms,
            num_bins=self.num_bins,
            z_score_theta="none",
            z_score_x="none",
        )
        dummy_t = torch.zeros(2, 1, device=device)
        dummy_x = torch.zeros(2, self.context_dim, device=device)
        return builder(dummy_t, dummy_x)

    @property
    def flow(self) -> nn.Module:
        if self._flow is None:
            raise RuntimeError("SBI flow not initialized — call ensure_flow() first.")
        return self._flow

    def ensure_flow(self, device: torch.device) -> None:
        if self._flow is None:
            self._flow = self._build_flow(device)

    def _log_prob_t(self, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """log q(t|context) for t (B,1), context (B,C). Returns (B,)."""
        lp = self.flow.log_prob(t, condition=context)
        if lp.dim() == 2 and lp.shape[0] == 1 and lp.shape[1] == t.shape[0]:
            lp = lp.squeeze(0)
        elif lp.dim() == 2 and lp.shape[1] == 1:
            lp = lp.squeeze(-1)
        return lp.reshape(-1)

    def log_prob(self, z: torch.Tensor, context: torch.Tensor, *, in_z_space: bool = False) -> torch.Tensor:
        """
        Log density at physical redshift ``z`` (B,).

        Default (``in_z_space=False``): NLL in t-space (no Jacobian) — used for training.
        With ``in_z_space=True``: includes log|dt/dz| for z-space density.
        """
        self.ensure_flow(context.device)
        t = self.target_transform.z_to_t(z)
        lp_t = self._log_prob_t(t.unsqueeze(-1), context)
        if in_z_space:
            lp_t = lp_t + self.target_transform.log_abs_det_dt_dz(z)
        return lp_t

    def loss(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(z, context, in_z_space=False).mean()

    @torch.no_grad()
    def sample(self, n_samples: int, context: torch.Tensor) -> torch.Tensor:
        """Sample z (n_samples, B) from q(z|context)."""
        self.ensure_flow(context.device)
        t = self.flow.sample((n_samples,), condition=context.unsqueeze(0))
        # shape (n_samples, 1, 1) for single context or (n_samples, B, 1)
        if t.dim() == 3 and t.shape[1] == 1:
            t = t.squeeze(1)
        z = self.target_transform.t_to_z(t.squeeze(-1))
        return z

    def z_grid(self, device: torch.device, z_min: float = 0.0, z_max: Optional[float] = None) -> torch.Tensor:
        z_max = z_max if z_max is not None else float(torch.expm1(torch.tensor(self.target_transform.y_max)).item())
        key = (device.type, z_min, z_max, self.n_grid)
        if self._z_grid_cache is None or getattr(self, "_grid_key", None) != key:
            self._z_grid_cache = torch.linspace(z_min, z_max, self.n_grid, device=device)
            self._grid_key = key
        return self._z_grid_cache

    def evaluate_grid(
        self,
        z_grid: torch.Tensor,
        context: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate normalized PDF on ``z_grid`` (G,) for each context row (B,C).

        Returns
        -------
        log_pdf_z : (B, G)
        pdf_z     : (B, G)  trapezoid-normalized over z
        """
        self.ensure_flow(context.device)
        B, G = context.shape[0], z_grid.numel()
        z_b = z_grid.unsqueeze(0).expand(B, G)
        t = self.target_transform.z_to_t(z_b.reshape(-1)).reshape(B, G)
        t_flat = t.reshape(B * G, 1)
        ctx_exp = context.unsqueeze(1).expand(B, G, -1).reshape(B * G, -1)
        log_pt = self._log_prob_t(t_flat, ctx_exp).reshape(B, G)
        log_j = self.target_transform.log_abs_det_dt_dz(z_b)
        log_pz = log_pt + log_j
        if temperature != 1.0:
            log_pz = log_pz / temperature
        pdf = log_pz.exp()
        dz = (z_grid[1] - z_grid[0]).item()
        norm = (pdf.sum(dim=-1) * dz).clamp(min=1e-12)
        pdf = pdf / norm.unsqueeze(-1)
        return log_pz, pdf

    @staticmethod
    def point_estimates_from_grid(
        z_grid: torch.Tensor,
        pdf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Posterior mean / median / mode from grid PDF (B,G)."""
        dz = (z_grid[1] - z_grid[0]).item()
        z_g = z_grid.unsqueeze(0)
        pdf_n = pdf / (pdf.sum(dim=-1, keepdim=True).clamp(min=1e-12))
        z_mean = (pdf_n * z_g).sum(dim=-1) * dz

        cdf = torch.cumulative_trapezoid(pdf_n, z_grid, dim=-1)
        # cumulative_trapezoid drops last point; pad to G
        cdf = torch.cat([cdf, torch.ones(pdf.shape[0], 1, device=pdf.device)], dim=-1)
        idx = (cdf >= 0.5).float().argmax(dim=-1)
        z_median = z_grid[idx]

        z_mode = z_grid[pdf.argmax(dim=-1)]
        return {"z_mean": z_mean, "z_median": z_median, "z_mode": z_mode}

    def pit_values(self, z_grid: torch.Tensor, pdf: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        dz = (z_grid[1] - z_grid[0]).item()
        pdf_n = pdf / (pdf.sum(dim=-1) * dz).clamp(min=1e-12).unsqueeze(-1)
        cdf = torch.cumulative_trapezoid(pdf_n, z_grid, dim=-1)
        cdf = torch.cat([torch.zeros(pdf.shape[0], 1, device=pdf.device), cdf], dim=-1)
        # linear interp
        G = z_grid.numel()
        idx_f = (z_true - z_grid[0]) / (z_grid[-1] - z_grid[0]) * (G - 1)
        idx_lo = idx_f.floor().long().clamp(0, G - 2)
        w = (idx_f - idx_lo.float()).clamp(0.0, 1.0)
        cdf_lo = cdf.gather(1, idx_lo.unsqueeze(1)).squeeze(1)
        cdf_hi = cdf.gather(1, (idx_lo + 1).unsqueeze(1)).squeeze(1)
        return (cdf_lo * (1 - w) + cdf_hi * w).clamp(0.0, 1.0)

    def forward(
        self,
        context: torch.Tensor,
        z_true: Optional[torch.Tensor] = None,
        *,
        z_pred_point: Optional[torch.Tensor] = None,
        fisher_lambda: float = 0.0,
        spread_lambda: float = 0.0,
        huber_lambda: float = 0.0,
        huber_delta: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        context : (B, C) conditional input (not raw encoder embedding).
        z_true  : (B,) physical redshift (caller converts from log target if needed).
        z_pred_point : (B,) deterministic point estimate (e.g. frozen MLP output).
        """
        self.ensure_flow(context.device)
        if z_pred_point is None:
            raise ValueError("SBINPEHead requires z_pred_point= for z_pred output.")

        result: Dict[str, torch.Tensor] = {"z_pred": z_pred_point}
        if z_true is not None:
            result["loss"] = self.loss(z_true, context)
        return result
