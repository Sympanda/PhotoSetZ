"""
Neural Spline Flow (NSF) head — continuous, expressive redshift posterior.

    p(z | X_g)  is parameterised by a monotone rational-quadratic spline CDF
    whose K-bin knot parameters (widths, heights, derivatives) are output by
    a small MLP conditioned on the encoder embedding.

Advantages over MDN
-------------------
* Fully flexible: can represent arbitrary unimodal or multimodal distributions
  without pre-specifying the number of modes.
* Exact, analytic CDF and its inverse — no numerical integration for PIT,
  quantiles, or sampling.
* Typically sharper posteriors and better calibration than MDN at similar
  parameter counts.

Advantages over BinnedPDF
--------------------------
* Continuous (not discretised), so gradients w.r.t. z are well-defined.
* Much more compact output: O(3K) spline params vs O(K) discrete probabilities
  for the same resolution.

Reference
---------
  Durkan et al. (2019) "Neural Spline Flows", NeurIPS.
  https://arxiv.org/abs/1906.04032
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimum derivative at spline knots (prevents degenerate zero-gradient regions)
_MIN_DERIV = 1e-3


def _rqs_forward(
    z:       torch.Tensor,   # (B,)  query redshifts, must be in [z_min, z_max]
    widths:  torch.Tensor,   # (B, K)  normalised bin widths  (sum=1, >0)
    heights: torch.Tensor,   # (B, K)  normalised bin heights (sum=1, >0)
    derivs:  torch.Tensor,   # (B, K+1) derivatives at knot points (>0)
    z_min:   float,
    z_max:   float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rational-quadratic spline forward pass (Durkan et al. 2019, Eq. 3–4).

    Maps z → (CDF(z), log_pdf(z)) in a numerically stable way.

    Returns
    -------
    cdf     : (B,) values in [0, 1]
    log_pdf : (B,) log p(z)
    """
    B, K = widths.shape
    z_range = z_max - z_min

    # Cumulative knot positions
    cum_widths  = F.pad(widths.cumsum(dim=-1),  (1, 0), value=0.0)  # (B, K+1)
    cum_heights = F.pad(heights.cumsum(dim=-1), (1, 0), value=0.0)  # (B, K+1)

    # Scale widths/heights to the actual z range / probability space
    cum_widths  = z_min  + z_range * cum_widths   # knot x-positions in [z_min, z_max]
    bin_widths  = z_range * widths                 # (B, K)
    bin_heights = heights                          # (B, K)  (CDF increments, sum=1)

    # Find which bin each z falls in  (B,)
    z_col = z.unsqueeze(-1).expand(B, K + 1)       # (B, K+1)
    # bin index: number of right-knot positions that are ≤ z
    k_idx = ((cum_widths <= z_col).sum(dim=-1) - 1).clamp(0, K - 1)  # (B,)

    # Gather per-bin parameters for each sample  (B,)
    def _gather(t):
        return t.gather(1, k_idx.unsqueeze(1)).squeeze(1)

    x_k   = _gather(cum_widths[:, :-1])   # left knot x
    y_k   = _gather(cum_heights[:, :-1])  # left knot CDF value
    w_k   = _gather(bin_widths)           # bin width
    h_k   = _gather(bin_heights)          # bin CDF increment
    d_k   = _gather(derivs[:, :-1])       # derivative at left knot
    d_k1  = _gather(derivs[:, 1:])        # derivative at right knot
    s_k   = h_k / w_k                    # secant slope

    # Normalised position within the bin  ξ ∈ [0, 1]
    xi = ((z - x_k) / w_k).clamp(0.0, 1.0)

    # RQ-spline CDF increment within the bin  (Durkan Eq. 3)
    denom = s_k + (d_k1 + d_k - 2.0 * s_k) * xi * (1.0 - xi)
    numer = h_k * (s_k * xi**2 + d_k * xi * (1.0 - xi))
    cdf = y_k + numer / denom.clamp(min=1e-8)

    # PDF  (Durkan Eq. 4)
    log_deriv_numer = (
        s_k**2 * (d_k1 * xi**2 + 2.0 * s_k * xi * (1.0 - xi) + d_k * (1.0 - xi)**2)
    )
    log_pdf = torch.log(log_deriv_numer.clamp(min=1e-8)) - 2.0 * torch.log(denom.clamp(min=1e-8))
    # Scale by 1/w_k (change of variables from ξ to z) and by 1/z_range for heights
    log_pdf = log_pdf - torch.log(w_k.clamp(min=1e-8))

    return cdf.clamp(0.0, 1.0), log_pdf


def _rqs_inverse(
    u:       torch.Tensor,   # (B,)  CDF target values in [0, 1]
    widths:  torch.Tensor,   # (B, K)
    heights: torch.Tensor,   # (B, K)
    derivs:  torch.Tensor,   # (B, K+1)
    z_min:   float,
    z_max:   float,
) -> torch.Tensor:
    """
    Inverse of the RQ-spline CDF: CDF⁻¹(u) via analytical quadratic formula
    (Durkan et al. 2019, Appendix B).

    Returns
    -------
    z : (B,) redshifts in [z_min, z_max]
    """
    B, K = widths.shape
    z_range = z_max - z_min

    cum_widths  = F.pad(widths.cumsum(-1),  (1, 0), value=0.0)
    cum_heights = F.pad(heights.cumsum(-1), (1, 0), value=0.0)
    cum_widths  = z_min + z_range * cum_widths

    # Find bin: number of cum_height knots ≤ u
    u_col = u.unsqueeze(-1).expand(B, K + 1)
    k_idx = ((cum_heights <= u_col).sum(-1) - 1).clamp(0, K - 1)

    def _gather(t):
        return t.gather(1, k_idx.unsqueeze(1)).squeeze(1)

    x_k  = _gather(cum_widths[:, :-1])
    y_k  = _gather(cum_heights[:, :-1])
    w_k  = z_range * _gather(widths)
    h_k  = _gather(heights)
    d_k  = _gather(derivs[:, :-1])
    d_k1 = _gather(derivs[:, 1:])
    s_k  = h_k / w_k.clamp(min=1e-8)

    eta = (u - y_k) / h_k.clamp(min=1e-8)    # CDF fraction within bin
    eta = eta.clamp(0.0, 1.0)

    # Quadratic formula for ξ  (Durkan App. B)
    a = h_k * (s_k - d_k) + (u - y_k) * (d_k1 + d_k - 2.0 * s_k)
    b = h_k * d_k          - (u - y_k) * (d_k1 + d_k - 2.0 * s_k)
    c = -s_k * (u - y_k)

    discriminant = (b**2 - 4.0 * a * c).clamp(min=0.0)
    xi = (2.0 * c) / (-b - discriminant.sqrt()).clamp(max=-1e-8)
    xi = xi.clamp(0.0, 1.0)

    return (x_k + xi * w_k).clamp(z_min, z_max)


class NeuralSplineFlow(nn.Module):
    """
    Neural Spline Flow (NSF) prediction head.

    Parameters
    ----------
    embed_dim   : encoder output dimension
    n_bins      : number of spline bins (more bins → more expressive, K≥4)
    z_min, z_max: redshift range of the spline
    hidden_dims : hidden layers in the parameter-prediction MLP
    dropout     : dropout probability in the MLP
    activation  : non-linearity (gelu | relu | silu | leaky_relu | tanh)
    deriv_min   : minimum knot derivative (numerical stability)

    Notes
    -----
    If log_target=True in your config, set z_min and z_max to the corresponding
    log(1+z) range.  For z ∈ [0, 3]: log(1+3) ≈ 1.386, so use z_max ≈ 1.4.
    For z ∈ [0, 2.5]: log(1+2.5) ≈ 1.317, use z_max ≈ 1.35.
    """

    _ACTIVATIONS = {
        "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh,
    }

    def __init__(
        self,
        embed_dim:   int,
        n_bins:      int   = 32,
        z_min:       float = 0.0,
        z_max:       float = 3.0,
        hidden_dims: List[int] = (128, 64),
        dropout:     float = 0.1,
        activation:  str   = "gelu",
        deriv_min:   float = _MIN_DERIV,
    ) -> None:
        super().__init__()
        self.n_bins    = n_bins
        self.z_min     = z_min
        self.z_max     = z_max
        self.deriv_min = deriv_min

        act_cls = self._ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'.")

        # Shared trunk MLP
        layers: List[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_cls(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Three separate output heads
        self.w_head = nn.Linear(in_dim, n_bins)       # → bin widths (softmax)
        self.h_head = nn.Linear(in_dim, n_bins)       # → bin heights / prob mass (softmax)
        self.d_head = nn.Linear(in_dim, n_bins + 1)   # → knot derivatives (softplus)

    def _spline_params(self, embedding: torch.Tensor):
        """Compute normalised spline parameters from the encoder embedding."""
        h       = self.trunk(embedding)
        widths  = torch.softmax(self.w_head(h), dim=-1)                  # (B, K)
        heights = torch.softmax(self.h_head(h), dim=-1)                  # (B, K)
        derivs  = F.softplus(self.d_head(h)) + self.deriv_min            # (B, K+1)
        return widths, heights, derivs

    # ------------------------------------------------------------------
    # Probabilistic query methods (same interface as MDN / BinnedPDF)
    # ------------------------------------------------------------------

    def cdf_at(
        self,
        widths:  torch.Tensor,
        heights: torch.Tensor,
        derivs:  torch.Tensor,
        z_query: torch.Tensor,   # (B,) — one value per galaxy
    ) -> torch.Tensor:
        """Evaluate the spline CDF at z_query[i] for each galaxy i."""
        cdf, _ = _rqs_forward(z_query, widths, heights, derivs, self.z_min, self.z_max)
        return cdf

    def pit_values(
        self,
        widths:     torch.Tensor,
        heights:    torch.Tensor,
        derivs:     torch.Tensor,
        z_true_raw: torch.Tensor,   # (B,) in prediction space
    ) -> torch.Tensor:
        """PIT = CDF(z_true) — Uniform[0,1] for a calibrated model."""
        return self.cdf_at(widths, heights, derivs, z_true_raw)

    def point_estimates_from_params(
        self,
        widths:  torch.Tensor,
        heights: torch.Tensor,
        derivs:  torch.Tensor,
        n_grid:  int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean, median, and mode from spline parameters.

        median  : exact analytic inverse CDF⁻¹(0.5)
        mode    : argmax of log-PDF evaluated on a dense grid
        mean    : ∫ z · p(z) dz via numerical integration on the same grid
        """
        # Median — analytical
        u_half  = torch.full((widths.shape[0],), 0.5, device=widths.device)
        z_median = _rqs_inverse(u_half, widths, heights, derivs, self.z_min, self.z_max)

        # Grid for mode and mean
        device = widths.device
        z_grid = torch.linspace(self.z_min, self.z_max, n_grid, device=device)   # (G,)
        B      = widths.shape[0]

        z_g_flat = z_grid.unsqueeze(0).expand(B, -1).reshape(-1)   # (B*G,)
        w_rep = widths.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
        h_rep = heights.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
        d_rep = derivs.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)

        _, log_pdf_flat = _rqs_forward(z_g_flat, w_rep, h_rep, d_rep, self.z_min, self.z_max)
        log_pdf = log_pdf_flat.reshape(B, n_grid)   # (B, G)

        # Mode
        z_mode = z_grid[log_pdf.argmax(dim=-1)]     # (B,)

        # Mean — numerical trapezoidal integration
        pdf    = log_pdf.exp()
        dz     = (self.z_max - self.z_min) / (n_grid - 1)
        z_mean = (pdf * z_grid.unsqueeze(0)).sum(-1) * dz   # (B,) — unnorm.
        norm   = pdf.sum(-1) * dz                           # (B,) — normalising factor
        z_mean = z_mean / norm.clamp(min=1e-8)

        return {"z_mean": z_mean, "z_median": z_median, "z_mode": z_mode}

    # ------------------------------------------------------------------
    # Regularisation penalties
    # ------------------------------------------------------------------

    def fisher_penalty(
        self,
        widths:  torch.Tensor,
        heights: torch.Tensor,
        derivs:  torch.Tensor,
        z_true:  torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Fisher information of the spline PDF evaluated at z_true:
            R = (d/dz log p(z))² |_{z=z_true}
          = (d²/dz² CDF(z) / p(z))²

        Computed numerically via central finite differences on log_pdf.
        Discourages sharp posteriors. Use for U-shaped PIT.
        """
        eps = 1e-3
        z_c = z_true.clamp(self.z_min + eps * 2, self.z_max - eps * 2)
        _, lp_c = _rqs_forward(z_c,       widths, heights, derivs, self.z_min, self.z_max)
        _, lp_r = _rqs_forward(z_c + eps, widths, heights, derivs, self.z_min, self.z_max)
        _, lp_l = _rqs_forward(z_c - eps, widths, heights, derivs, self.z_min, self.z_max)
        d_log_p = (lp_r - lp_l) / (2.0 * eps)   # d/dz log p ≈ score
        return d_log_p.pow(2).mean()

    def spread_penalty(
        self,
        widths:  torch.Tensor,
        heights: torch.Tensor,
        derivs:  torch.Tensor,
        n_grid:  int = 128,
    ) -> torch.Tensor:
        """
        Spread penalty: variance of the spline PDF distribution.
        E[z²] - E[z]² integrated numerically on a grid.
        Discourages broad posteriors. Use for n-shaped PIT.
        """
        z_g  = torch.linspace(self.z_min, self.z_max, n_grid, device=widths.device)
        B    = widths.shape[0]
        z_rep = z_g.unsqueeze(0).expand(B, -1).reshape(-1)
        w_rep = widths.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
        h_rep = heights.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
        d_rep = derivs.unsqueeze(1).expand(-1, n_grid, -1).reshape(B * n_grid, -1)
        _, log_pdf = _rqs_forward(z_rep, w_rep, h_rep, d_rep, self.z_min, self.z_max)
        pdf   = log_pdf.exp().reshape(B, n_grid)                    # (B, G)
        dz    = (self.z_max - self.z_min) / (n_grid - 1)
        norm  = (pdf * dz).sum(-1, keepdim=True).clamp(min=1e-8)
        pdf_n = pdf / norm                                           # normalised
        z_g_  = z_g.unsqueeze(0)
        z_mean = (pdf_n * z_g_ * dz).sum(-1)                        # (B,)
        var    = (pdf_n * (z_g_ - z_mean.unsqueeze(1)).pow(2) * dz).sum(-1)  # (B,)
        return var.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embedding: torch.Tensor,               # (B, embed_dim)
        z_true:    Optional[torch.Tensor] = None,  # (B,)
        fisher_lambda: float = 0.0,
        spread_lambda: float = 0.0,
        huber_lambda: float = 0.0,
        huber_delta: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        widths, heights, derivs = self._spline_params(embedding)

        # Primary point estimate: mode (best single prediction for the challenge)
        n_mode = 128
        device = embedding.device
        B = embedding.shape[0]
        z_grid  = torch.linspace(self.z_min, self.z_max, n_mode, device=device)
        z_rep   = z_grid.unsqueeze(0).expand(B, -1).reshape(-1)
        w_rep = widths.unsqueeze(1).expand(-1, n_mode, -1).reshape(B * n_mode, -1)
        h_rep = heights.unsqueeze(1).expand(-1, n_mode, -1).reshape(B * n_mode, -1)
        d_rep = derivs.unsqueeze(1).expand(-1, n_mode, -1).reshape(B * n_mode, -1)
        _, log_pdf_rep = _rqs_forward(z_rep, w_rep, h_rep, d_rep, self.z_min, self.z_max)
        z_pred = z_grid[log_pdf_rep.reshape(B, n_mode).argmax(-1)]   # (B,)

        result: Dict[str, torch.Tensor] = {
            "z_pred":  z_pred,
            "widths":  widths,
            "heights": heights,
            "derivs":  derivs,
        }

        if z_true is not None:
            z_clamp = z_true.clamp(self.z_min + 1e-6, self.z_max - 1e-6)
            _, log_pdf = _rqs_forward(z_clamp, widths, heights, derivs, self.z_min, self.z_max)
            loss = -log_pdf.mean()

            if fisher_lambda > 0.0:
                loss = loss + fisher_lambda * self.fisher_penalty(widths, heights, derivs, z_clamp)
            if spread_lambda > 0.0:
                loss = loss + spread_lambda * self.spread_penalty(widths, heights, derivs)
            if huber_lambda > 0.0:
                loss = loss + huber_lambda * F.huber_loss(
                    z_pred, z_true, delta=huber_delta,
                )

            result["loss"] = loss

        return result
