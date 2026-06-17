"""
Mixture Density Network (MDN) head — continuous multimodal redshift posterior.

    p(z | X_g) = Σ_c π_c(X_g) N(z | μ_c(X_g), σ_c²(X_g))

Loss: negative log-likelihood under the mixture.
Point estimates: mean (Σ π_c μ_c), median (CDF⁻¹(0.5)), mode (argmax PDF).

Reference: Bishop (1994) "Mixture Density Networks".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG_SQRT_2PI = 0.5 * torch.tensor(2.0 * 3.141592653589793).log()


class MDN(nn.Module):
    """
    Parameters
    ----------
    embed_dim : int
        Dimension of the encoder output.
    n_components : int
        Number of Gaussian mixture components.
    hidden_dims : list[int]
        Hidden layers before the output projections.
    dropout : float
    sigma_min : float
        Minimum allowed component standard deviation (numerical stability).
    activation : str
        Non-linearity: gelu | relu | silu | leaky_relu | tanh.
    """

    _ACTIVATIONS = {
        "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh,
    }

    def __init__(
        self,
        embed_dim: int,
        n_components: int = 5,
        hidden_dims: List[int] = (64,),
        dropout: float = 0.1,
        sigma_min: float = 0.01,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.n_components = n_components
        self.sigma_min = sigma_min

        act_cls = self._ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'.")

        layers: List[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_cls(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.pi_head    = nn.Linear(in_dim, n_components)        # mixture logits
        self.mu_head    = nn.Linear(in_dim, n_components)        # means
        self.sigma_head = nn.Linear(in_dim, n_components)        # log stds

    def _params(self, embedding: torch.Tensor):
        h      = self.trunk(embedding)                           # (B, in_dim)
        pi     = torch.softmax(self.pi_head(h), dim=-1)         # (B, C)
        mu     = self.mu_head(h)                                 # (B, C)  — unbounded
        sigma  = F.softplus(self.sigma_head(h)) + self.sigma_min # (B, C)  — positive
        return pi, mu, sigma

    @staticmethod
    def _log_gaussian(z: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Log of N(z | mu, sigma²) — shapes broadcast."""
        return -(z - mu) ** 2 / (2.0 * sigma ** 2) - sigma.log() - _LOG_SQRT_2PI.to(z.device)

    # ------------------------------------------------------------------
    # Probabilistic query methods
    # ------------------------------------------------------------------

    def cdf_at(
        self,
        pi: torch.Tensor,    # (B, C)
        mu: torch.Tensor,    # (B, C)
        sigma: torch.Tensor, # (B, C)
        z_query: torch.Tensor,  # (B,) — one value per galaxy
    ) -> torch.Tensor:
        """
        Evaluate the mixture CDF at z_query[i] for galaxy i.
        Uses the standard Normal CDF (error function).
        Returns (B,) tensor of CDF values in [0, 1].
        """
        from torch.distributions import Normal
        z = z_query.unsqueeze(-1)           # (B, 1) → broadcast over C
        normal_cdf = Normal(mu, sigma).cdf(z)   # (B, C)
        return (pi * normal_cdf).sum(dim=-1)    # (B,)

    def pit_values(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        z_true_raw: torch.Tensor,  # (B,) in prediction space (log(1+z) if log_target)
    ) -> torch.Tensor:
        """
        Probability Integral Transform: CDF_i(z_true_i).
        Uniform[0,1] for a perfectly calibrated model.
        """
        return self.cdf_at(pi, mu, sigma, z_true_raw)

    def point_estimates_from_params(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        n_grid: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean, median, and mode from MDN parameters.
        All returned values are in the same space as mu.

        Parameters
        ----------
        n_grid : resolution of the z-grid used for median/mode estimation.
        """
        # Mean: closed-form weighted sum of component means
        z_mean = (pi * mu).sum(dim=-1)   # (B,)

        # Grid range: cover ±5σ around the widest component
        z_lo = float((mu - 5 * sigma).min().item())
        z_hi = float((mu + 5 * sigma).max().item())
        device = mu.device
        z_grid = torch.linspace(z_lo, z_hi, n_grid, device=device)   # (G,)

        # CDF on grid: (B, G)
        from torch.distributions import Normal
        z_g  = z_grid.view(1, 1, -1)          # (1, 1, G)
        mu_  = mu.unsqueeze(-1)                # (B, C, 1)
        sig_ = sigma.unsqueeze(-1)             # (B, C, 1)
        pi_  = pi.unsqueeze(-1)                # (B, C, 1)
        cdf_grid = (pi_ * Normal(mu_, sig_).cdf(z_g)).sum(dim=1)   # (B, G)

        # Median: first grid point where CDF ≥ 0.5, with linear interpolation
        above = (cdf_grid >= 0.5)                       # (B, G)
        idx_hi = above.long().argmax(dim=1).clamp(1, n_grid - 1)   # (B,)
        idx_lo = (idx_hi - 1).clamp(0, n_grid - 2)
        cdf_lo = cdf_grid.gather(1, idx_lo.unsqueeze(1)).squeeze(1)
        cdf_hi = cdf_grid.gather(1, idx_hi.unsqueeze(1)).squeeze(1)
        z_lo_v = z_grid[idx_lo]
        z_hi_v = z_grid[idx_hi]
        t = (0.5 - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-8)
        z_median = z_lo_v + t * (z_hi_v - z_lo_v)     # (B,)

        # Mode: argmax of log-PDF on the grid
        log_pi_ = torch.log(pi_.clamp(min=1e-8))       # (B, C, 1)
        log_g   = self._log_gaussian(z_g, mu_, sig_)   # (B, C, G)
        log_pdf = torch.logsumexp(log_pi_ + log_g, dim=1)  # (B, G)
        z_mode  = z_grid[log_pdf.argmax(dim=-1)]            # (B,)

        return {"z_mean": z_mean, "z_median": z_median, "z_mode": z_mode}

    # ------------------------------------------------------------------
    # Regularisation penalties
    # ------------------------------------------------------------------

    def fisher_penalty(
        self,
        pi:    torch.Tensor,   # (B, C)
        mu:    torch.Tensor,
        sigma: torch.Tensor,
        z_true: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Fisher information penalty evaluated at z_true:
            R = E[ (d/dz log p(z|x))² |_{z=z_true} ]

        Discourages *sharp* (over-confident) posteriors.
        Use when PIT histogram has a U-shape (under-dispersed model).
        fisher_lambda > 0 adds this to the loss.
        """
        z_exp   = z_true.unsqueeze(1)                          # (B, 1)
        log_p   = self._log_gaussian(z_exp, mu, sigma)         # (B, C)
        comp_p  = pi * log_p.exp()                             # (B, C)
        p_z     = comp_p.sum(dim=-1).clamp(min=1e-8)           # (B,)

        # d/dz log p = [Σ_c π_c N(z|μ_c,σ_c) · -(z-μ_c)/σ_c²] / p(z)
        score_numer = (comp_p * (-(z_exp - mu) / sigma.pow(2))).sum(dim=-1)  # (B,)
        score       = score_numer / p_z                                        # (B,)
        return score.pow(2).mean()

    def spread_penalty(
        self,
        pi:    torch.Tensor,   # (B, C)
        sigma: torch.Tensor,   # (B, C)
    ) -> torch.Tensor:
        """
        Spread (over-dispersion) penalty:
            R = E[ Σ_c π_c σ_c² ]

        Discourages *broad* (over-dispersed) posteriors by penalising
        the expected variance of the mixture.
        Use when PIT histogram has an n-shape (our case).
        spread_lambda > 0 adds this to the loss.
        """
        return (pi * sigma.pow(2)).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embedding: torch.Tensor,                    # (B, embed_dim)
        z_true: Optional[torch.Tensor] = None,      # (B,)
        fisher_lambda: float = 0.0,
        spread_lambda: float = 0.0,
        huber_lambda: float = 0.0,
        huber_delta: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        pi, mu, sigma = self._params(embedding)  # each (B, C)

        # Primary point estimate: mixture mean Σ_c π_c μ_c (matches Huber aux + metrics)
        z_pred = (pi * mu).sum(dim=-1)  # (B,)

        result: Dict[str, torch.Tensor] = {
            "z_pred": z_pred,
            "pi":     pi,
            "mu":     mu,
            "sigma":  sigma,
        }

        if z_true is not None:
            # NLL = -log Σ_c π_c N(z | μ_c, σ_c)
            z_exp   = z_true.unsqueeze(1)
            log_p   = self._log_gaussian(z_exp, mu, sigma)
            log_pi  = torch.log(pi.clamp(min=1e-8))
            log_mix = torch.logsumexp(log_pi + log_p, dim=-1)
            loss    = -log_mix.mean()

            if fisher_lambda > 0.0:
                loss = loss + fisher_lambda * self.fisher_penalty(pi, mu, sigma, z_true)
            if spread_lambda > 0.0:
                loss = loss + spread_lambda * self.spread_penalty(pi, sigma)
            if huber_lambda > 0.0:
                z_point = (pi * mu).sum(dim=-1)
                loss = loss + huber_lambda * F.huber_loss(
                    z_point, z_true, delta=huber_delta,
                )

            result["loss"] = loss

        return result
