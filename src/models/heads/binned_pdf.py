"""
Binned PDF head — predicts a discrete redshift posterior over K uniform bins.

    p(z_k | X_g) = softmax_k(W h + b),  k = 1, …, K

Loss: negative log-likelihood at the true redshift bin (cross-entropy).
Point estimates: mean (weighted average), median (CDF inversion), mode (argmax bin).

This head is typically more stable to train than an MDN and easier to
calibrate; it naturally captures multimodal posteriors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinnedPDF(nn.Module):
    """
    Parameters
    ----------
    embed_dim : int
        Dimension of the encoder output.
    n_bins : int
        Number of redshift bins.
    z_min, z_max : float
        Redshift range covered by the bins.
    hidden_dims : list[int]
        Hidden layers before the output projection.
    dropout : float
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
        n_bins: int = 100,
        z_min: float = 0.0,
        z_max: float = 3.0,
        hidden_dims: List[int] = (64,),
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.z_min  = z_min
        self.z_max  = z_max

        act_cls = self._ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'.")

        # Bin centres — registered as a buffer so they move with the model device
        edges = torch.linspace(z_min, z_max, n_bins + 1)
        bin_centres = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("bin_centres", bin_centres)   # (K,)
        bin_width = (z_max - z_min) / n_bins
        self.register_buffer("bin_width", torch.tensor(bin_width))

        layers: List[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_cls(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_bins))
        self.net = nn.Sequential(*layers)

    def _z_to_bin(self, z: torch.Tensor) -> torch.Tensor:
        """Map continuous redshift values to bin indices (clamped)."""
        idx = ((z - self.z_min) / (self.z_max - self.z_min) * self.n_bins).long()
        return idx.clamp(0, self.n_bins - 1)

    # ------------------------------------------------------------------
    # Probabilistic query methods
    # ------------------------------------------------------------------

    def cdf_at(self, probs: torch.Tensor, z_query: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the discrete CDF at z_query[i] for each galaxy i.

        Uses linear interpolation between adjacent bin edges so that
        the result is continuous rather than a staircase.

        Parameters
        ----------
        probs   : (B, K) softmax probabilities
        z_query : (B,) — one query point per galaxy, in same space as bin_centres

        Returns
        -------
        (B,) CDF values in [0, 1]
        """
        # Cumulative sum over bins gives the CDF at each bin's right edge
        cumprobs = probs.cumsum(dim=-1)   # (B, K)

        # Map z_query to fractional bin position
        frac = (z_query - self.z_min) / (self.z_max - self.z_min) * self.n_bins  # (B,)
        frac = frac.clamp(0.0, float(self.n_bins))

        # Bin index where the query falls
        k = frac.long().clamp(0, self.n_bins - 1)    # (B,)

        # CDF up to the start of bin k  (=0 for k=0)
        cdf_lo = torch.where(k > 0,
                             cumprobs.gather(1, (k - 1).clamp(min=0).unsqueeze(1)).squeeze(1),
                             torch.zeros_like(frac))
        # Add fractional contribution of the current bin
        p_k = probs.gather(1, k.unsqueeze(1)).squeeze(1)    # (B,)
        frac_within_bin = (frac - k.float()).clamp(0.0, 1.0)
        return (cdf_lo + p_k * frac_within_bin).clamp(0.0, 1.0)

    def pit_values(
        self,
        probs: torch.Tensor,       # (B, K)
        z_true_raw: torch.Tensor,  # (B,) in same space as bin_centres
    ) -> torch.Tensor:
        """
        Probability Integral Transform: CDF_i(z_true_i).
        Uniform[0,1] for a perfectly calibrated model.
        """
        return self.cdf_at(probs, z_true_raw)

    def point_estimates_from_probs(
        self,
        probs: torch.Tensor,  # (B, K)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean, median, and mode from bin probabilities.
        All returned values are in the same space as bin_centres.
        """
        # Mean: weighted average (same as existing z_pred)
        z_mean = (probs * self.bin_centres).sum(dim=-1)   # (B,)

        # Mode: bin centre with the highest probability
        z_mode = self.bin_centres[probs.argmax(dim=-1)]   # (B,)

        # Median: find bin where cumulative probability crosses 0.5,
        # then interpolate within that bin
        cumprobs = probs.cumsum(dim=-1)                    # (B, K)
        above    = (cumprobs >= 0.5)                       # (B, K)
        k_hi     = above.long().argmax(dim=1).clamp(1, self.n_bins - 1)   # (B,)
        k_lo     = k_hi - 1

        cdf_lo = cumprobs.gather(1, k_lo.unsqueeze(1)).squeeze(1)
        cdf_hi = cumprobs.gather(1, k_hi.unsqueeze(1)).squeeze(1)
        bin_w  = float(self.bin_width.item())
        # Left edge of bin k_lo
        z_lo_v = self.bin_centres[k_lo] - bin_w * 0.5
        t      = (0.5 - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-8)
        z_median = z_lo_v + t * bin_w                     # (B,)

        return {"z_mean": z_mean, "z_median": z_median, "z_mode": z_mode}

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Regularisation penalties
    # ------------------------------------------------------------------

    def fisher_penalty(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Fisher information proxy: mean squared finite-difference gradient of the PDF.
        Discourages sharp posteriors (over-confident). Use for U-shaped PIT.
        """
        dw = float(self.bin_width)
        # Central finite differences along z; reflect at boundaries
        pdf = probs / dw                                 # (B, K)
        pdf_right = torch.cat([pdf[:, 1:], pdf[:, -1:]], dim=-1)
        pdf_left  = torch.cat([pdf[:, :1], pdf[:, :-1]], dim=-1)
        grad = (pdf_right - pdf_left) / (2.0 * dw)      # (B, K)
        return (grad.pow(2) * probs).sum(dim=-1).mean()

    def spread_penalty(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Spread penalty: expected squared deviation from the mean.
        Discourages broad posteriors (over-dispersed). Use for n-shaped PIT.
        """
        centres = self.bin_centres                       # (K,)
        z_mean  = (probs * centres).sum(dim=-1)          # (B,)
        var     = (probs * (centres - z_mean.unsqueeze(1)).pow(2)).sum(dim=-1)  # (B,)
        return var.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embedding: torch.Tensor,               # (B, embed_dim)
        z_true: Optional[torch.Tensor] = None, # (B,)
        fisher_lambda: float = 0.0,
        spread_lambda: float = 0.0,
        huber_lambda: float = 0.0,
        huber_delta: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        logits = self.net(embedding)            # (B, K)
        probs  = torch.softmax(logits, dim=-1)  # (B, K)

        # Primary point estimate: weighted mean of bin centres
        z_pred = (probs * self.bin_centres).sum(dim=-1)  # (B,)

        result: Dict[str, torch.Tensor] = {
            "z_pred":  z_pred,
            "probs":   probs,             # full posterior (B, K)
            "logits":  logits,
        }

        if z_true is not None:
            bin_idx = self._z_to_bin(z_true)
            loss    = F.cross_entropy(logits, bin_idx)

            if fisher_lambda > 0.0:
                loss = loss + fisher_lambda * self.fisher_penalty(probs)
            if spread_lambda > 0.0:
                loss = loss + spread_lambda * self.spread_penalty(probs)
            if huber_lambda > 0.0:
                loss = loss + huber_lambda * F.huber_loss(
                    z_pred, z_true, delta=huber_delta,
                )

            result["loss"] = loss

        return result
