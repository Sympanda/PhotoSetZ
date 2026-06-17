"""Context construction and projection for SBI/NPE posterior heads."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from src.coverage_summary import (
    LOG_LAMBDA_IDX,
    LOG_LAMBDA_REF,
    LOG_LAMBDA_SCALE,
    SIGMA_REF,
    SIGMA_SCALE,
    _token_feature_indices,
)

# Extended fixed-size summaries for SBI (filter-inventory independent)
SBI_COVERAGE_SUMMARY_DIM = 11


def compute_sbi_coverage_summary(
    tokens: torch.Tensor,
    key_padding_mask: torch.Tensor,
    *,
    n_total_filters: int,
    include_errors: bool = False,
    add_detection_flags: bool = False,
) -> torch.Tensor:
    """
    Fixed-size coverage / wavelength summaries from variable token sets.

    Returns (B, SBI_COVERAGE_SUMMARY_DIM):
        n_tokens_norm, n_det, n_nondet, frac_det, frac_nondet,
        lambda_min_norm, lambda_max_norm, log_lambda_span_norm,
        largest_gap_norm, mean_sigma_norm, median_sigma_norm
    """
    B, N, _ = tokens.shape
    device = tokens.device
    active = ~key_padding_mask
    n_active = active.float().sum(dim=-1)
    n_denom = max(float(n_total_filters), 1.0)
    feats = _token_feature_indices(include_errors, add_detection_flags)

    out = torch.zeros(B, SBI_COVERAGE_SUMMARY_DIM, device=device)
    out[:, 0] = n_active / n_denom

    for b in range(B):
        idx = active[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        t = tokens[b, idx]
        n_det = float(idx.numel())
        n_nondet = max(n_denom - n_det, 0.0)
        out[b, 1] = n_det / n_denom
        out[b, 2] = n_nondet / n_denom
        out[b, 3] = n_det / n_denom
        out[b, 4] = n_nondet / n_denom

        lam = t[:, LOG_LAMBDA_IDX] * LOG_LAMBDA_SCALE + LOG_LAMBDA_REF
        lam_sorted, _ = lam.sort()
        lam_min, lam_max = lam_sorted[0], lam_sorted[-1]
        span = (lam_max - lam_min).clamp(min=0.0)
        out[b, 5] = (lam_min - LOG_LAMBDA_REF) / LOG_LAMBDA_SCALE
        out[b, 6] = (lam_max - LOG_LAMBDA_REF) / LOG_LAMBDA_SCALE
        out[b, 7] = span / LOG_LAMBDA_SCALE
        if lam_sorted.numel() > 1:
            gaps = lam_sorted[1:] - lam_sorted[:-1]
            out[b, 8] = gaps.max() / LOG_LAMBDA_SCALE
        else:
            out[b, 8] = 0.0

        if feats["sigma"] is not None and t.size(1) > feats["sigma"]:
            sig = t[:, feats["sigma"]] * SIGMA_SCALE + SIGMA_REF
            out[b, 9] = (sig.mean() - SIGMA_REF) / SIGMA_SCALE
            out[b, 10] = (sig.median() - SIGMA_REF) / SIGMA_SCALE

        if feats["detected"] is not None and t.size(1) > feats["detected"]:
            det = t[:, feats["detected"]]
            n_d = det.sum().item()
            n_nd = max(idx.numel() - n_d, 0.0)
            out[b, 1] = n_d / n_denom
            out[b, 2] = n_nd / n_denom
            out[b, 3] = n_d / max(n_d + n_nd, 1.0)
            out[b, 4] = n_nd / max(n_d + n_nd, 1.0)

    return out


class DensityContextProjection(nn.Module):
    """Compact LayerNorm + MLP projection for density-estimator context."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int] = (128, 64),
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        out_dim = out_dim or hidden_dims[-1]
        layers: List[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        if d != out_dim:
            layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SBIContextBuilder(nn.Module):
    """Build conditional context vectors for SBI/NPE from encoder + point head."""

    def __init__(
        self,
        *,
        context_mode: str,
        repr_dim: int,
        include_point_prediction: bool,
        include_coverage_summary: bool,
        context_projection_enabled: bool,
        projection_hidden: List[int],
        projection_dropout: float,
        projection_layer_norm: bool,
        context_dim: Optional[int],
        n_total_filters: int,
        include_errors: bool,
        add_detection_flags: bool,
        detach_context: bool = False,
    ) -> None:
        super().__init__()
        self.context_mode = context_mode
        self.include_point_prediction = include_point_prediction
        self.include_coverage_summary = include_coverage_summary
        self.n_total_filters = n_total_filters
        self.include_errors = include_errors
        self.add_detection_flags = add_detection_flags
        self.detach_context = detach_context

        raw_dim = self._raw_context_dim(repr_dim)
        if context_projection_enabled:
            out_dim = context_dim or projection_hidden[-1]
            self.projection: Optional[DensityContextProjection] = DensityContextProjection(
                raw_dim,
                hidden_dims=projection_hidden,
                out_dim=out_dim,
                dropout=projection_dropout,
                layer_norm=projection_layer_norm,
            )
            self.context_dim = out_dim
        else:
            self.projection = None
            self.context_dim = raw_dim

    def _raw_context_dim(self, repr_dim: int) -> int:
        d = 0
        mode = self.context_mode
        if mode in ("frozen_point", "point_only", "ensemble_summary"):
            if self.include_point_prediction:
                d += 1
        if mode in ("frozen_point", "latent_only", "ensemble_summary"):
            d += repr_dim
        if self.include_coverage_summary:
            d += SBI_COVERAGE_SUMMARY_DIM
        if d == 0:
            raise ValueError(f"SBI context_mode={mode!r} produced empty context.")
        return d

    def forward(
        self,
        h: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        y_hat_log: torch.Tensor,
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        mode = self.context_mode

        if mode in ("frozen_point", "point_only", "ensemble_summary"):
            if self.include_point_prediction:
                parts.append(y_hat_log.unsqueeze(-1))
        if mode in ("frozen_point", "latent_only", "ensemble_summary"):
            parts.append(h)
        if self.include_coverage_summary:
            parts.append(
                compute_sbi_coverage_summary(
                    tokens,
                    key_padding_mask,
                    n_total_filters=self.n_total_filters,
                    include_errors=self.include_errors,
                    add_detection_flags=self.add_detection_flags,
                )
            )

        ctx = torch.cat(parts, dim=-1)
        if self.detach_context:
            ctx = ctx.detach()
        if self.projection is not None:
            ctx = self.projection(ctx)
        return ctx
