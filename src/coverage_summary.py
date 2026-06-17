"""
Fixed-size wavelength/coverage summaries from variable token sets.

Summaries are batch-invariant for a given galaxy: denominators use a fixed
``n_total_filters`` from the dataset registry, not the max tokens in the batch.
"""

from __future__ import annotations

from typing import Optional

import torch

# Number of scalar features appended when use_coverage_summary=True
COVERAGE_SUMMARY_DIM = 8

# Normalisation constants (match dataset.py)
LOG_LAMBDA_REF = 4.0
LOG_LAMBDA_SCALE = 0.7
SIGMA_REF = 0.1
SIGMA_SCALE = 0.5

# Token layout: [mag, log_lambda, log_delta_lambda, survey, (sigma), (flags...)]
MAG_IDX = 0
LOG_LAMBDA_IDX = 1


def _token_feature_indices(
    include_errors: bool,
    add_detection_flags: bool,
) -> dict:
    sigma_idx = 4 if include_errors else None
    detected_idx = None
    if add_detection_flags:
        base = 5 if include_errors else 4
        detected_idx = base
    return {"sigma": sigma_idx, "detected": detected_idx}


def compute_coverage_summary(
    tokens: torch.Tensor,
    key_padding_mask: torch.Tensor,
    *,
    n_total_filters: int,
    include_errors: bool = False,
    add_detection_flags: bool = False,
) -> torch.Tensor:
    """
    Compute fixed-size coverage summaries per galaxy.

    Parameters
    ----------
    tokens : (B, N, token_dim)
    key_padding_mask : (B, N)  True = padding
    n_total_filters : fixed registry filter count (denominator)

    Returns
    -------
    summary : (B, COVERAGE_SUMMARY_DIM)
        [n_tokens_norm, lambda_min_norm, lambda_max_norm, lambda_span_norm,
         max_gap_norm, mean_gap_norm, mean_sigma_norm, frac_nondetected]
    """
    B, N, _ = tokens.shape
    device = tokens.device
    active = ~key_padding_mask                                                    # (B, N)
    n_active = active.float().sum(dim=-1)                                         # (B,)
    n_denom = max(float(n_total_filters), 1.0)

    feats = _token_feature_indices(include_errors, add_detection_flags)

    # Default summaries for empty token sets
    out = torch.zeros(B, COVERAGE_SUMMARY_DIM, device=device)
    out[:, 0] = n_active / n_denom

    for b in range(B):
        idx = active[b].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        t = tokens[b, idx]
        lam = t[:, LOG_LAMBDA_IDX] * LOG_LAMBDA_SCALE + LOG_LAMBDA_REF            # denorm log10 λ

        lam_sorted, _ = lam.sort()
        lam_min = lam_sorted[0]
        lam_max = lam_sorted[-1]
        span = (lam_max - lam_min).clamp(min=0.0)
        if lam_sorted.numel() > 1:
            gaps = lam_sorted[1:] - lam_sorted[:-1]
            max_gap = gaps.max()
            mean_gap = gaps.mean()
        else:
            max_gap = torch.zeros((), device=device)
            mean_gap = torch.zeros((), device=device)

        out[b, 1] = (lam_min - LOG_LAMBDA_REF) / LOG_LAMBDA_SCALE
        out[b, 2] = (lam_max - LOG_LAMBDA_REF) / LOG_LAMBDA_SCALE
        out[b, 3] = span / LOG_LAMBDA_SCALE
        out[b, 4] = max_gap / LOG_LAMBDA_SCALE
        out[b, 5] = mean_gap / LOG_LAMBDA_SCALE

        if feats["sigma"] is not None and t.size(1) > feats["sigma"]:
            sig = t[:, feats["sigma"]] * SIGMA_SCALE + SIGMA_REF
            out[b, 6] = ((sig.mean() - SIGMA_REF) / SIGMA_SCALE)

        if feats["detected"] is not None and t.size(1) > feats["detected"]:
            detected = t[:, feats["detected"]]
            out[b, 7] = (1.0 - detected).mean()

    return out
