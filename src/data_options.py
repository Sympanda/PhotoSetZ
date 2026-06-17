"""
Resolve data-config options with backwards-compatible legacy field aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import DataConfig


@dataclass
class ResolvedDataOptions:
    include_errors: bool
    strict_error_columns: bool
    encode_nondetections: bool
    nondetection_policy: str
    nondetection_mag_fill: float
    nondetection_err_fill: float
    add_detection_flags: bool
    token_dim: int


def compute_token_dim(
    include_errors: bool = False,
    add_detection_flags: bool = False,
) -> int:
    """Expected per-token feature dimension (independent of filter inventory)."""
    dim = 4
    if include_errors:
        dim += 1
    if add_detection_flags:
        dim += 2
    return dim


def resolve_data_options(dc: DataConfig) -> ResolvedDataOptions:
    """
    Map config fields to dataset behaviour.

    Legacy aliases
    --------------
    preserve_nondetections=True  → encode_nondetections + keep_token
    allow_missing_error_cols     → inverse of strict_error_columns when set
    """
    strict = dc.strict_error_columns
    # Deprecated: allow_missing_error_cols=True means non-strict (zero-fill)
    if dc.allow_missing_error_cols:
        strict = False

    encode = dc.encode_nondetections
    policy = dc.nondetection_policy
    if dc.preserve_nondetections:
        encode = True
        policy = "keep_token"

    if policy not in ("drop", "keep_token"):
        raise ValueError(
            f"nondetection_policy must be 'drop' or 'keep_token', got '{policy}'"
        )

    return ResolvedDataOptions(
        include_errors=dc.include_errors,
        strict_error_columns=strict,
        encode_nondetections=encode,
        nondetection_policy=policy,
        nondetection_mag_fill=dc.nondetection_mag_fill,
        nondetection_err_fill=dc.nondetection_err_fill,
        add_detection_flags=dc.add_detection_flags,
        token_dim=compute_token_dim(dc.include_errors, dc.add_detection_flags),
    )


def dataset_kwargs_from_config(dc: DataConfig, **overrides) -> dict:
    """Build GalaxyDataset keyword arguments from DataConfig."""
    opts = resolve_data_options(dc)
    kw = {
        "include_errors": opts.include_errors,
        "strict_error_columns": opts.strict_error_columns,
        "encode_nondetections": opts.encode_nondetections,
        "nondetection_policy": opts.nondetection_policy,
        "nondetection_mag_fill": opts.nondetection_mag_fill,
        "nondetection_err_fill": opts.nondetection_err_fill,
        "add_detection_flags": opts.add_detection_flags,
    }
    kw.update(overrides)
    return kw
