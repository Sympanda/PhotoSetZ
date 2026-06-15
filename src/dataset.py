"""
Galaxy photometry dataset for DeepSetZ.

Each galaxy is represented as a variable-length set of photometric tokens.
Each token corresponds to one filter measurement with features:

    Without errors (include_errors=False, token_dim=4):
        x_i = [m_i_norm, log_λ_eff_norm, log_Δλ_norm, survey_id_norm]

    With errors    (include_errors=True,  token_dim=5):
        x_i = [m_i_norm, log_λ_eff_norm, log_Δλ_norm, survey_id_norm, σ_m_norm]

Non-detections / missing bands are represented as NaN in the parquet.
Galaxies where ALL filter columns are NaN are excluded.  Per-galaxy NaN
masks are applied at __getitem__ time so that tokens for missing bands are
never fed to the network — no imputation needed.

During training a stratified dropout strategy further augments coverage:

    Complete   (~15%)  — all available filters
    Preset     (~25%)  — named realistic survey combos (DECaLS, LSST-only, …)
    Survey     (~25%)  — drop 1–N entire surveys, optionally thin afterwards
    Aggressive (~35%)  — random per-filter rate in [low, high]

All modes enforce min_filters (default 3) from the pool of *valid* (non-NaN)
filters so the model always receives at least some signal.

Batching pads variable-length token sets; key_padding_mask=True marks padding
(PyTorch nn.MultiheadAttention convention).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .filters import FilterInfo, build_filter_registry


# ------------------------------------------------------------------
# Normalisation constants
# ------------------------------------------------------------------
MAG_REF           = 25.0
MAG_SCALE         = 3.0
LOG_LAMBDA_REF    = 4.0    # log10(10 000 Å)
LOG_LAMBDA_SCALE  = 0.7
LOG_DLAMBDA_REF   = 3.0    # log10(1 000 Å)
LOG_DLAMBDA_SCALE = 0.5
SURVEY_SCALE      = 3.0    # survey_id in {0,1,2,3} → [0, 1]
SIGMA_REF         = 0.1    # reference error ~0.1 mag
SIGMA_SCALE       = 0.5    # σ_norm = (σ - 0.1) / 0.5

TOKEN_DIM_BASE   = 4   # without magnitude errors
TOKEN_DIM_ERRORS = 5   # with magnitude errors


# ------------------------------------------------------------------
# Named preset survey combinations
# Both Ellen column names and PZDC variants are listed; the dataset
# silently ignores any names not in its registry.
# ------------------------------------------------------------------
_ROMAN_ELLEN = ["Roman_Y106", "Roman_J129", "Roman_H158", "Roman_F184", "Roman_K213"]
_ROMAN_PZDC_LETTERS  = ["mag_Y_roman", "mag_J_roman", "mag_H_roman",
                         "mag_F_roman", "mag_K_roman"]
_ROMAN_PZDC_NUMBERS  = ["mag_F062_roman", "mag_F087_roman", "mag_F106_roman",
                         "mag_F129_roman", "mag_F158_roman",
                         "mag_F184_roman", "mag_F213_roman"]
_ROMAN_ALL = _ROMAN_ELLEN + _ROMAN_PZDC_LETTERS + _ROMAN_PZDC_NUMBERS

_LSST_ALL  = ["mag_u_lsst", "mag_g_lsst", "mag_r_lsst",
               "mag_i_lsst", "mag_z_lsst", "mag_y_lsst"]
_EUCLID_ALL = ["Euclid_Y", "Euclid_J", "Euclid_H",
               "mag_Y_euclid", "mag_J_euclid", "mag_H_euclid"]
_WISE_ALL  = ["WISE_W1", "WISE_W2"]

SURVEY_PRESETS: Dict[str, List[str]] = {
    "decals":          ["mag_g_lsst", "mag_r_lsst", "mag_z_lsst"],
    "hsc_wide":        ["mag_g_lsst", "mag_r_lsst", "mag_i_lsst",
                        "mag_z_lsst", "mag_y_lsst"],
    "lsst_only":       _LSST_ALL,
    "lsst_roman":      _LSST_ALL + _ROMAN_ALL,
    "lsst_euclid":     _LSST_ALL + _EUCLID_ALL,
    "lsst_wise":       _LSST_ALL + _WISE_ALL,
    "roman_only":      _ROMAN_ALL,
    "roman_euclid":    _ROMAN_ALL + _EUCLID_ALL,
    "lsst_wise_full":  _LSST_ALL + _WISE_ALL,
}


# ------------------------------------------------------------------
# Dropout configuration
# ------------------------------------------------------------------

@dataclass
class DropoutConfig:
    """
    Controls the stratified filter dropout applied during training.

    Sampling probabilities (must sum to ~1.0):
        p_complete      : keep all filters
        p_preset        : use a named survey preset
        p_survey_drop   : drop 1–max_surveys_to_drop entire surveys
        p_aggressive    : random per-filter dropout with rate in
                          [aggressive_rate_low, aggressive_rate_high]

    After survey-level dropping, an additional per-filter thinning step
    can be applied with probability p_filter_after_survey.

    min_filters is enforced from the pool of *valid* (non-NaN) filters.
    """
    p_complete:              float = 0.15
    p_preset:                float = 0.25
    p_survey_drop:           float = 0.25
    p_aggressive:            float = 0.35

    aggressive_rate_low:     float = 0.3
    aggressive_rate_high:    float = 0.8

    max_surveys_to_drop:     int   = 2
    p_filter_after_survey:   float = 0.4
    filter_rate_after_survey: float = 0.25

    min_filters:             int   = 3

    def __post_init__(self):
        total = self.p_complete + self.p_preset + self.p_survey_drop + self.p_aggressive
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Dropout probabilities must sum to 1.0, got {total:.4f}. "
                "Adjust p_complete, p_preset, p_survey_drop, p_aggressive."
            )


_NO_DROPOUT = DropoutConfig(
    p_complete=1.0, p_preset=0.0, p_survey_drop=0.0, p_aggressive=0.0,
    min_filters=1,
)


class FilterDropout:
    """
    Applies the stratified dropout strategy to a boolean keep-mask.

    Survey groups are built dynamically from the FilterInfo.survey attribute
    so the class works with any column naming convention (Ellen or PZDC).
    """

    def __init__(self, cfg: DropoutConfig, filter_infos: List[FilterInfo]) -> None:
        self.cfg   = cfg
        self.names = [fi.col_name for fi in filter_infos]
        self.n     = len(filter_infos)
        self._name_to_idx = {n: i for i, n in enumerate(self.names)}

        # Build survey groups dynamically from the registry
        _survey_to_idx: Dict[str, list] = defaultdict(list)
        for i, fi in enumerate(filter_infos):
            _survey_to_idx[fi.survey].append(i)
        self._survey_indices: Dict[str, np.ndarray] = {
            s: np.array(idxs, dtype=np.intp)
            for s, idxs in _survey_to_idx.items()
        }
        self._survey_keys = list(self._survey_indices.keys())

        # Pre-resolve preset → index arrays (silently skip absent columns)
        self._preset_indices: Dict[str, np.ndarray] = {}
        for pname, cols in SURVEY_PRESETS.items():
            idx = np.array([self._name_to_idx[c] for c in cols
                            if c in self._name_to_idx], dtype=np.intp)
            if len(idx) >= cfg.min_filters:
                self._preset_indices[pname] = idx
        self._preset_keys = list(self._preset_indices.keys())

    def _enforce_min(self, keep: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Force at least min_filters tokens to be kept, drawing only from the
        pool of valid (non-NaN) filters.
        """
        n_keep = int(keep.sum())
        target = min(self.cfg.min_filters, int(valid.sum()))
        if n_keep < target:
            # Candidates: valid but not yet kept
            candidates = np.where(valid & ~keep)[0]
            needed = target - n_keep
            if len(candidates) >= needed:
                chosen = np.random.choice(candidates, size=needed, replace=False)
                keep[chosen] = True
            else:
                keep[candidates] = True   # keep all remaining valid ones
        return keep

    def apply(self, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return a boolean keep-mask of shape (n_filters,).

        Parameters
        ----------
        valid_mask : array of bool, shape (n_filters,) or None
            If provided, represents which filters have non-NaN observations.
            Dropout is only applied within this set; _enforce_min also only
            uses valid filters.  None means all filters are valid.
        """
        if valid_mask is None:
            valid_mask = np.ones(self.n, dtype=bool)

        cfg  = self.cfg
        roll = np.random.random()

        # ── Complete (all valid) ─────────────────────────────────────
        if roll < cfg.p_complete:
            return valid_mask.copy()

        # ── Preset ──────────────────────────────────────────────────
        roll -= cfg.p_complete
        if roll < cfg.p_preset and self._preset_keys:
            key  = self._preset_keys[np.random.randint(len(self._preset_keys))]
            keep = np.zeros(self.n, dtype=bool)
            keep[self._preset_indices[key]] = True
            keep &= valid_mask   # can't keep filters we don't have
            return self._enforce_min(keep, valid_mask)

        # ── Survey-level drop ────────────────────────────────────────
        roll -= cfg.p_preset
        if roll < cfg.p_survey_drop:
            keep = valid_mask.copy()
            n_drop = np.random.randint(1, cfg.max_surveys_to_drop + 1)
            drop_surveys = np.random.choice(
                self._survey_keys,
                size=min(n_drop, len(self._survey_keys)),
                replace=False,
            )
            for s in drop_surveys:
                keep[self._survey_indices[s]] = False

            if np.random.random() < cfg.p_filter_after_survey:
                on_idx = np.where(keep)[0]
                thin   = np.random.random(len(on_idx)) < cfg.filter_rate_after_survey
                keep[on_idx[thin]] = False

            return self._enforce_min(keep, valid_mask)

        # ── Aggressive per-filter ────────────────────────────────────
        rate = np.random.uniform(cfg.aggressive_rate_low, cfg.aggressive_rate_high)
        keep = (np.random.random(self.n) >= rate) & valid_mask
        return self._enforce_min(keep, valid_mask)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class GalaxyDataset(Dataset):
    """
    Loads galaxies from a parquet file and presents each as a token set.

    Parameters
    ----------
    parquet_path : str or Path
    res_dir : str or Path
        Directory containing .res filter transmission files.
    target_col : str
        Redshift label column (e.g. ``"true_redshift"`` for Ellen data,
        ``"redshift"`` for PZDC data).
    dropout_cfg : DropoutConfig or None
        Stratified dropout for training.  Pass None for val/test (no dropout).
    active_surveys : list[str] or None
        Restrict to specific surveys ("lsst", "roman", …) or individual
        column names.  None / empty → use all available.
    log_target : bool
        Store redshifts as log(1+z).  All evaluation code reverses this.
    include_errors : bool
        If True, append a normalised σ_m feature to each token, extending
        token_dim from 4 → 5.  Requires the parquet to contain error columns
        named ``{col_name}_err``; missing error columns are filled with 0.0
        (i.e. the "perfect measurement" sentinel).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        res_dir: str | Path,
        target_col: str = "true_redshift",
        dropout_cfg: Optional[DropoutConfig] = None,
        active_surveys: Optional[List[str]] = None,
        log_target: bool = False,
        include_errors: bool = False,
    ) -> None:
        self.dropout_cfg    = dropout_cfg or _NO_DROPOUT
        self.log_target     = log_target
        self.include_errors = include_errors

        # Load catalogue first so we know which columns are available
        df = pd.read_parquet(parquet_path)

        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in parquet. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        # Build registry restricted to columns present in this parquet
        self.registry: Dict[str, FilterInfo] = build_filter_registry(
            res_dir,
            active_surveys=active_surveys or [],
            available_cols=set(df.columns),
        )
        if not self.registry:
            raise ValueError(
                "Filter registry is empty after filtering to available columns. "
                "Check res_dir, active_surveys, and that the parquet contains "
                "recognised photometric columns."
            )

        self.filters: List[FilterInfo] = list(self.registry.values())
        self.n_filters = len(self.filters)
        self._col_names = [fi.col_name for fi in self.filters]

        # ── Row filtering ────────────────────────────────────────────
        # Keep rows where the redshift target is valid AND at least one
        # photometric filter is non-NaN.
        valid_z  = df[target_col].notna()
        n_valid  = df[self._col_names].notna().sum(axis=1)
        valid_ph = n_valid >= 1
        df = df[valid_z & valid_ph].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No valid rows remain after filtering "
                f"(target='{target_col}', n_filter_cols={self.n_filters}). "
                "Check column names and parquet file."
            )

        # ── Object IDs (preserved for qp submission format) ──────────
        if "object_id" in df.columns:
            self.object_ids: np.ndarray = df["object_id"].values.copy()
        else:
            self.object_ids = np.arange(len(df), dtype=np.int64)

        # ── Magnitudes (may contain NaN for non-detections) ──────────
        self.magnitudes: np.ndarray = df[self._col_names].values.astype(np.float32)

        # ── Magnitude errors ─────────────────────────────────────────
        if include_errors:
            err_cols = []
            for fi in self.filters:
                ecol = fi.err_col_name
                if ecol and ecol in df.columns:
                    err_cols.append(df[ecol].values.astype(np.float32))
                else:
                    # No error column → fill with 0.0 (perfect measurement)
                    err_cols.append(np.zeros(len(df), dtype=np.float32))
            # shape (n_rows, n_filters)
            self.errors: Optional[np.ndarray] = np.stack(err_cols, axis=1)
        else:
            self.errors = None

        # ── Redshifts ────────────────────────────────────────────────
        raw_z = df[target_col].values.astype(np.float32)
        self.redshifts: np.ndarray = np.log1p(raw_z) if log_target else raw_z

        # ── Static per-filter metadata: (n_filters, 3) ───────────────
        self._filter_meta: np.ndarray = np.stack([
            np.array([
                (np.log10(fi.lambda_eff)   - LOG_LAMBDA_REF)  / LOG_LAMBDA_SCALE,
                (np.log10(fi.delta_lambda) - LOG_DLAMBDA_REF) / LOG_DLAMBDA_SCALE,
                fi.survey_id / SURVEY_SCALE,
            ], dtype=np.float32)
            for fi in self.filters
        ])  # (n_filters, 3)

        # ── Token dimension ───────────────────────────────────────────
        self.token_dim: int = TOKEN_DIM_ERRORS if include_errors else TOKEN_DIM_BASE

        # ── Dropout engine ────────────────────────────────────────────
        self._dropout = FilterDropout(self.dropout_cfg, self.filters)

    def set_dropout_cfg(self, dropout_cfg: Optional[DropoutConfig]) -> None:
        """Swap dropout strategy at runtime (e.g. curriculum full-filter warmup)."""
        self.dropout_cfg = dropout_cfg or _NO_DROPOUT
        self._dropout = FilterDropout(self.dropout_cfg, self.filters)

    def __len__(self) -> int:
        return len(self.redshifts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mags = self.magnitudes[idx]                        # (n_filters,)

        # Per-galaxy valid mask: True = filter has a measured (non-NaN) value
        valid = ~np.isnan(mags)

        # Normalise magnitudes; fill NaN positions with 0 (they'll be masked out)
        mag_norm = np.where(valid, (mags - MAG_REF) / MAG_SCALE, 0.0).astype(np.float32)

        # (n_filters, token_dim)
        tokens = np.concatenate([mag_norm[:, None], self._filter_meta], axis=1)

        if self.errors is not None:
            sigmas    = self.errors[idx]                   # (n_filters,)
            # Clip to [0, ∞), fill NaN with 0.0 (treated as perfect measurement)
            sigmas    = np.where(np.isnan(sigmas), 0.0, np.clip(sigmas, 0.0, None))
            sigma_norm = ((sigmas - SIGMA_REF) / SIGMA_SCALE).astype(np.float32)
            tokens    = np.concatenate([tokens, sigma_norm[:, None]], axis=1)

        # Combine data validity with training dropout
        keep   = self._dropout.apply(valid_mask=valid)
        tokens = tokens[keep]

        return (
            torch.from_numpy(tokens),
            torch.zeros(len(tokens), dtype=torch.bool),   # padding mask added in collate
            torch.tensor(self.redshifts[idx], dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Convenience: inspect dropout distribution
    # ------------------------------------------------------------------
    def sample_dropout_stats(self, n_samples: int = 10_000) -> Dict[str, float]:
        """
        Draw n_samples keep-masks and report average active filter counts.
        Useful for sanity-checking the dropout config.
        """
        # Use a fully-valid mask (no NaNs) to test purely the dropout logic
        full_valid = np.ones(self.n_filters, dtype=bool)
        counts = np.array([
            self._dropout.apply(valid_mask=full_valid).sum()
            for _ in range(n_samples)
        ])
        return {
            "mean_filters":   float(counts.mean()),
            "median_filters": float(np.median(counts)),
            "min_filters":    int(counts.min()),
            "max_filters":    int(counts.max()),
            "pct_3_or_fewer": float((counts <= 3).mean()),
            "pct_complete":   float((counts == self.n_filters).mean()),
        }


# ------------------------------------------------------------------
# Collate function
# ------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length token sets to the same length within a batch.

    Returns
    -------
    tokens : FloatTensor (B, max_N, token_dim)
    key_padding_mask : BoolTensor (B, max_N)
        True = padding position (PyTorch nn.MultiheadAttention convention).
    redshifts : FloatTensor (B,)
    """
    token_list, _, redshift_list = zip(*batch)
    max_n    = max(t.size(0) for t in token_list)
    token_dim = token_list[0].size(1)
    B        = len(token_list)

    tokens_padded = torch.zeros(B, max_n, token_dim)
    mask          = torch.ones(B, max_n, dtype=torch.bool)   # True = padding

    for i, t in enumerate(token_list):
        n = t.size(0)
        tokens_padded[i, :n] = t
        mask[i, :n]          = False

    redshifts = torch.stack(redshift_list)
    return tokens_padded, mask, redshifts
