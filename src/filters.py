"""
Filter registry for the DeepSetZ photometric redshift framework.

Reads wavelength-transmission .res files where available and computes
effective wavelength (λ_eff) and RMS bandwidth (Δλ) for each filter.
Filters without .res files use hardcoded approximate values from the
literature / instrument documentation.

Two naming conventions are supported out-of-the-box:

  Ellen / noiseless mocks  — "Roman_Y106", "mag_u_lsst", …
  DESC PZ Data Challenge   — "mag_F106_roman", "mag_Y_roman",
                              "mag_u_lsst", …  (same LSST names)

Pass ``available_cols`` (the set of columns in your parquet file) to
``build_filter_registry`` and only the matching entries will be returned.
This makes dataset swapping automatic — no config change required beyond
pointing to the new data paths.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class FilterInfo:
    name: str               # unique key, same as col_name
    col_name: str           # column name in the parquet file
    err_col_name: str       # error column name (empty string if unavailable)
    survey: str             # 'lsst' | 'roman' | 'euclid' | 'wise'
    survey_id: int          # integer survey index used as a token feature
    lambda_eff: float       # effective wavelength in Angstrom
    delta_lambda: float     # RMS bandwidth in Angstrom


SURVEY_IDS: Dict[str, int] = {
    "lsst":   0,
    "roman":  1,
    "euclid": 2,
    "wise":   3,
}

# ------------------------------------------------------------------
# Hardcoded filter metadata
# ------------------------------------------------------------------
# Ellen / noiseless-mock column names
_HARDCODED_ELLEN: List[Dict] = [
    dict(name="Roman_F184", col_name="Roman_F184", survey="roman",  lambda_eff=18400.0, delta_lambda=2200.0),
    dict(name="Roman_K213", col_name="Roman_K213", survey="roman",  lambda_eff=21300.0, delta_lambda=2600.0),
    dict(name="Euclid_Y",   col_name="Euclid_Y",   survey="euclid", lambda_eff=10640.0, delta_lambda=2100.0),
    dict(name="Euclid_J",   col_name="Euclid_J",   survey="euclid", lambda_eff=13690.0, delta_lambda=3160.0),
    dict(name="Euclid_H",   col_name="Euclid_H",   survey="euclid", lambda_eff=17700.0, delta_lambda=3960.0),
    dict(name="WISE_W1",    col_name="WISE_W1",     survey="wise",   lambda_eff=33526.0, delta_lambda=6626.0),
    dict(name="WISE_W2",    col_name="WISE_W2",     survey="wise",   lambda_eff=46028.0, delta_lambda=10097.0),
]

# DESC PZ Data Challenge column names for Roman bands.
# Two naming conventions are both added; whichever is found in the parquet is used.
# λ_eff / Δλ values match the Roman HLWAS filter set (Akeson et al. 2019 + Roman docs).
_HARDCODED_PZDC: List[Dict] = [
    # Letter-band names  (mag_Y_roman, mag_J_roman, …)
    dict(name="mag_Y_roman",  col_name="mag_Y_roman",  survey="roman", lambda_eff=10595.0, delta_lambda=1820.0),
    dict(name="mag_J_roman",  col_name="mag_J_roman",  survey="roman", lambda_eff=12930.0, delta_lambda=1760.0),
    dict(name="mag_H_roman",  col_name="mag_H_roman",  survey="roman", lambda_eff=15800.0, delta_lambda=2820.0),
    dict(name="mag_F_roman",  col_name="mag_F_roman",  survey="roman", lambda_eff=18400.0, delta_lambda=2200.0),
    dict(name="mag_K_roman",  col_name="mag_K_roman",  survey="roman", lambda_eff=21300.0, delta_lambda=2600.0),
    # Filter-number names (mag_F106_roman, mag_F129_roman, …)
    dict(name="mag_F062_roman", col_name="mag_F062_roman", survey="roman", lambda_eff= 6250.0, delta_lambda= 800.0),
    dict(name="mag_F087_roman", col_name="mag_F087_roman", survey="roman", lambda_eff= 8730.0, delta_lambda= 750.0),
    dict(name="mag_F106_roman", col_name="mag_F106_roman", survey="roman", lambda_eff=10595.0, delta_lambda=1820.0),
    dict(name="mag_F129_roman", col_name="mag_F129_roman", survey="roman", lambda_eff=12930.0, delta_lambda=1760.0),
    dict(name="mag_F158_roman", col_name="mag_F158_roman", survey="roman", lambda_eff=15800.0, delta_lambda=2820.0),
    dict(name="mag_F184_roman", col_name="mag_F184_roman", survey="roman", lambda_eff=18400.0, delta_lambda=2200.0),
    dict(name="mag_F213_roman", col_name="mag_F213_roman", survey="roman", lambda_eff=21300.0, delta_lambda=2600.0),
    # PZDC LSST columns are identical to Ellen, so they come from _RES_MAP below.
    # Euclid / WISE PZDC variants (if the challenge includes them)
    dict(name="mag_Y_euclid",   col_name="mag_Y_euclid",   survey="euclid", lambda_eff=10640.0, delta_lambda=2100.0),
    dict(name="mag_J_euclid",   col_name="mag_J_euclid",   survey="euclid", lambda_eff=13690.0, delta_lambda=3160.0),
    dict(name="mag_H_euclid",   col_name="mag_H_euclid",   survey="euclid", lambda_eff=17700.0, delta_lambda=3960.0),
]

_HARDCODED = _HARDCODED_ELLEN + _HARDCODED_PZDC

# Mapping from .res file stem → (survey, parquet column name [Ellen convention])
_RES_MAP: Dict[str, tuple] = {
    "DC2LSST_u":  ("lsst",  "mag_u_lsst"),
    "DC2LSST_g":  ("lsst",  "mag_g_lsst"),
    "DC2LSST_r":  ("lsst",  "mag_r_lsst"),
    "DC2LSST_i":  ("lsst",  "mag_i_lsst"),
    "DC2LSST_z":  ("lsst",  "mag_z_lsst"),
    "DC2LSST_y":  ("lsst",  "mag_y_lsst"),
    "roman_Y106": ("roman", "Roman_Y106"),
    "roman_J129": ("roman", "Roman_J129"),
    "roman_H158": ("roman", "Roman_H158"),
}


def _load_res(path: Path) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read a two-column wavelength-transmission .res file."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def _filter_stats(wavelengths: np.ndarray, transmission: np.ndarray) -> tuple[float, float]:
    """
    Compute effective wavelength and RMS bandwidth from a transmission curve.

        λ_eff  = ∫ λ T(λ) dλ  /  ∫ T(λ) dλ
        Δλ     = sqrt( ∫ (λ - λ_eff)² T(λ) dλ  /  ∫ T(λ) dλ )
    """
    T = np.maximum(transmission, 0.0)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]
    norm = _trapz(T, wavelengths)
    if norm == 0.0:
        return float(wavelengths.mean()), float(wavelengths.std())
    lambda_eff = _trapz(wavelengths * T, wavelengths) / norm
    variance   = _trapz((wavelengths - lambda_eff) ** 2 * T, wavelengths) / norm
    return float(lambda_eff), float(np.sqrt(variance))


def build_filter_registry(
    res_dir: str | Path,
    active_surveys: List[str] | None = None,
    available_cols: Set[str] | None = None,
) -> Dict[str, FilterInfo]:
    """
    Build the complete filter registry.

    Parameters
    ----------
    res_dir : path
        Directory containing .res filter transmission files.
    active_surveys : list[str] or None
        Optional allowlist.  Each entry may be:
          - a survey name:  "lsst" | "roman" | "euclid" | "wise"
          - a column name:  e.g. "mag_g_lsst", "Roman_Y106"
        Entries are matched case-insensitively.  Empty list or None → all.
    available_cols : set[str] or None
        If provided (typically ``set(df.columns)``), only include filters
        whose col_name actually exists in the dataset.  This allows the same
        registry to automatically adapt to Ellen vs PZDC parquet files.

    Returns a dict keyed by col_name, sorted by λ_eff.
    """
    res_dir = Path(res_dir)
    registry: Dict[str, FilterInfo] = {}

    # Entries derived from .res files (Ellen naming for LSST + Roman Y/J/H)
    for stem, (survey, col_name) in _RES_MAP.items():
        path = res_dir / f"{stem}.res"
        if not path.exists():
            continue
        result = _load_res(path)
        if result is None:
            continue
        wl, tr   = result
        lam_eff, dlam = _filter_stats(wl, tr)
        registry[col_name] = FilterInfo(
            name=col_name,
            col_name=col_name,
            err_col_name=f"{col_name}_err",
            survey=survey,
            survey_id=SURVEY_IDS[survey],
            lambda_eff=lam_eff,
            delta_lambda=dlam,
        )

    # Hardcoded entries (Ellen and PZDC variants)
    for h in _HARDCODED:
        col = h["col_name"]
        registry[col] = FilterInfo(
            name=col,
            col_name=col,
            err_col_name=f"{col}_err",
            survey=h["survey"],
            survey_id=SURVEY_IDS[h["survey"]],
            lambda_eff=h["lambda_eff"],
            delta_lambda=h["delta_lambda"],
        )

    # Sort by effective wavelength (bluest first)
    registry = dict(sorted(registry.items(), key=lambda kv: kv[1].lambda_eff))

    # Filter to columns that actually exist in the dataset (auto-detect naming convention)
    if available_cols is not None:
        registry = {k: v for k, v in registry.items() if v.col_name in available_cols}

    # Apply active_surveys allowlist
    if active_surveys:
        allowed = {s.lower() for s in active_surveys}
        registry = {
            k: v for k, v in registry.items()
            if v.survey.lower() in allowed or v.col_name.lower() in allowed
        }

    return registry


def print_registry(registry: Dict[str, FilterInfo]) -> None:
    """Pretty-print the filter registry."""
    print(f"{'Filter':<22} {'Survey':<8} {'λ_eff (Å)':<12} {'Δλ (Å)':<10} {'Col':<22} {'Err col'}")
    print("-" * 90)
    for fi in registry.values():
        print(f"{fi.name:<22} {fi.survey:<8} {fi.lambda_eff:<12.1f} {fi.delta_lambda:<10.1f} "
              f"{fi.col_name:<22} {fi.err_col_name or '—'}")


if __name__ == "__main__":
    import sys
    res_dir = sys.argv[1] if len(sys.argv) > 1 else "data/ellen"
    reg = build_filter_registry(res_dir)
    print_registry(reg)
