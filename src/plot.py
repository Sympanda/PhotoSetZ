"""
Plotting and post-training diagnostics for DeepSetZ.

Generates all figures into <out_dir>/plots/:

    training_curves.png     — loss, σ_NMAD, bias, outlier rate vs epoch
    scatter.png             — z_phot vs z_spec (full test set)
    delta_z.png             — Δz = (z_phot−z_spec)/(1+z_spec) distribution
    survey_metrics.png      — σ_NMAD / outlier rate for every survey combination
    survey_metrics.csv      — same data as a CSV table

Survey combinations evaluated
------------------------------
All 2^N − 1 non-empty subsets of the active surveys are tested
(4 surveys → 15 combinations), plus every named SURVEY_PRESET that
is a subset of the active filters.  Combinations are ordered by the
total number of filters (fewest first).
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Matplotlib in non-interactive mode (safe for MPS / headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .dataset import GalaxyDataset, collate_fn, SURVEY_PRESETS
from .dataloader_utils import resolve_num_workers
from .evaluate import compute_metrics, OUTLIER_THRESHOLD


# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------
_SURVEY_COLOURS = {
    "lsst":   "#4C8BE2",
    "roman":  "#E2734C",
    "euclid": "#4CE27A",
    "wise":   "#C34CE2",
}
_COMBO_CMAP = plt.cm.viridis


# ------------------------------------------------------------------
# 1. Training curves
# ------------------------------------------------------------------

def plot_training_curves(history: List[dict], out_dir: Path) -> None:
    epochs        = [r["epoch"]                  for r in history]
    train_loss    = [r.get("train_loss",   np.nan) for r in history]
    val_loss      = [r.get("val_loss",     np.nan) for r in history]
    val_drop_loss = [r.get("val_drop_loss",np.nan) for r in history]
    sigma_nmad    = [r.get("val_sigma_nmad", np.nan) for r in history]
    bias          = [r.get("val_bias",     np.nan) for r in history]
    outlier       = [r.get("val_outlier_rate", np.nan) for r in history]
    lr            = [r.get("lr",           np.nan) for r in history]

    has_drop = any(not np.isnan(v) for v in val_drop_loss)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training curves", fontsize=14, fontweight="bold")

    def _plot(ax, y_train, y_val, label, ylabel, logy=False, y_val2=None, val2_label=None):
        ax.plot(epochs, y_train, label="train", color="#4C8BE2", lw=1.5)
        if y_val is not None:
            ax.plot(epochs, y_val, label="val (clean)", color="#E2734C", lw=1.5)
        if y_val2 is not None:
            ax.plot(epochs, y_val2, label=val2_label or "val (drop)",
                    color="#8B4CE2", lw=1.2, ls="--", alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], train_loss, val_loss, "Loss", "Loss", logy=True,
          y_val2=val_drop_loss if has_drop else None, val2_label="val (dropout)")
    _plot(axes[0, 1], sigma_nmad, None,     "σ_NMAD",       "σ_NMAD")
    axes[0, 1].plot(epochs, sigma_nmad, color="#E2734C", lw=1.5)
    _plot(axes[0, 2], bias,       None,     "Bias",          "median(Δz)")
    axes[0, 2].axhline(0, color="k", lw=0.8, ls="--")
    _plot(axes[1, 0], outlier,    None,     "Outlier rate",  "Fraction |Δz|>0.15")
    _plot(axes[1, 1], lr,         None,     "Learning rate", "LR", logy=True)

    # Blank last panel
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# ------------------------------------------------------------------
# 2. Scatter plot: z_phot vs z_spec
# ------------------------------------------------------------------

def plot_scatter(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    out_dir: Path,
    title: str = "DeepSetZ — full test set",
    filename: str = "scatter.png",
) -> None:
    delta_z = (z_pred - z_true) / (1.0 + z_true)
    sigma   = 1.4826 * np.median(np.abs(delta_z - np.median(delta_z)))
    outlier = (np.abs(delta_z) > OUTLIER_THRESHOLD).mean()
    bias    = np.median(delta_z)

    zmax = max(z_true.max(), z_pred.max()) * 1.05

    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hexbin(z_true, z_pred, gridsize=80, cmap="YlOrRd",
                  mincnt=1, bins="log")
    fig.colorbar(h, ax=ax, label="log10(count)")
    ax.plot([0, zmax], [0, zmax], "k--", lw=1.0, label="1:1")
    z_line = np.array([0.0, zmax])
    ax.plot(z_line, z_line + OUTLIER_THRESHOLD * (1 + z_line), "k:", lw=0.7, alpha=0.5)
    ax.plot(z_line, z_line - OUTLIER_THRESHOLD * (1 + z_line), "k:", lw=0.7, alpha=0.5)
    ax.set_xlim(0, zmax)
    ax.set_ylim(0, zmax)
    ax.set_xlabel("$z_{\\rm spec}$", fontsize=12)
    ax.set_ylabel("$z_{\\rm phot}$", fontsize=12)
    ax.set_title(title, fontsize=11)
    stats = (f"N={len(z_true):,}\n"
             f"σ_NMAD={sigma:.4f}\n"
             f"bias={bias:+.4f}\n"
             f"outlier={outlier*100:.1f}%")
    ax.text(0.04, 0.96, stats, transform=ax.transAxes,
            va="top", fontsize=9, family="monospace",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# ------------------------------------------------------------------
# 3. Δz distribution histogram
# ------------------------------------------------------------------

def plot_delta_z(
    z_true: np.ndarray,
    z_pred: np.ndarray,
    out_dir: Path,
    title: str = "Δz distribution",
    filename: str = "delta_z.png",
) -> None:
    delta_z = (z_pred - z_true) / (1.0 + z_true)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(delta_z, bins=120, range=(-0.5, 0.5),
            color="#4C8BE2", edgecolor="none", alpha=0.8, density=True)
    ax.axvline(0,  color="k",   lw=1.0, ls="--", label="zero")
    ax.axvline(np.median(delta_z), color="#E2734C", lw=1.5,
               label=f"bias={np.median(delta_z):+.4f}")
    ax.axvline( OUTLIER_THRESHOLD, color="gray", lw=0.8, ls=":")
    ax.axvline(-OUTLIER_THRESHOLD, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("$\\Delta z = (z_{\\rm phot} - z_{\\rm spec}) / (1 + z_{\\rm spec})$",
                  fontsize=11)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# ------------------------------------------------------------------
# 4. Survey-subset evaluation + plots
# ------------------------------------------------------------------

def _all_survey_combos(
    active_survey_names: List[str],
    survey_to_cols: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Generate all non-empty subsets of the active survey groups,
    returning a dict {label: [col_names]}.

    Parameters
    ----------
    active_survey_names : list of survey names present in the registry
    survey_to_cols : mapping from survey name → list of col_names
                     (derived from the live filter registry)
    """
    combos: Dict[str, List[str]] = {}
    for r in range(1, len(active_survey_names) + 1):
        for surveys in itertools.combinations(active_survey_names, r):
            cols = []
            for s in surveys:
                cols.extend(survey_to_cols.get(s, []))
            label = "+".join(surveys)
            combos[label] = cols
    return combos


def evaluate_survey_subsets(
    model,
    test_ds: GalaxyDataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
    log_target: bool = False,
) -> Dict[str, dict]:
    """
    Evaluate model metrics for every survey combination and every named preset
    that is a subset of the active filters.

    For each combination, the full 16-token batch is loaded and non-subset
    tokens are masked out — the model sees only the relevant filters.

    Returns a dict  {combo_label: metrics_dict}.
    """
    model.eval()
    active_cols  = set(fi.col_name for fi in test_ds.filters)
    col_to_idx   = {fi.col_name: i for i, fi in enumerate(test_ds.filters)}

    # Build survey → col_names mapping from the live registry
    from collections import defaultdict as _dd
    _survey_to_cols: Dict[str, List[str]] = _dd(list)
    for fi in test_ds.filters:
        _survey_to_cols[fi.survey].append(fi.col_name)
    survey_to_cols = dict(_survey_to_cols)
    active_surv    = sorted(survey_to_cols.keys())

    # Build survey combinations
    combos = _all_survey_combos(active_surv, survey_to_cols)

    # Add named presets that are fully covered by active filters
    for pname, cols in SURVEY_PRESETS.items():
        valid = [c for c in cols if c in active_cols]
        if len(valid) >= 1:
            label = f"preset:{pname}({len(valid)}f)"
            combos[label] = valid

    # Full-set entry
    combos["all (full set)"] = list(active_cols)

    # Build a DataLoader over the test set (no dropout)
    loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = resolve_num_workers(num_workers),
    )

    # Precompute: for each combo, which token indices to keep
    combo_indices: Dict[str, List[int]] = {}
    for label, cols in combos.items():
        idx = sorted([col_to_idx[c] for c in cols if c in col_to_idx])
        if idx:
            combo_indices[label] = idx

    results: Dict[str, dict] = {}

    with torch.no_grad():
        # Collect full-batch data once
        all_tokens, all_masks, all_z = [], [], []
        for tokens, mask, z in loader:
            all_tokens.append(tokens)
            all_masks.append(mask)
            all_z.append(z)
        all_tokens = torch.cat(all_tokens, dim=0)   # (N, max_N, D)
        all_masks  = torch.cat(all_masks,  dim=0)   # (N, max_N)
        all_z      = torch.cat(all_z,      dim=0)   # (N,)

        # z_true in training space; convert to real z for metrics
        z_true_np = all_z.numpy()
        if log_target:
            z_true_np = np.expm1(z_true_np)

        for label, indices in combo_indices.items():
            # Build a mask that keeps only the subset tokens
            sub_mask = torch.ones_like(all_masks)   # True = padding
            sub_mask[:, indices] = all_masks[:, indices]  # restore valid flags

            # Run in mini-batches to avoid OOM
            z_preds = []
            for start in range(0, len(all_tokens), batch_size):
                t = all_tokens[start:start+batch_size].to(device)
                m = sub_mask[start:start+batch_size].to(device)
                out = model(t, m)
                z_preds.append(out["z_pred"].cpu())

            z_pred_np = torch.cat(z_preds).numpy()
            if log_target:
                z_pred_np = np.expm1(z_pred_np)
            metrics = compute_metrics(z_pred_np, z_true_np)
            metrics["n_filters"] = len(indices)
            results[label] = metrics

    return results


def plot_survey_metrics(
    subset_results: Dict[str, dict],
    out_dir: Path,
) -> None:
    """
    Bar charts of σ_NMAD and outlier rate for every evaluated subset,
    grouped by number of filters and coloured by that count.
    """
    # Sort by n_filters then label
    items = sorted(subset_results.items(), key=lambda x: (x[1]["n_filters"], x[0]))
    labels      = [k for k, _ in items]
    sigma_nmads = [v["sigma_nmad"]    for _, v in items]
    outliers    = [v["outlier_rate"]  for _, v in items]
    biases      = [v["bias"]          for _, v in items]
    n_filters   = [v["n_filters"]     for _, v in items]

    # Colour by n_filters
    max_f  = max(n_filters) if n_filters else 1
    colors = [_COMBO_CMAP(nf / max_f) for nf in n_filters]

    n = len(labels)
    rmses = [v["rmse"] for _, v in items]

    fig, axes = plt.subplots(4, 1, figsize=(max(12, n * 0.55), 16),
                             constrained_layout=True)
    fig.suptitle("Metrics by survey subset", fontsize=13, fontweight="bold")

    for ax, vals, ylabel, title in [
        (axes[0], sigma_nmads, "σ_NMAD",           "Scatter (σ_NMAD)"),
        (axes[1], outliers,    "Outlier rate",      f"Outlier rate (|Δz|>{OUTLIER_THRESHOLD})"),
        (axes[2], biases,      "Bias median(Δz)",   "Bias"),
        (axes[3], rmses,       "RMSE(Δz)",          "RMSE"),
    ]:
        bars = ax.bar(range(n), vals, color=colors, edgecolor="none", width=0.7)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        if "Bias" in title:
            ax.axhline(0, color="k", lw=0.8, ls="--")

        # Annotate bar tops
        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6, rotation=90)

    # Colorbar to show n_filters scale
    sm = plt.cm.ScalarMappable(cmap=_COMBO_CMAP,
                               norm=mcolors.Normalize(vmin=1, vmax=max_f))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="horizontal", location="bottom",
                 label="Number of filters", shrink=0.5, pad=0.02, aspect=40)

    out_path = out_dir / "survey_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")

    # Also write CSV
    csv_path = out_dir / "survey_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["subset", "n_filters", "sigma_nmad", "bias",
                        "outlier_rate", "rmse", "mae"],
        )
        writer.writeheader()
        for label, metrics in subset_results.items():
            writer.writerow({
                "subset":       label,
                "n_filters":    metrics["n_filters"],
                "sigma_nmad":   f"{metrics['sigma_nmad']:.5f}",
                "bias":         f"{metrics['bias']:+.5f}",
                "outlier_rate": f"{metrics['outlier_rate']:.5f}",
                "rmse":         f"{metrics['rmse']:.5f}",
                "mae":          f"{metrics['mae']:.5f}",
            })
    print(f"  Saved {csv_path.name}")


# ------------------------------------------------------------------
# 5. Dual scatter plot: mean vs median (probabilistic heads)
# ------------------------------------------------------------------

def plot_scatter_dual(
    z_true: np.ndarray,
    z_mean: np.ndarray,
    z_median: np.ndarray,
    out_dir: Path,
    title: str = "",
    filename: str = "scatter_mean_median.png",
) -> None:
    """Side-by-side scatter plots for the mean and median point estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, z_pred, est_label in [
        (axes[0], z_mean,   "Mean"),
        (axes[1], z_median, "Median"),
    ]:
        delta_z = (z_pred - z_true) / (1.0 + z_true)
        sigma   = 1.4826 * np.median(np.abs(delta_z - np.median(delta_z)))
        outlier = (np.abs(delta_z) > OUTLIER_THRESHOLD).mean()
        bias    = np.median(delta_z)
        zmax    = max(z_true.max(), z_pred.max()) * 1.05

        h = ax.hexbin(z_true, z_pred, gridsize=80, cmap="YlOrRd", mincnt=1, bins="log")
        fig.colorbar(h, ax=ax, label="log10(count)")
        z_line = np.array([0.0, zmax])
        ax.plot(z_line, z_line, "k--", lw=1.0, label="1:1")
        ax.plot(z_line, z_line + OUTLIER_THRESHOLD * (1 + z_line), "k:", lw=0.7, alpha=0.5)
        ax.plot(z_line, z_line - OUTLIER_THRESHOLD * (1 + z_line), "k:", lw=0.7, alpha=0.5)
        ax.set_xlim(0, zmax); ax.set_ylim(0, zmax)
        ax.set_xlabel("$z_{\\rm spec}$", fontsize=11)
        ax.set_ylabel(f"$z_{{\\rm phot}}$ ({est_label})", fontsize=11)
        ax.set_title(f"Point estimate: {est_label}", fontsize=10)
        stats = (f"N={len(z_true):,}\n"
                 f"σ_NMAD={sigma:.4f}\n"
                 f"bias={bias:+.4f}\n"
                 f"outlier={outlier*100:.1f}%")
        ax.text(0.04, 0.96, stats, transform=ax.transAxes,
                va="top", fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# ------------------------------------------------------------------
# 6. Calibration: PIT histogram + coverage / reliability diagram
# ------------------------------------------------------------------

def _compute_coverage(pit: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """
    Coverage fraction for each equal-tails credible interval level α.

    For galaxy i, define coverage(α) = 1 iff
        PIT_i ∈ [(1−α)/2, (1+α)/2]
    This is equivalent to checking that z_true falls inside the α-CI
    defined by the equal-tails quantiles of the predicted distribution.
    """
    coverage = np.array([
        np.mean((pit >= (1 - a) / 2) & (pit <= (1 + a) / 2))
        for a in alphas
    ])
    return coverage


def plot_calibration(
    pit_values: np.ndarray,
    out_dir: Path,
    title: str = "Calibration",
    filename: str = "calibration.png",
) -> None:
    """
    Combined calibration diagnostic: PIT histogram + coverage/reliability diagram.

    PIT (Probability Integral Transform)
    -------------------------------------
    For each galaxy, PIT = CDF_predicted(z_true).  If the model is perfectly
    calibrated the PITs are Uniform[0, 1].  Excess probability near 0 or 1
    indicates under-dispersion (overconfident); excess probability near 0.5
    indicates over-dispersion (underconfident).

    Coverage / reliability diagram
    --------------------------------
    For a range of confidence levels α the fraction of galaxies whose z_true
    falls inside the predicted equal-tails α-credible interval is plotted.
    A diagonal line represents perfect calibration.  Points below the diagonal
    mean the model is overconfident; above means underconfident.
    """
    pit = np.asarray(pit_values, dtype=float)
    pit = pit[np.isfinite(pit)]  # guard against any NaN PIT values

    alphas   = np.linspace(0.0, 1.0, 101)
    coverage = _compute_coverage(pit, alphas)

    fig, (ax_pit, ax_cov) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # ── PIT histogram ──────────────────────────────────────────────
    ax_pit.hist(pit, bins=25, range=(0, 1), density=True,
                color="#4C8BE2", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax_pit.axhline(1.0, color="k", lw=1.5, ls="--", label="Uniform (ideal)")
    ax_pit.set_xlim(0, 1)
    ax_pit.set_xlabel("PIT value", fontsize=11)
    ax_pit.set_ylabel("Density", fontsize=11)
    ax_pit.set_title("PIT Histogram", fontsize=11)
    ax_pit.legend(fontsize=9)
    ax_pit.grid(True, axis="y", alpha=0.3)

    # Annotate departure from uniform
    ks_stat = float(np.max(np.abs(
        np.sort(pit) - np.linspace(0, 1, len(pit))
    )))
    ax_pit.text(0.97, 0.97, f"KS={ks_stat:.3f}",
                transform=ax_pit.transAxes, ha="right", va="top",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    # ── Coverage / reliability diagram ─────────────────────────────
    # Shaded region shows over-/under-confidence
    ax_cov.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=3)
    ax_cov.fill_between(alphas, alphas, coverage,
                        where=(coverage > alphas),
                        alpha=0.25, color="#4CE27A", label="Over-dispersed")
    ax_cov.fill_between(alphas, alphas, coverage,
                        where=(coverage < alphas),
                        alpha=0.25, color="#E2734C", label="Under-dispersed")
    ax_cov.plot(alphas, coverage, color="#4C8BE2", lw=2.0, label="Model", zorder=4)
    ax_cov.set_xlim(0, 1); ax_cov.set_ylim(0, 1)
    ax_cov.set_xlabel("Confidence level α", fontsize=11)
    ax_cov.set_ylabel("Coverage fraction", fontsize=11)
    ax_cov.set_title("Coverage / Reliability Diagram", fontsize=11)
    ax_cov.legend(fontsize=9, loc="upper left")
    ax_cov.grid(True, alpha=0.3)

    # Annotate mean absolute calibration error
    mace = float(np.mean(np.abs(coverage - alphas)))
    ax_cov.text(0.97, 0.03, f"MACE={mace:.3f}",
                transform=ax_cov.transAxes, ha="right", va="bottom",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


def plot_calibration_comparison(
    pit_before: np.ndarray,
    pit_after: np.ndarray,
    out_dir: Path,
    title: str = "Calibration — before vs after post-hoc σ",
    filename: str = "calibration_comparison.png",
    sigma_scale: float | None = None,
) -> None:
    """Side-by-side PIT + coverage for raw and post-hoc-scaled posteriors."""
    pit_b = np.asarray(pit_before, dtype=float)
    pit_a = np.asarray(pit_after, dtype=float)
    pit_b = pit_b[np.isfinite(pit_b)]
    pit_a = pit_a[np.isfinite(pit_a)]

    alphas = np.linspace(0.0, 1.0, 101)
    cov_b  = _compute_coverage(pit_b, alphas)
    cov_a  = _compute_coverage(pit_a, alphas)
    mace_b = float(np.mean(np.abs(cov_b - alphas)))
    mace_a = float(np.mean(np.abs(cov_a - alphas)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    scale_note = f"  (σ × {sigma_scale:.3f})" if sigma_scale is not None else ""
    fig.suptitle(f"{title}{scale_note}", fontsize=12, fontweight="bold")

    for ax, pit, label in (
        (axes[0, 0], pit_b, "Before (raw)"),
        (axes[0, 1], pit_a, "After (post-hoc)"),
    ):
        ax.hist(pit, bins=25, range=(0, 1), density=True,
                color="#4C8BE2", edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axhline(1.0, color="k", lw=1.5, ls="--")
        ax.set_xlim(0, 1)
        ax.set_title(f"PIT — {label}", fontsize=10)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.grid(True, axis="y", alpha=0.3)

    for ax, cov, mace, label in (
        (axes[1, 0], cov_b, mace_b, "Before (raw)"),
        (axes[1, 1], cov_a, mace_a, "After (post-hoc)"),
    ):
        ax.plot([0, 1], [0, 1], "k--", lw=1.5)
        ax.fill_between(alphas, alphas, cov,
                        where=(cov > alphas), alpha=0.25, color="#4CE27A")
        ax.fill_between(alphas, alphas, cov,
                        where=(cov < alphas), alpha=0.25, color="#E2734C")
        ax.plot(alphas, cov, color="#4C8BE2", lw=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Coverage — {label}", fontsize=10)
        ax.set_xlabel("Confidence level α")
        ax.set_ylabel("Coverage fraction")
        ax.text(0.97, 0.03, f"MACE={mace:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path.name}")


# ------------------------------------------------------------------
# Master entry point
# ------------------------------------------------------------------

def generate_all_plots(
    history: List[dict],
    z_true: np.ndarray,
    z_pred: np.ndarray,
    subset_results: Dict[str, dict],
    out_dir: Path,
    run_name: str = "",
    prob_outputs: dict | None = None,
    plots_subdir: str | None = None,
    stage_label: str = "",
) -> Path:
    """
    Generate all post-training diagnostic plots.

    Parameters
    ----------
    prob_outputs : optional dict with keys z_mean, z_median, z_mode, pit, head_type.
                   When provided (MDN / BinnedPDF / NSF heads), also saves
                   scatter_mean_median.png and calibration.png.
    plots_subdir : optional subdirectory under plots/ (e.g. "stage1", "stage2").
    stage_label  : prefix for plot titles (e.g. "Stage 1 — Point (MLP)").

    Returns
    -------
    Path to the directory where plots were written.
    """
    plots_dir = out_dir / "plots"
    if plots_subdir:
        plots_dir = plots_dir / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots → {plots_dir}")

    prefix = f"{stage_label} — " if stage_label else ""
    title = (
        f"DeepSetZ — {run_name} — {prefix}full test set"
        if run_name else f"DeepSetZ — {prefix}full test set"
    )

    plot_training_curves(history,                    plots_dir)
    plot_scatter(z_true, z_pred,                     plots_dir, title=title)
    plot_delta_z(z_true, z_pred,                     plots_dir, title=f"Δz — {run_name}")
    plot_survey_metrics(subset_results,              plots_dir)

    if prob_outputs is not None:
        head_type = prob_outputs.get("head_type", "")
        z_mean    = prob_outputs["z_mean"]
        z_median  = prob_outputs["z_median"]
        pit       = prob_outputs["pit"]

        plot_scatter_dual(
            z_true, z_mean, z_median, plots_dir,
            title=f"{title} — Mean / Median",
        )
        plot_calibration(pit, plots_dir, title=f"Calibration — {run_name} ({head_type})")

        # Individual scatter plots for each point estimate
        plot_scatter(z_true, z_mean, plots_dir,
                     title=f"{title} (mean)",
                     filename="scatter_mean.png")
        plot_scatter(z_true, z_median, plots_dir,
                     title=f"{title} (median)",
                     filename="scatter_median.png")

        # Delta-z for mean and median
        plot_delta_z(z_true, z_mean, plots_dir,
                     title=f"Δz (mean) — {run_name}",
                     filename="delta_z_mean.png")
        plot_delta_z(z_true, z_median, plots_dir,
                     title=f"Δz (median) — {run_name}",
                     filename="delta_z_median.png")

    print("  Done.\n")
    return plots_dir
