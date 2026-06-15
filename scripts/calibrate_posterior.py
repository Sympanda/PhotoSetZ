#!/usr/bin/env python3
"""
Fit post-hoc σ scaling on an existing run and write calibration/post_hoc.json.

Also evaluates calibration on the held-out test set and writes diagnostic plots
under plots/ (calibration_post_hoc.png, calibration_comparison.png).

Does not create a new output directory — artefacts live under the existing run.

Usage
-----
    python scripts/calibrate_posterior.py outputs/ts1_10yr_08_st
    python scripts/calibrate_posterior.py outputs/ts1_10yr_08_st --role posterior
    python scripts/calibrate_posterior.py outputs/ts1_10yr_08_st --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.platform_fix  # noqa: F401  — before torch on macOS

import torch
from torch.utils.data import DataLoader, Subset

from src.calibration import compute_mace, run_post_hoc_calibration
from src.config import load_config
from src.dataset import GalaxyDataset, collate_fn
from src.dataloader_utils import resolve_num_workers, shutdown_dataloaders
from src.evaluate import compute_metrics, compute_pit_metrics
from src.plot import plot_calibration, plot_calibration_comparison
from src.run_artifacts import (
    ROLE_END_TO_END,
    ROLE_POSTERIOR,
    checkpoint_path,
    load_post_hoc,
)
from src.train import build_model, collect_prob_outputs, get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc posterior σ calibration")
    p.add_argument("run_dir", help="Path to outputs/<run_name>")
    p.add_argument(
        "--role", default="auto",
        choices=["auto", ROLE_POSTERIOR, ROLE_END_TO_END],
        help="Which checkpoint to calibrate (default: auto-detect)",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip test-set plots and test_metrics_post_hoc.json",
    )
    return p.parse_args()


def _resolve_role(run_dir: Path, role: str) -> str:
    if role != "auto":
        return role
    if checkpoint_path(run_dir, ROLE_POSTERIOR):
        return ROLE_POSTERIOR
    return ROLE_END_TO_END


def _evaluate_test_set(
    model,
    cfg,
    device,
    run_dir: Path,
    sigma_scale: float,
    *,
    write_plots: bool,
) -> dict:
    """Collect raw vs scaled posteriors on the test split; save metrics + plots."""
    test_path = ROOT / cfg.data.test_path
    res_dir   = ROOT / cfg.data.res_dir
    test_ds = GalaxyDataset(
        parquet_path=test_path,
        res_dir=res_dir,
        target_col=cfg.data.target_col,
        dropout_cfg=None,
        active_surveys=cfg.data.active_surveys or None,
        log_target=cfg.data.log_target,
        include_errors=cfg.data.include_errors,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=resolve_num_workers(cfg.training.num_workers),
    )

    try:
        raw  = collect_prob_outputs(
            model, test_loader, device, log_target=cfg.data.log_target, sigma_scale=1.0,
        )
        if raw is None:
            print("  [test] Skipped — head is not probabilistic.")
            return {}

        scaled = collect_prob_outputs(
            model, test_loader, device, log_target=cfg.data.log_target,
            sigma_scale=sigma_scale,
        )
        z_true = raw["z_true"]

        def _calib_block(pit):
            pit_m = compute_pit_metrics(pit)
            pit_m["mace"] = compute_mace(pit)
            return pit_m

        cal_before = _calib_block(raw["pit"])
        cal_after  = _calib_block(scaled["pit"])

        point = {}
        for est in ("z_mean", "z_median", "z_mode"):
            m_raw = compute_metrics(raw[est], z_true)
            m_scl = compute_metrics(scaled[est], z_true)
            point[est] = {
                "before": m_raw,
                "after":  m_scl,
            }

        payload = {
            "sigma_scale":   sigma_scale,
            "head_type":     raw["head_type"],
            "evaluated_on":  "test",
            "n_test":        len(z_true),
            "calibration": {
                "before": cal_before,
                "after":  cal_after,
            },
            "point_metrics": point,
        }

        out_path = run_dir / "test_metrics_post_hoc.json"
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  [test] Saved {out_path.name}")

        print(f"\n  Test-set calibration (σ × {sigma_scale:.3f}):")
        print(f"    MACE  {cal_before['mace']:.4f} → {cal_after['mace']:.4f}")
        print(f"    KS    {cal_before['pit_ks']:.4f} → {cal_after['pit_ks']:.4f}")
        print(f"    PIT RMSE {cal_before['pit_rmse']:.4f} → {cal_after['pit_rmse']:.4f}")

        print(f"\n  Test-set point metrics (unchanged — only σ is scaled):")
        m = point["z_mean"]["before"]
        print(f"    z_mean  σ_NMAD={m['sigma_nmad']:.4f}  bias={m['bias']:+.4f}  "
              f"outlier={m['outlier_rate']*100:.1f}%")

        if write_plots:
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            run_label = run_dir.name
            head      = raw["head_type"]
            print(f"\n  Writing plots → {plots_dir}")
            plot_calibration(
                scaled["pit"], plots_dir,
                title=f"Calibration (post-hoc) — {run_label} ({head})",
                filename="calibration_post_hoc.png",
            )
            plot_calibration_comparison(
                raw["pit"], scaled["pit"], plots_dir,
                title=f"Calibration — {run_label} ({head})",
                sigma_scale=sigma_scale,
            )

        return payload
    finally:
        shutdown_dataloaders(test_loader)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    cfg = load_config(run_dir / "config.yaml")
    role = _resolve_role(run_dir, args.role)
    ckpt = checkpoint_path(run_dir, role)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoint for role '{role}' in {run_dir}. "
            f"Expected best_posterior.pt or best_model.pt."
        )

    set_seed(cfg.training.seed)
    device = get_device()

    train_path = ROOT / cfg.data.train_path
    res_dir    = ROOT / cfg.data.res_dir
    train_ds = GalaxyDataset(
        parquet_path=train_path,
        res_dir=res_dir,
        target_col=cfg.data.target_col,
        dropout_cfg=None,
        active_surveys=cfg.data.active_surveys or None,
        log_target=cfg.data.log_target,
        include_errors=cfg.data.include_errors,
    )
    n_total = len(train_ds)
    n_val   = max(1, int(0.1 * n_total))
    rng = torch.Generator().manual_seed(cfg.training.seed)
    idx = torch.randperm(n_total, generator=rng).tolist()
    val_idx = idx[n_total - n_val:]

    val_loader = DataLoader(
        Subset(train_ds, val_idx),
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=resolve_num_workers(cfg.training.num_workers),
    )

    head_type = cfg.head.type
    if role == ROLE_POSTERIOR and cfg.training.split_training:
        head_type = cfg.training.stage2.head or cfg.head.type

    model = build_model(cfg, device=device, head_type=head_type)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    print(f"Run: {run_dir.name}  |  checkpoint: {ckpt.name}  |  head: {head_type}")
    try:
        phc_payload = run_post_hoc_calibration(
            model, val_loader, device, run_dir, cfg,
            checkpoint_role=role,
        )
    finally:
        shutdown_dataloaders(val_loader)

    if phc_payload is None:
        saved = load_post_hoc(run_dir)
        if saved is None:
            print("  No post-hoc calibration produced — skipping test evaluation.")
            return
        sigma_scale = float(saved["sigma_scale"])
    else:
        sigma_scale = float(phc_payload["sigma_scale"])

    if not args.no_plots:
        print(f"\n{'='*65}")
        print("  Test-set evaluation (post-hoc σ scaling)")
        print(f"{'='*65}")
        _evaluate_test_set(
            model, cfg, device, run_dir, sigma_scale,
            write_plots=True,
        )


if __name__ == "__main__":
    main()
