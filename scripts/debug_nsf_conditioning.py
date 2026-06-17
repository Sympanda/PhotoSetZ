#!/usr/bin/env python3
"""
Debug checks that NSF conditioning is active.

Usage
-----
    python scripts/debug_nsf_conditioning.py outputs/ts1_10yr_nsf_01_st
    python scripts/debug_nsf_conditioning.py --random   # untrained NSF sanity checks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import src.platform_fix  # noqa: F401

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data_options import dataset_kwargs_from_config
from src.dataset import GalaxyDataset, collate_fn
from src.dataloader_utils import resolve_num_workers, shutdown_dataloaders
from src.models.heads.nsf import NeuralSplineFlow, _rqs_forward
from src.run_artifacts import checkpoint_path, ROLE_END_TO_END, ROLE_POSTERIOR
from src.checkpoint_loader import load_model_from_checkpoint
from src.train import build_model, get_device, set_seed


def _resolve_ckpt(run_dir: Path) -> Path:
    for role in (ROLE_POSTERIOR, ROLE_END_TO_END):
        p = checkpoint_path(run_dir, role)
        if p:
            return p
    raise FileNotFoundError(f"No checkpoint in {run_dir}")


def context_sensitivity_tests(head: NeuralSplineFlow, embeddings: torch.Tensor, z_true: torch.Tensor):
    """Tests A & B from the code-fix instructions."""
    h = embeddings
    y = z_true[: h.size(0)]

    with torch.no_grad():
        w1, h1, d1 = head._spline_params(h[:32])
        w2, h2, d2 = head._spline_params(h[32:64])
        same_y = y[:32]
        _, lp1 = _rqs_forward(same_y, w1, h1, d1, head.z_min, head.z_max)
        _, lp2 = _rqs_forward(same_y, w2, h2, d2, head.z_min, head.z_max)
        diff = (lp1 - lp2).abs().mean().item()
        print(f"Test A — context sensitivity |log p diff|: {diff:.6f}")
        if diff < 1e-5:
            print("  WARNING: NSF appears insensitive to context.")

        perm = torch.randperm(h.size(0), device=h.device)
        lp_all = []
        for i in range(h.size(0)):
            w, ht, d = head._spline_params(h[i : i + 1])
            _, lp = _rqs_forward(y[i : i + 1], w, ht, d, head.z_min, head.z_max)
            lp_all.append(lp)
        lp_correct = torch.cat(lp_all).mean().item()

        lp_shuf = []
        for i in range(h.size(0)):
            w, ht, d = head._spline_params(h[perm[i] : perm[i] + 1])
            _, lp = _rqs_forward(y[i : i + 1], w, ht, d, head.z_min, head.z_max)
            lp_shuf.append(lp)
        lp_shuffle = torch.cat(lp_shuf).mean().item()
        print(f"Test B — mean log p (matched): {lp_correct:.4f}  (shuffled): {lp_shuffle:.4f}")
        if lp_correct <= lp_shuffle:
            print("  WARNING: shuffled context is not worse — check conditioning wiring.")


def sampling_shape_test(head: NeuralSplineFlow, embeddings: torch.Tensor):
    """Test D — posterior modes should differ across objects."""
    with torch.no_grad():
        widths, heights, derivs = head._spline_params(embeddings[:16])
        ests = head.point_estimates_from_params(widths, heights, derivs, n_grid=128)
        modes = ests["z_mode"].detach().cpu().numpy()
    spread = float(modes.max() - modes.min())
    print(f"Test D — mode spread over 16 objects: {spread:.6f}  "
          f"(min={modes.min():.4f}, max={modes.max():.4f})")
    if spread < 1e-5:
        print("  WARNING: all posterior modes are identical — possible collapse.")


def random_untrained_checks():
    """Sanity checks on a fresh NSF (should show context sensitivity)."""
    torch.manual_seed(42)
    head = NeuralSplineFlow(embed_dim=64, n_bins=16, z_min=0.0, z_max=1.4, hidden_dims=[32])
    h = torch.randn(64, 64)
    y = torch.rand(64) * 1.0 + 0.2
    print("Untrained NSF checks:")
    context_sensitivity_tests(head, h, y)
    sampling_shape_test(head, h)


def main():
    parser = argparse.ArgumentParser(description="NSF conditioning debug checks")
    parser.add_argument("run_dir", nargs="?", help="Path to outputs/<run_name>")
    parser.add_argument("--random", action="store_true", help="Run on untrained NSF only")
    parser.add_argument("--n-batches", type=int, default=4)
    args = parser.parse_args()

    if args.random or not args.run_dir:
        random_untrained_checks()
        return

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    set_seed(cfg.training.seed)
    device = get_device()

    train_path = ROOT / cfg.data.train_path
    res_dir = ROOT / cfg.data.res_dir
    ds = GalaxyDataset(
        parquet_path=train_path,
        res_dir=res_dir,
        target_col=cfg.data.target_col,
        dropout_cfg=None,
        active_surveys=cfg.data.active_surveys or None,
        log_target=cfg.data.log_target,
        **dataset_kwargs_from_config(cfg.data),
    )
    loader = DataLoader(
        ds, batch_size=64, shuffle=True, collate_fn=collate_fn,
        num_workers=resolve_num_workers(0),
    )

    ckpt = _resolve_ckpt(run_dir)
    model = load_model_from_checkpoint(
        cfg, ckpt, device=device, n_total_filters=ds.n_filters,
    )
    model.eval()

    if not isinstance(model.head, NeuralSplineFlow):
        print(f"Head is {type(model.head).__name__}, not NSF — exiting.")
        return

    print(f"Run: {run_dir.name}  |  checkpoint: {ckpt.name}  |  device: {device}")
    head = model.head
    embeddings, z_true = [], []
    try:
        with torch.no_grad():
            for i, (tokens, mask, z) in enumerate(loader):
                if i >= args.n_batches:
                    break
                tokens = tokens.to(device)
                mask = mask.to(device)
                h_set = model.encoder(tokens, mask)
                if model.bottleneck is not None:
                    h_set = model.bottleneck(h_set)
                if model.density_context_branch or model.use_coverage_summary:
                    emb = model._nsf_density_context(h_set, tokens, mask)
                elif model.use_coverage:
                    emb = torch.cat([h_set, model._coverage_scalar(mask)], dim=-1)
                else:
                    emb = h_set
                embeddings.append(emb)
                z_true.append(z.to(device))
        h = torch.cat(embeddings, dim=0)
        y = torch.cat(z_true, dim=0)
        if h.size(0) < 64:
            print(f"Only {h.size(0)} samples — need ≥64 for tests A/B.")
            return
        context_sensitivity_tests(head, h, y)
        sampling_shape_test(head, h)
    finally:
        shutdown_dataloaders(loader)


if __name__ == "__main__":
    main()
