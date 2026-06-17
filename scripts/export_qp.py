"""
Export DeepSetZ posterior predictions to qp format for PZDC submission.

Produces an HDF5 file in qp format containing:
  - Full p(z) posteriors for every object in the test file
  - Ancillary data: 'zmode' (point estimate) and 'object_id'

Supported head types
--------------------
  MDN       → qp.Ensemble(qp.mixmod, ...)   — Gaussian mixture
  BinnedPDF → qp.Ensemble(qp.interp, ...)   — interpolated grid
  NSF       → qp.Ensemble(qp.interp, ...)   — evaluated on a dense grid
  SBI_NPE   → qp.Ensemble(qp.interp, ...)   — evaluated on a dense z-grid
  MLP       → qp.Ensemble(qp.interp, ...)   — approximated as narrow Gaussian

Usage
-----
    python scripts/export_qp.py \\
        --config  outputs/my_run/config.yaml \\
        --ckpt    outputs/my_run/best_model.pt \\
        --test    data/pzdc/pz_challenge_taskset_1_cardinal_test_10yr.parquet \\
        --output  submissions/ts1_cardinal_10yr_pz_estimate.hdf5

    # Or use a raw HDF5 test file directly (auto-converted)
    python scripts/export_qp.py \\
        --config  outputs/my_run/config.yaml \\
        --ckpt    outputs/my_run/best_model.pt \\
        --test    data/pzdc/hdf5/pz_challenge_taskset_1_cardinal_test_10yr.hdf5 \\
        --output  submissions/ts1_cardinal_10yr_pz_estimate.hdf5
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.checkpoint_loader import load_model_from_checkpoint
from src.dataset import GalaxyDataset, collate_fn
from src.train import get_device

import qp


# --------------------------------------------------------------------------
# Ensemble builders
# --------------------------------------------------------------------------

def _build_ensemble_mdn(
    pi:    np.ndarray,  # (N, C)
    mu:    np.ndarray,  # (N, C)
    sigma: np.ndarray,  # (N, C)
    log_target: bool,
) -> "qp.Ensemble":
    if log_target:
        # Convert Gaussian parameters from log(1+z) space to approximate real-z space
        # (the mixture in log space, interpreted at exp(mu)-1 ≈ centres)
        mu_real    = np.expm1(mu)
        sigma_real = sigma * np.exp(mu)   # Jacobian: d(exp(t)-1)/dt = exp(t)
        return qp.Ensemble(qp.mixmod, data={"means": mu_real, "stds": sigma_real, "weights": pi})
    return qp.Ensemble(qp.mixmod, data={"means": mu, "stds": sigma, "weights": pi})


def _build_ensemble_grid(
    pdf_grid:  np.ndarray,  # (N, G)
    z_grid:    np.ndarray,  # (G,)
    log_target: bool,
) -> "qp.Ensemble":
    """Use an interpolated grid representation — works for BinnedPDF, NSF, and MLP."""
    if log_target:
        # Transform grid to real-z space using change of variables
        # PDF_real(z) = PDF_log(t) * dt/dz  where t = log(1+z), dt/dz = 1/(1+z)
        z_real  = np.expm1(z_grid)                    # (G,)
        jacobian = 1.0 / (1.0 + z_real)               # dt/dz
        pdf_real = pdf_grid * jacobian[None, :]        # (N, G)
        return qp.Ensemble(qp.interp, data={"xvals": z_real, "yvals": pdf_real})
    return qp.Ensemble(qp.interp, data={"xvals": z_grid, "yvals": pdf_grid})


def _build_ensemble_mlp(
    z_pred:    np.ndarray,  # (N,)  point estimates
    log_target: bool,
    sigma_approx: float = 0.05,
) -> "qp.Ensemble":
    """Approximate MLP predictions as narrow Gaussians for qp compatibility."""
    if log_target:
        z_pred = np.expm1(z_pred)
    N = len(z_pred)
    means   = z_pred[:, None]                     # (N, 1)
    stds    = np.full((N, 1), sigma_approx)
    weights = np.ones((N, 1))
    return qp.Ensemble(qp.mixmod, data={"means": means, "stds": stds, "weights": weights})


# --------------------------------------------------------------------------
# Main export function
# --------------------------------------------------------------------------

@torch.no_grad()
def export_qp(
    config_path: str | Path,
    ckpt_path:   str | Path,
    test_path:   str | Path,
    output_path: str | Path,
    batch_size:  int = 1024,
    nsf_grid:    int = 300,   # grid points for NSF / BinnedPDF interp output
) -> None:
    """
    Load a DeepSetZ model, run inference on a test set, and write a qp HDF5.

    Parameters
    ----------
    config_path  : YAML config used for training (saved in outputs/<run>/config.yaml)
    ckpt_path    : Trained model checkpoint (best_model.pt or final_model.pt)
    test_path    : Test parquet or HDF5 file (HDF5 is auto-converted to a temp parquet)
    output_path  : Destination HDF5 file for the qp ensemble
    nsf_grid     : Number of evaluation points for NSF / BinnedPDF grid output
    """
    cfg         = load_config(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Convert HDF5 → parquet if needed ────────────────────────────────
    test_path = Path(test_path)
    _tmp_parquet = None
    if test_path.suffix in (".hdf5", ".h5"):
        import h5py, pandas as pd
        print(f"  Converting {test_path.name} → temporary parquet …")
        with h5py.File(test_path, "r") as f:
            df = pd.DataFrame({k: f[k][()] for k in f.keys()})
        _tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        df.to_parquet(_tmp.name)
        test_path    = Path(_tmp.name)
        _tmp_parquet = test_path

    # ── Load dataset ─────────────────────────────────────────────────────
    root        = Path(config_path).parent.parent   # outputs/<run>/config.yaml → repo root
    res_dir     = root / cfg.data.res_dir
    log_target  = cfg.data.log_target

    # The official test files have no redshift column — use a dummy target
    dummy_target = cfg.data.target_col
    try:
        test_ds = GalaxyDataset(
            parquet_path   = test_path,
            res_dir        = res_dir,
            target_col     = dummy_target,
            dropout_cfg    = None,
            active_surveys = cfg.data.active_surveys or None,
            log_target     = log_target,
            include_errors = cfg.data.include_errors,
        )
    except (ValueError, KeyError):
        # Target column absent — use object_id as a dummy stand-in
        # This is expected for the official unlabelled test files
        print(f"  [info] No target column '{dummy_target}' found — "
              "using dummy redshifts for export (labels are absent, that's OK).")
        import pandas as pd
        df = pd.read_parquet(test_path)
        df["_dummy_z"] = 0.5
        _tmp2 = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        df.to_parquet(_tmp2.name)
        test_ds = GalaxyDataset(
            parquet_path   = Path(_tmp2.name),
            res_dir        = res_dir,
            target_col     = "_dummy_z",
            dropout_cfg    = None,
            active_surveys = cfg.data.active_surveys or None,
            log_target     = False,
            include_errors = cfg.data.include_errors,
        )

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    object_ids = test_ds.object_ids

    # ── Build model ───────────────────────────────────────────────────────
    device = get_device()
    model  = load_model_from_checkpoint(
        cfg, ckpt_path, device=device, n_total_filters=test_ds.n_filters,
    )
    model.eval()

    from src.models.sbi_stage2 import SBIStage2Model
    head = model.sbi_head if isinstance(model, SBIStage2Model) else model.head
    head_type = head.__class__.__name__
    print(f"  Head type: {head_type}")
    print(f"  Test galaxies: {len(test_ds):,}")

    # ── Inference ────────────────────────────────────────────────────────
    from src.models.heads.mdn        import MDN
    from src.models.heads.binned_pdf import BinnedPDF
    from src.models.heads.nsf        import NeuralSplineFlow
    from src.models.heads.sbi_npe    import SBINPEHead

    all_z_pred, all_pi, all_mu, all_sigma = [], [], [], []
    all_probs, all_widths, all_heights, all_derivs = [], [], [], []
    all_sbi_pdf = []

    for tokens, mask, _ in loader:
        tokens_d = tokens.to(device)
        mask_d   = mask.to(device)
        out = model(tokens_d, mask_d)
        all_z_pred.append(out["z_pred"].cpu())
        if isinstance(head, MDN):
            all_pi.append(out["pi"].cpu())
            all_mu.append(out["mu"].cpu())
            all_sigma.append(out["sigma"].cpu())
        elif isinstance(head, BinnedPDF):
            all_probs.append(out["probs"].cpu())
        elif isinstance(head, NeuralSplineFlow):
            all_widths.append(out["widths"].cpu())
            all_heights.append(out["heights"].cpu())
            all_derivs.append(out["derivs"].cpu())
        elif isinstance(head, SBINPEHead):
            h = model._encode(tokens_d, mask_d)
            ctx = model.context_builder(h, tokens_d, mask_d, out["z_pred"])
            z_grid = head.z_grid(device)
            _, pdf = head.evaluate_grid(z_grid, ctx)
            all_sbi_pdf.append(pdf.cpu())

    z_pred_raw = torch.cat(all_z_pred).numpy()

    # z_mode in real z space (required by PZDC submission format)
    z_mode = np.expm1(z_pred_raw) if log_target else z_pred_raw

    # ── Build qp ensemble ─────────────────────────────────────────────────
    print(f"  Building qp ensemble …")

    if isinstance(head, MDN):
        pi    = torch.cat(all_pi).numpy()
        mu    = torch.cat(all_mu).numpy()
        sigma = torch.cat(all_sigma).numpy()
        ens   = _build_ensemble_mdn(pi, mu, sigma, log_target)

    elif isinstance(head, BinnedPDF):
        probs        = torch.cat(all_probs).numpy()    # (N, K)
        bin_centres  = head.bin_centres.cpu().numpy()  # (K,)
        bin_width    = float(head.bin_width.item())
        # qp.interp expects PDF values at the bin-centre x-positions
        pdf_grid     = probs / bin_width               # (N, K)
        ens = _build_ensemble_grid(pdf_grid, bin_centres, log_target)

    elif isinstance(head, NeuralSplineFlow):
        from src.models.heads.nsf import _rqs_forward
        widths  = torch.cat(all_widths)
        heights = torch.cat(all_heights)
        derivs  = torch.cat(all_derivs)
        N       = widths.shape[0]

        # Evaluate PDF on a dense z-grid
        z_grid  = torch.linspace(head.z_min + 1e-5, head.z_max - 1e-5, nsf_grid)
        pdf_rows = []
        chunk = 256
        for start in range(0, N, chunk):
            w = widths[start:start+chunk]
            h = heights[start:start+chunk]
            d = derivs[start:start+chunk]
            B = w.shape[0]
            z_rep = z_grid.unsqueeze(0).expand(B, -1).reshape(-1)
            w_rep = w.unsqueeze(1).expand(-1, nsf_grid, -1).reshape(B * nsf_grid, -1)
            h_rep = h.unsqueeze(1).expand(-1, nsf_grid, -1).reshape(B * nsf_grid, -1)
            d_rep = d.unsqueeze(1).expand(-1, nsf_grid, -1).reshape(B * nsf_grid, -1)
            _, log_pdf = _rqs_forward(z_rep, w_rep, h_rep, d_rep, head.z_min, head.z_max)
            pdf_rows.append(log_pdf.exp().reshape(B, nsf_grid).numpy())
        pdf_grid = np.vstack(pdf_rows)   # (N, nsf_grid)
        ens = _build_ensemble_grid(pdf_grid, z_grid.numpy(), log_target)

    elif isinstance(head, SBINPEHead):
        pdf_grid = torch.cat(all_sbi_pdf).numpy()
        z_grid   = head.z_grid(device).cpu().numpy()
        ens = _build_ensemble_grid(pdf_grid, z_grid, log_target=False)

    else:
        # MLP regressor — approximate as narrow Gaussian
        ens = _build_ensemble_mlp(z_pred_raw, log_target)

    # ── Attach ancillary data and write ───────────────────────────────────
    ens.set_ancil({"zmode": z_mode, "object_id": object_ids})
    ens.write_to(str(output_path))

    if _tmp_parquet is not None:
        _tmp_parquet.unlink(missing_ok=True)

    print(f"  Written: {output_path}  ({output_path.stat().st_size / 1024:.1f} kB)")
    print(f"  {len(test_ds):,} objects  |  head={head_type}  |  log_target={log_target}")
    print(f"  z_mode range: [{z_mode.min():.3f}, {z_mode.max():.3f}]")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DeepSetZ p(z) posteriors to qp HDF5 for PZDC submission"
    )
    parser.add_argument("--config",  required=True, help="Path to config.yaml")
    parser.add_argument("--ckpt",    required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--test",    required=True, help="Test parquet or HDF5 file")
    parser.add_argument("--output",  required=True, help="Output qp HDF5 file path")
    parser.add_argument("--batch",   type=int, default=1024, help="Inference batch size")
    parser.add_argument("--nsf-grid",type=int, default=300,
                        help="Grid points for NSF/BinnedPDF PDF evaluation (default: 300)")
    args = parser.parse_args()

    print("=" * 65)
    print("  DeepSetZ — qp Export for PZDC Submission")
    print("=" * 65)

    export_qp(
        config_path = args.config,
        ckpt_path   = args.ckpt,
        test_path   = args.test,
        output_path = args.output,
        batch_size  = args.batch,
        nsf_grid    = args.nsf_grid,
    )

    print("Done.")


if __name__ == "__main__":
    main()
