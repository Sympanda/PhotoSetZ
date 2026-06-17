"""
Tests for SED-like filter handling extension (backwards compatibility + new modes).

Run:  python -m unittest tests.test_sed_extension -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config, DataConfig, ModelConfig, HeadConfig, TrainingConfig, load_config
from src.coverage_summary import COVERAGE_SUMMARY_DIM, compute_coverage_summary
from src.data_options import compute_token_dim, resolve_data_options
from src.dataset import GalaxyDataset, collate_fn, DropoutConfig


class TestBackwardsCompat(unittest.TestCase):
    def test_default_token_dim(self):
        opts = resolve_data_options(DataConfig())
        self.assertEqual(opts.token_dim, 4)
        self.assertFalse(opts.encode_nondetections)

    def test_errors_token_dim(self):
        self.assertEqual(compute_token_dim(include_errors=True), 5)

    def test_old_config_loads(self):
        cfg = load_config(ROOT / "configs" / "ts1_10yr_st_nsf.yaml")
        opts = resolve_data_options(cfg.data)
        self.assertEqual(opts.token_dim, 5)
        self.assertFalse(cfg.model.use_coverage_summary)


class TestDetectionFlags(unittest.TestCase):
    def test_token_dim_with_flags(self):
        self.assertEqual(
            compute_token_dim(include_errors=True, add_detection_flags=True),
            7,
        )


class TestCoverageSummary(unittest.TestCase):
    def test_fixed_shape_batch_invariant(self):
        B, N, td = 4, 10, 5
        tokens = torch.randn(B, N, td)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, 5:] = True
        s1 = compute_coverage_summary(
            tokens, mask, n_total_filters=20, include_errors=True,
        )
        # Same first object in batch with different padding width
        tokens2 = tokens.clone()
        mask2 = mask.clone()
        mask2[1, 8:] = True
        s2 = compute_coverage_summary(
            tokens2, mask2, n_total_filters=20, include_errors=True,
        )
        self.assertEqual(s1.shape, (B, COVERAGE_SUMMARY_DIM))
        self.assertAlmostEqual(s1[0, 0].item(), s2[0, 0].item(), places=5)


class TestNondetectionTokens(unittest.TestCase):
    def _tiny_dataset(self, **kwargs):
        df = pd.DataFrame({
            "redshift": [0.5],
            "mag_g_lsst": [np.nan],
            "mag_r_lsst": [22.0],
            "mag_g_lsst_err": [0.05],
            "mag_r_lsst_err": [0.04],
        })
        tmp = tempfile.mkdtemp()
        pq = Path(tmp) / "t.parquet"
        df.to_parquet(pq)
        return GalaxyDataset(
            parquet_path=pq,
            res_dir=ROOT / "data" / "ellen",
            target_col="redshift",
            active_surveys=["lsst"],
            include_errors=True,
            dropout_cfg=DropoutConfig(
                p_complete=1.0, p_preset=0, p_survey_drop=0, p_aggressive=0,
                min_filters=1,
            ),
            **kwargs,
        )

    def test_drop_policy_omits_nan(self):
        ds = self._tiny_dataset(
            encode_nondetections=False, nondetection_policy="drop",
        )
        tokens, _, _ = ds[0]
        self.assertEqual(tokens.shape[1], 5)
        self.assertEqual(tokens.shape[0], 1)

    def test_keep_token_emits_nondetection(self):
        ds = self._tiny_dataset(
            encode_nondetections=True,
            nondetection_policy="keep_token",
            add_detection_flags=True,
        )
        self.assertEqual(ds.token_dim, 7)
        tokens, _, _ = ds[0]
        self.assertEqual(tokens.shape[0], 2)
        # NaN band: is_detected=0, is_nondetected=1
        nan_row = tokens[0]
        self.assertAlmostEqual(nan_row[-2].item(), 0.0)
        self.assertAlmostEqual(nan_row[-1].item(), 1.0)
        det_row = tokens[1]
        self.assertAlmostEqual(det_row[-2].item(), 1.0)
        self.assertAlmostEqual(det_row[-1].item(), 0.0)


class TestStrictErrors(unittest.TestCase):
    def test_strict_raises(self):
        df = pd.DataFrame({"redshift": [0.5], "mag_g_lsst": [22.0]})
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "t.parquet"
            df.to_parquet(pq)
            with self.assertRaises(ValueError):
                GalaxyDataset(
                    parquet_path=pq,
                    res_dir=ROOT / "data" / "ellen",
                    target_col="redshift",
                    active_surveys=["lsst"],
                    include_errors=True,
                    strict_error_columns=True,
                )


class TestNSFGridCalibration(unittest.TestCase):
    def test_grid_pdf_integrates(self):
        from src.calibration import nsf_grid_log_pdf
        from src.models.heads.nsf import NeuralSplineFlow

        head = NeuralSplineFlow(embed_dim=8, n_bins=8, z_min=0.0, z_max=1.4, hidden_dims=[16])
        h = torch.randn(4, 8)
        w, ht, d = head._spline_params(h)
        with torch.no_grad():
            _, pdf, dz = nsf_grid_log_pdf(head, w, ht, d, n_grid=128, temperature=1.2)
        integral = (pdf.sum(dim=-1) * dz).numpy()
        np.testing.assert_allclose(integral, 1.0, atol=0.05)


class TestEncoderBottleneck(unittest.TestCase):
    def test_build_with_bottleneck(self):
        from src.config import load_config
        from src.train import build_model

        cfg = load_config(ROOT / "configs" / "ts1_10yr_st_nsf_sed.yaml")
        model = build_model(cfg, n_total_filters=10)
        self.assertIsNotNone(model.bottleneck)
        self.assertEqual(model.bottleneck.latent_dim, 64)
        w = model.head.trunk[0].weight
        self.assertEqual(w.shape[1], 64)

    def test_bottleneck_disabled_by_default(self):
        from src.config import load_config
        from src.train import build_model

        cfg = load_config(ROOT / "configs" / "ts1_10yr_st_nsf.yaml")
        model = build_model(cfg, n_total_filters=10)
        self.assertIsNone(model.bottleneck)


class TestCheckpointLoader(unittest.TestCase):
    def test_old_run_without_use_coverage_in_yaml(self):
        from src.config import load_config
        from src.checkpoint_loader import load_model_from_checkpoint

        run_dir = ROOT / "outputs" / "06_set_transformer_mlp"
        if not (run_dir / "best_model.pt").exists():
            self.skipTest("06_set_transformer_mlp checkpoint not present")
        cfg = load_config(run_dir / "config.yaml")
        model = load_model_from_checkpoint(cfg, run_dir / "best_model.pt")
        self.assertFalse(model.use_coverage)
        w = model.head.net[0].weight
        self.assertEqual(w.shape[1], 128)

    def test_nsf_run_with_coverage(self):
        from src.config import load_config
        from src.checkpoint_loader import load_model_from_checkpoint

        run_dir = ROOT / "outputs" / "ts1_10yr_nsf_01_st"
        if not (run_dir / "best_model.pt").exists():
            self.skipTest("nsf run not present")
        cfg = load_config(run_dir / "config.yaml")
        model = load_model_from_checkpoint(cfg, run_dir / "best_model.pt")
        self.assertTrue(model.use_coverage)
        w = model.head.trunk[0].weight
        self.assertEqual(w.shape[1], 385)


if __name__ == "__main__":
    unittest.main()
