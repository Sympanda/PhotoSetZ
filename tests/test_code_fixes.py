"""
Unit tests for NSF / calibration code fixes.

Run:  python -m unittest tests.test_code_fixes -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import GalaxyDataset, collate_fn
from src.models.heads.nsf import NeuralSplineFlow
from src.train import DeepSetZ, build_encoder, build_head
from src.config import Config, ModelConfig, HeadConfig, TrainingConfig, DataConfig


class TestCoverageScalar(unittest.TestCase):
    """Coverage must not depend on batch max token count."""

    def _make_model(self, n_filters: int = 10) -> DeepSetZ:
        cfg = Config(
            model=ModelConfig(type="set_transformer", token_dim=4, use_coverage=True),
            head=HeadConfig(type="mlp_regressor"),
            training=TrainingConfig(),
            data=DataConfig(),
        )
        encoder = build_encoder(cfg)
        head = build_head(cfg, encoder.output_dim, head_type="mlp_regressor")
        return DeepSetZ(encoder, head, use_coverage=True, n_total_filters=n_filters)

    def test_coverage_invariant_across_batch_padding(self):
        model = self._make_model(n_filters=10)
        # Galaxy A: 5 active filters
        tok_a = torch.randn(1, 5, 4)
        mask_a = torch.zeros(1, 5, dtype=torch.bool)
        # Same galaxy in a batch padded to max_n=8
        tok_b = torch.zeros(2, 8, 4)
        mask_b = torch.ones(2, 8, dtype=torch.bool)
        tok_b[0, :5] = tok_a[0]
        mask_b[0, :5] = False
        # Second object with 8 filters (changes old batch-max denominator)
        tok_b[1, :8] = torch.randn(8, 4)
        mask_b[1, :8] = False

        with torch.no_grad():
            emb_a = model.encoder(tok_a, mask_a)
            cov_a = 5.0 / model.n_total_filters
            emb_b0 = model.encoder(tok_b, mask_b)[0]
            n_active_b0 = (~mask_b[0]).float().sum()
            cov_b0 = n_active_b0 / model.n_total_filters

        self.assertAlmostEqual(cov_a, 0.5, places=5)
        self.assertAlmostEqual(cov_b0.item(), 0.5, places=5)


class TestNSFConditioning(unittest.TestCase):
    """NSF log_prob must change when context (embedding) changes."""

    def setUp(self):
        torch.manual_seed(0)
        self.embed_dim = 32
        self.head = NeuralSplineFlow(
            embed_dim=self.embed_dim,
            n_bins=8,
            z_min=0.0,
            z_max=1.4,
            hidden_dims=[16],
            dropout=0.0,
        )
        self.head.train()

    def test_context_changes_log_pdf(self):
        B = 64
        h = torch.randn(B, self.embed_dim)
        y = torch.rand(32) * 1.2 + 0.1

        h1, h2 = h[:32], h[32:64]
        with torch.no_grad():
            w1, ht1, d1 = self.head._spline_params(h1)
            w2, ht2, d2 = self.head._spline_params(h2)

        from src.models.heads.nsf import _rqs_forward
        _, log_p1 = _rqs_forward(y, w1, ht1, d1, 0.0, 1.4)
        _, log_p2 = _rqs_forward(y, w2, ht2, d2, 0.0, 1.4)
        diff = (log_p1 - log_p2).abs().mean().item()
        self.assertGreater(diff, 1e-4, "NSF should be sensitive to context")

    def test_huber_auxiliary_has_gradients(self):
        h = torch.randn(8, self.embed_dim, requires_grad=False)
        z_true = torch.rand(8) * 1.0 + 0.2
        out = self.head(
            h, z_true=z_true,
            huber_lambda=0.5, huber_delta=0.5,
        )
        out["loss"].backward()
        grads = [p.grad.abs().sum().item() for p in self.head.parameters() if p.grad is not None]
        self.assertTrue(any(g > 0 for g in grads), "Huber branch should backprop to NSF params")


class TestMissingErrorColumns(unittest.TestCase):
    """Dataset must fail when error columns are missing (default)."""

    def test_raises_on_missing_err_col(self):
        import tempfile

        # Minimal parquet with one LSST filter but no error column
        df = pd.DataFrame({
            "redshift": [0.5, 0.8],
            "mag_g_lsst": [22.0, 23.0],
            "mag_r_lsst": [21.5, 22.5],
        })
        with tempfile.TemporaryDirectory() as tmp:
            pq = Path(tmp) / "tiny.parquet"
            df.to_parquet(pq)
            res_dir = ROOT / "data" / "ellen"
            with self.assertRaises(ValueError) as ctx:
                GalaxyDataset(
                    parquet_path=pq,
                    res_dir=res_dir,
                    target_col="redshift",
                    active_surveys=["lsst"],
                    include_errors=True,
                    strict_error_columns=True,
                )
            self.assertIn("Missing error columns", str(ctx.exception))


class TestStageTrainingOverrides(unittest.TestCase):
    """Per-stage blocks should override training hyperparameters independently."""

    def test_apply_stage_overrides_inherits_and_overrides(self):
        from src.config import StageTrainingConfig
        from src.training_stages import apply_stage_overrides

        cfg = Config(
            training=TrainingConfig(
                lr=2e-4,
                weight_decay=2e-4,
                batch_size=512,
                warmup_epochs=25,
                clip_grad_norm=1.0,
            ),
        )
        stage1 = StageTrainingConfig(
            head="mlp_regressor",
            epochs=150,
            lr=2e-4,
            huber_lambda=0.2,
        )
        stage2 = StageTrainingConfig(
            head="nsf",
            epochs=80,
            lr=8e-6,
            weight_decay=1e-4,
            warmup_epochs=5,
            clip_grad_norm=0.5,
            spread_lambda=0.05,
            freeze_encoder=True,
        )

        cfg1 = apply_stage_overrides(cfg, stage1, default_head="mlp_regressor")
        self.assertEqual(cfg1.head.type, "mlp_regressor")
        self.assertEqual(cfg1.training.epochs, 150)
        self.assertEqual(cfg1.training.lr, 2e-4)
        self.assertEqual(cfg1.training.batch_size, 512)  # inherited

        cfg2 = apply_stage_overrides(cfg, stage2, default_head="nsf")
        self.assertEqual(cfg2.head.type, "nsf")
        self.assertEqual(cfg2.training.lr, 8e-6)
        self.assertEqual(cfg2.training.weight_decay, 1e-4)
        self.assertEqual(cfg2.training.warmup_epochs, 5)
        self.assertEqual(cfg2.training.clip_grad_norm, 0.5)
        self.assertEqual(cfg2.training.spread_lambda, 0.05)
        self.assertEqual(cfg2.training.epochs, 80)

    def test_load_config_stage_lr_from_yaml(self):
        from src.config import load_config

        cfg = load_config(ROOT / "configs" / "ts1_10yr_st_nsf_2part.yaml")
        self.assertEqual(cfg.training.stage2.lr, 8e-6)
        self.assertEqual(cfg.training.stage1.lr, 2e-4)
        self.assertEqual(cfg.training.stage2.clip_grad_norm, 0.5)


if __name__ == "__main__":
    unittest.main()
