"""
Acceptance tests for SBI/NPE posterior head.

Run:  python -m unittest tests.test_sbi_npe -v
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config, load_config
from src.models.target_transforms import RedshiftTransform


class TestImportCompatibility(unittest.TestCase):
    """Old configs must load without sbi installed."""

    def test_load_mlp_config(self):
        cfg = load_config(ROOT / "configs" / "deepsets.yaml")
        self.assertEqual(cfg.head.type, "mlp_regressor")

    def test_heads_registry_no_sbi_import(self):
        mod = importlib.import_module("src.models.heads")
        self.assertIn("nsf", mod.HEAD_REGISTRY)


class TestRedshiftTransform(unittest.TestCase):
    def test_log1p_logit_roundtrip(self):
        rt = RedshiftTransform("log1p_logit", y_min=0.0, y_max=1.45)
        z = torch.tensor([0.0, 0.05, 0.5, 1.0, 2.0])
        z2 = rt.t_to_z(rt.z_to_t(z))
        self.assertTrue(torch.allclose(z, z2, atol=1e-4))


@unittest.skipIf(importlib.util.find_spec("sbi") is None, "sbi not installed")
class TestSBINPEHead(unittest.TestCase):
    def setUp(self):
        from src.models.heads.sbi_context import SBIContextBuilder
        from src.models.heads.sbi_npe import SBINPEHead

        self.ctx_builder = SBIContextBuilder(
            context_mode="frozen_point",
            repr_dim=8,
            include_point_prediction=True,
            include_coverage_summary=False,
            context_projection_enabled=True,
            projection_hidden=[16, 8],
            projection_dropout=0.0,
            projection_layer_norm=True,
            context_dim=None,
            n_total_filters=10,
            include_errors=False,
            add_detection_flags=False,
        )
        self.head = SBINPEHead(
            context_dim=self.ctx_builder.context_dim,
            hidden_features=16,
            num_transforms=2,
            num_bins=4,
            n_grid=128,
        )

    def test_log_prob_changes_with_context(self):
        self.head.ensure_flow(torch.device("cpu"))
        z = torch.tensor([0.3, 0.5, 0.7])
        c1 = torch.randn(3, self.ctx_builder.context_dim)
        c2 = torch.randn(3, self.ctx_builder.context_dim)
        lp1 = self.head.log_prob(z, c1)
        lp2 = self.head.log_prob(z, c2)
        self.assertFalse(torch.allclose(lp1, lp2))

    def test_grid_normalization(self):
        self.head.ensure_flow(torch.device("cpu"))
        ctx = torch.randn(4, self.ctx_builder.context_dim)
        z_grid = self.head.z_grid(torch.device("cpu"), z_min=0.0, z_max=2.5)
        _, pdf = self.head.evaluate_grid(z_grid, ctx)
        dz = (z_grid[1] - z_grid[0]).item()
        integrals = (pdf.sum(dim=-1) * dz)
        self.assertTrue(torch.allclose(integrals, torch.ones_like(integrals), atol=0.05))

    def test_tiny_overfit_decreases_nll(self):
        from src.models.heads.sbi_npe import SBINPEHead

        head = SBINPEHead(context_dim=4, hidden_features=16, num_transforms=2, num_bins=4)
        head.ensure_flow(torch.device("cpu"))
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        ctx = torch.randn(32, 4)
        z = torch.rand(32) * 1.5
        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = head.loss(z, ctx)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])


class TestSBIDependencyError(unittest.TestCase):
    def test_require_sbi_raises_without_package(self):
        from src.models.heads import sbi_npe as mod
        if importlib.util.find_spec("sbi") is not None:
            self.skipTest("sbi is installed")
        with self.assertRaises(ImportError):
            mod.require_sbi()


class TestConfigSBIBlock(unittest.TestCase):
    def test_sbi_config_loads(self):
        cfg = load_config(ROOT / "configs" / "ts1_10yr_st_sbi_npe.yaml")
        self.assertEqual(cfg.head.type, "sbi_npe")
        self.assertEqual(cfg.head.sbi_npe.context_mode, "frozen_point")
        self.assertTrue(cfg.training.split_training)


if __name__ == "__main__":
    unittest.main()
