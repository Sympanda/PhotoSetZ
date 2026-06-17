"""Stage-2 model: frozen encoder + MLP point head + trainable SBI/NPE density head."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads.mlp_regressor import MLPRegressor
from src.models.heads.sbi_context import SBIContextBuilder
from src.models.heads.sbi_npe import SBINPEHead


class SBIStage2Model(nn.Module):
    """
    Two-stage PhotoSetZ compressor + SBI density estimator.

    Stage 1 (encoder + MLP) is frozen by default.  Stage 2 trains only
    ``SBIContextBuilder`` + ``SBINPEHead``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        point_head: MLPRegressor,
        sbi_head: SBINPEHead,
        context_builder: SBIContextBuilder,
        *,
        bottleneck: nn.Module | None = None,
        use_coverage: bool = False,
        n_total_filters: int = 1,
        log_target: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.point_head = point_head
        self.sbi_head = sbi_head
        self.context_builder = context_builder
        self.use_coverage = use_coverage
        self.n_total_filters = max(int(n_total_filters), 1)
        self.log_target = log_target

    @property
    def head(self) -> SBINPEHead:
        """Alias for code paths that inspect ``model.head``."""
        return self.sbi_head

    def _coverage_scalar(self, key_padding_mask: torch.Tensor) -> torch.Tensor:
        n_active = (~key_padding_mask).float().sum(dim=-1, keepdim=True)
        return n_active / self.n_total_filters

    def _encode(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(tokens, key_padding_mask)
        if self.bottleneck is not None:
            h = self.bottleneck(h)
        return h

    def _point_embedding(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        emb = h
        if self.use_coverage:
            emb = torch.cat([emb, self._coverage_scalar(key_padding_mask)], dim=-1)
        return emb

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        z_true: torch.Tensor | None = None,
    ) -> dict:
        h = self._encode(tokens, key_padding_mask)
        point_emb = self._point_embedding(h, key_padding_mask)
        point_out = self.point_head(point_emb)
        y_hat_log = point_out["z_pred"]

        context = self.context_builder(h, tokens, key_padding_mask, y_hat_log)

        z_phys = None
        if z_true is not None:
            z_phys = torch.expm1(z_true) if self.log_target else z_true

        return self.sbi_head(
            context,
            z_true=z_phys,
            z_pred_point=y_hat_log,
        )

    def trainable_parameters(self):
        for mod in (self.context_builder, self.sbi_head):
            yield from mod.parameters()

    def freeze_compressor(self) -> None:
        for mod in (self.encoder, self.bottleneck, self.point_head):
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
