"""Redshift target transforms for SBI / NPE density heads."""

from __future__ import annotations

import torch


class RedshiftTransform:
    """
    Map physical redshift ``z`` to an unconstrained target ``t`` for density estimation.

    Supported modes
    ---------------
    log1p_logit : z -> y=log(1+z) -> u in (0,1) -> t=logit(u)   [recommended]
    log1p       : z -> y=log(1+z)
    z_logit     : z -> u in (0,1) -> t=logit(u)
    z           : t = z (diagnostic only)
    """

    def __init__(
        self,
        mode: str = "log1p_logit",
        y_min: float = 0.0,
        y_max: float = 1.45,
        eps: float = 1e-5,
        z_max: float | None = None,
    ) -> None:
        self.mode = mode
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.eps = float(eps)
        self.z_max = z_max

    @property
    def y_range(self) -> float:
        return self.y_max - self.y_min

    def z_to_y(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode in ("log1p_logit", "log1p"):
            return torch.log1p(z.clamp(min=0.0))
        return z

    def y_to_z(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode in ("log1p_logit", "log1p"):
            return torch.expm1(y)
        return y

    def y_to_t(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "log1p_logit":
            u = (y - self.y_min) / self.y_range
            u = u.clamp(self.eps, 1.0 - self.eps)
            return torch.log(u / (1.0 - u))
        if self.mode == "z_logit":
            z_max = self.z_max if self.z_max is not None else self.y_max
            u = (y / z_max).clamp(self.eps, 1.0 - self.eps)
            return torch.log(u / (1.0 - u))
        return y

    def t_to_y(self, t: torch.Tensor) -> torch.Tensor:
        if self.mode in ("log1p_logit", "z_logit"):
            u = torch.sigmoid(t)
            base = self.y_min if self.mode == "log1p_logit" else 0.0
            span = self.y_range if self.mode == "log1p_logit" else (self.z_max or self.y_max)
            return base + u * span
        return t

    def z_to_t(self, z: torch.Tensor) -> torch.Tensor:
        return self.y_to_t(self.z_to_y(z))

    def t_to_z(self, t: torch.Tensor) -> torch.Tensor:
        return self.y_to_z(self.t_to_y(t))

    def log_abs_det_dt_dz(self, z: torch.Tensor) -> torch.Tensor:
        """log |dt/dz| for change-of-variables when converting p_t -> p_z."""
        if self.mode == "log1p_logit":
            y = torch.log1p(z.clamp(min=0.0))
            u = ((y - self.y_min) / self.y_range).clamp(self.eps, 1.0 - self.eps)
            det = self.y_range * u * (1.0 - u) * (1.0 + z.clamp(min=0.0))
            return -torch.log(det.clamp(min=1e-12))
        if self.mode == "log1p":
            return -torch.log1p(z.clamp(min=0.0)).clamp(min=1e-12)
        if self.mode == "z_logit":
            z_max = self.z_max if self.z_max is not None else self.y_max
            u = (z / z_max).clamp(self.eps, 1.0 - self.eps)
            det = z_max * u * (1.0 - u)
            return -torch.log(det.clamp(min=1e-12))
        return torch.zeros_like(z)
