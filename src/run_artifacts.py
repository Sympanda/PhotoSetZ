"""
Standard paths and discovery helpers for DeepSetZ training runs.

Layout (single run directory — one config name, multiple artefacts)
--------------------------------------------------------------------
outputs/<run_name>/
    config.yaml
    best_model.pt              # end-to-end training
    best_point.pt              # split training — stage 1 (encoder + MLP)
    best_posterior.pt          # split training — stage 2 (encoder + MDN/NSF/…)
    calibration/
        post_hoc.json          # fitted σ scale factor (no duplicate checkpoint)
    history.json               # end-to-end or stage 2
    history_stage1.json
    test_metrics.json
    test_metrics_point.json
    test_metrics_posterior.json
    plots/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Checkpoint filenames
CKPT_END_TO_END  = "best_model.pt"
CKPT_POINT       = "best_point.pt"
CKPT_POSTERIOR   = "best_posterior.pt"

CALIB_DIR        = "calibration"
POST_HOC_FILE    = "calibration/post_hoc.json"

# Model roles exposed to the evaluation notebook
ROLE_END_TO_END  = "end_to_end"
ROLE_POINT       = "point"
ROLE_POSTERIOR   = "posterior"
ROLE_CALIBRATED  = "posterior_calibrated"


def run_is_complete(run_dir: Path) -> bool:
    """True if the directory looks like a finished DeepSetZ run."""
    if not (run_dir / "config.yaml").exists():
        return False
    return any((run_dir / name).exists() for name in (
        CKPT_END_TO_END, CKPT_POINT, CKPT_POSTERIOR,
    ))


def available_roles(run_dir: Path) -> List[str]:
    """Return model roles available for interactive evaluation."""
    roles: List[str] = []
    if (run_dir / CKPT_POINT).exists():
        roles.append(ROLE_POINT)
    if (run_dir / CKPT_POSTERIOR).exists():
        roles.append(ROLE_POSTERIOR)
        if (run_dir / POST_HOC_FILE).exists():
            roles.append(ROLE_CALIBRATED)
    if (run_dir / CKPT_END_TO_END).exists():
        roles.append(ROLE_END_TO_END)
        if (run_dir / POST_HOC_FILE).exists() and ROLE_CALIBRATED not in roles:
            roles.append(ROLE_CALIBRATED)
    return roles


def role_label(role: str) -> str:
    return {
        ROLE_POINT:      "Point (MLP)",
        ROLE_POSTERIOR:  "Posterior (PDF)",
        ROLE_CALIBRATED: "Posterior + post-hoc σ",
        ROLE_END_TO_END: "End-to-end",
    }.get(role, role)


def checkpoint_path(run_dir: Path, role: str) -> Optional[Path]:
    """Resolve checkpoint file for a model role."""
    mapping = {
        ROLE_POINT:      CKPT_POINT,
        ROLE_POSTERIOR:  CKPT_POSTERIOR,
        ROLE_CALIBRATED: CKPT_POSTERIOR,
        ROLE_END_TO_END: CKPT_END_TO_END,
    }
    name = mapping.get(role)
    if name is None:
        return None
    path = run_dir / name
    return path if path.exists() else None


def load_post_hoc(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load post-hoc calibration JSON if present."""
    path = run_dir / POST_HOC_FILE
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def post_hoc_sigma_scale(run_dir: Path) -> Optional[float]:
    data = load_post_hoc(run_dir)
    if data is None:
        return None
    return float(data.get("sigma_scale", 1.0))


def save_post_hoc(run_dir: Path, payload: Dict[str, Any]) -> Path:
    calib_dir = run_dir / CALIB_DIR
    calib_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / POST_HOC_FILE
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path
