"""
PZDC Submission Wrappers — DeepSetZ entry.

Implements the three subtask functions required by the PZ Data Challenge:

  Subtask 1:  p(z) estimates already provided as qp HDF5 files.
              (run export_qp.py to generate them)

  Subtask 2:  run_tasksetN_estimation_only(model_file, test_file, output_file)
              Use a pre-trained checkpoint to estimate p(z) on a new test file.

  Subtask 3:  run_tasksetN_training_and_estimation(train_file, test_file, output_file)
              Train a new model from scratch, then estimate p(z) on the test file.

Usage (subtask 2 example)
--------------------------
    from scripts.submission import run_taskset1_estimation_only
    run_taskset1_estimation_only(
        model_file  = "outputs/pzdc_ts1/best_model.pt",
        test_file   = "data/pzdc/hdf5/pz_challenge_taskset_1_cardinal_test_10yr.hdf5",
        output_file = "submissions/pz_challenge_taskset_1_cardinal_pz_estimate_10yr.hdf5",
    )
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# Allow import from repo root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

# Map task-set number to (config_path, ckpt_path) defaults.
# Override these if you have a different run name.
_DEFAULT_CONFIGS = {
    1: ROOT / "configs" / "pzdc_ts1_10yr.yaml",
    2: ROOT / "configs" / "pzdc_ts2_10yr.yaml",
    3: ROOT / "configs" / "pzdc_ts3_10yr.yaml",
    4: ROOT / "configs" / "pzdc_ts4_1yr.yaml",
}

# Pre-trained checkpoints — update paths after training
_DEFAULT_CKPTS = {
    1: ROOT / "outputs" / "pzdc_ts1_cardinal_10yr" / "best_model.pt",
    2: ROOT / "outputs" / "pzdc_ts2_cardinal_10yr" / "best_model.pt",
    3: ROOT / "outputs" / "pzdc_ts3_cardinal_10yr" / "best_model.pt",
    4: ROOT / "outputs" / "pzdc_ts4_cardinal_1yr"  / "best_model.pt",
}


def _find_ckpt(run_dir: Path) -> Path:
    """Return best_model.pt if it exists, else final_model.pt."""
    for name in ("best_model.pt", "final_model.pt"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _run_export(
    config_path:  str | Path,
    ckpt_path:    str | Path,
    test_file:    str | Path,
    output_file:  str | Path,
) -> None:
    """Call export_qp.py as a subprocess to avoid import-time GPU init."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_qp.py"),
        "--config",  str(config_path),
        "--ckpt",    str(ckpt_path),
        "--test",    str(test_file),
        "--output",  str(output_file),
    ]
    result = subprocess.run(cmd, check=True, capture_output=False)
    return result


def _run_train(config_path: str | Path, output_dir: Path) -> Path:
    """Train a model with the given config; return the checkpoint path."""
    cmd = [
        sys.executable,
        str(ROOT / "src" / "train.py"),
        str(config_path),
    ]
    subprocess.run(cmd, check=True, capture_output=False, cwd=str(ROOT))
    # Find the run's output directory
    import yaml
    with open(config_path) as f:
        cfg_data = yaml.safe_load(f) or {}
    run_name  = cfg_data.get("run_name", "run")
    out_dir   = ROOT / cfg_data.get("output_dir", "outputs") / run_name
    return _find_ckpt(out_dir)


def _prepare_output_dir(output_file: str | Path) -> None:
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Subtask 2 — estimation only (pre-trained models)
# --------------------------------------------------------------------------

def run_taskset1_estimation_only(
    model_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """
    Estimate p(z) for Task Set 1 using a pre-trained DeepSetZ model.

    Parameters
    ----------
    model_file  : Path to a trained checkpoint (.pt file or directory containing one)
    test_file   : Path to the TS1 test HDF5 or parquet file
    output_file : Where to write the qp HDF5 submission file
    config_file : Path to the YAML config (defaults to configs/pzdc_ts1_10yr.yaml)
    """
    _prepare_output_dir(output_file)
    model_file = Path(model_file)
    ckpt = _find_ckpt(model_file) if model_file.is_dir() else model_file
    cfg  = config_file or _DEFAULT_CONFIGS[1]
    _run_export(cfg, ckpt, test_file, output_file)


def run_taskset2_estimation_only(
    model_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """Estimate p(z) for Task Set 2 using a pre-trained model."""
    _prepare_output_dir(output_file)
    model_file = Path(model_file)
    ckpt = _find_ckpt(model_file) if model_file.is_dir() else model_file
    cfg  = config_file or _DEFAULT_CONFIGS[2]
    _run_export(cfg, ckpt, test_file, output_file)


def run_taskset3_estimation_only(
    model_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """Estimate p(z) for Task Set 3 using a pre-trained model."""
    _prepare_output_dir(output_file)
    model_file = Path(model_file)
    ckpt = _find_ckpt(model_file) if model_file.is_dir() else model_file
    cfg  = config_file or _DEFAULT_CONFIGS[3]
    _run_export(cfg, ckpt, test_file, output_file)


def run_taskset4_estimation_only(
    model_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """Estimate p(z) for Task Set 4 using a pre-trained model."""
    _prepare_output_dir(output_file)
    model_file = Path(model_file)
    ckpt = _find_ckpt(model_file) if model_file.is_dir() else model_file
    cfg  = config_file or _DEFAULT_CONFIGS[4]
    _run_export(cfg, ckpt, test_file, output_file)


# --------------------------------------------------------------------------
# Subtask 3 — training + estimation
# --------------------------------------------------------------------------

def run_taskset1_training_and_estimation(
    train_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """
    Train a DeepSetZ model on TS1 training data, then estimate p(z) on the test set.

    NOTE: This will launch a full training run, which may take tens of minutes.
    Set appropriate hyperparameters in the config YAML before calling.
    """
    _prepare_output_dir(output_file)
    cfg = config_file or _DEFAULT_CONFIGS[1]

    # Patch the config's train_path to point to the provided train_file
    import yaml, copy
    with open(cfg) as f:
        cfg_data = yaml.safe_load(f) or {}
    cfg_data.setdefault("data", {})["train_path"] = str(train_file)

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        yaml.dump(cfg_data, tmp)
        tmp_cfg = Path(tmp.name)

    try:
        ckpt = _run_train(tmp_cfg, ROOT / "outputs")
        _run_export(tmp_cfg, ckpt, test_file, output_file)
    finally:
        tmp_cfg.unlink(missing_ok=True)


def run_taskset2_training_and_estimation(
    train_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """Train a DeepSetZ model on TS2 data, then estimate p(z)."""
    _prepare_output_dir(output_file)
    cfg = config_file or _DEFAULT_CONFIGS[2]
    import yaml
    with open(cfg) as f:
        cfg_data = yaml.safe_load(f) or {}
    cfg_data.setdefault("data", {})["train_path"] = str(train_file)
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        yaml.dump(cfg_data, tmp)
        tmp_cfg = Path(tmp.name)
    try:
        ckpt = _run_train(tmp_cfg, ROOT / "outputs")
        _run_export(tmp_cfg, ckpt, test_file, output_file)
    finally:
        tmp_cfg.unlink(missing_ok=True)


def run_taskset3_training_and_estimation(
    train_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """Train a DeepSetZ model on TS3 data, then estimate p(z)."""
    _prepare_output_dir(output_file)
    cfg = config_file or _DEFAULT_CONFIGS[3]
    import yaml
    with open(cfg) as f:
        cfg_data = yaml.safe_load(f) or {}
    cfg_data.setdefault("data", {})["train_path"] = str(train_file)
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        yaml.dump(cfg_data, tmp)
        tmp_cfg = Path(tmp.name)
    try:
        ckpt = _run_train(tmp_cfg, ROOT / "outputs")
        _run_export(tmp_cfg, ckpt, test_file, output_file)
    finally:
        tmp_cfg.unlink(missing_ok=True)


def run_taskset4_training_and_estimation(
    train_file:  str | Path,
    test_file:   str | Path,
    output_file: str | Path,
    config_file: str | Path | None = None,
) -> None:
    """Train a DeepSetZ model on TS4 data, then estimate p(z)."""
    _prepare_output_dir(output_file)
    cfg = config_file or _DEFAULT_CONFIGS[4]
    import yaml
    with open(cfg) as f:
        cfg_data = yaml.safe_load(f) or {}
    cfg_data.setdefault("data", {})["train_path"] = str(train_file)
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as tmp:
        yaml.dump(cfg_data, tmp)
        tmp_cfg = Path(tmp.name)
    try:
        ckpt = _run_train(tmp_cfg, ROOT / "outputs")
        _run_export(tmp_cfg, ckpt, test_file, output_file)
    finally:
        tmp_cfg.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# CLI smoke check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("DeepSetZ submission wrappers loaded successfully.")
    print("Available functions:")
    for fn in [
        "run_taskset1_estimation_only",  "run_taskset1_training_and_estimation",
        "run_taskset2_estimation_only",  "run_taskset2_training_and_estimation",
        "run_taskset3_estimation_only",  "run_taskset3_training_and_estimation",
        "run_taskset4_estimation_only",  "run_taskset4_training_and_estimation",
    ]:
        print(f"  {fn}")
