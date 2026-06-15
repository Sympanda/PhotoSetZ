"""
Regenerate all post-training plots for a completed run.

Usage
-----
    python src/plot_run.py outputs/01_deepsets_mlp
    python src/plot_run.py outputs/01_deepsets_mlp --run_name my_label
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plot import generate_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate plots from a saved DeepSetZ run directory."
    )
    parser.add_argument(
        "run_dir",
        help="Path to the run output directory (e.g. outputs/01_deepsets_mlp)",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Label for plot titles. Defaults to the run directory name.",
    )
    args = parser.parse_args()

    run_dir  = Path(args.run_dir)
    run_name = args.run_name or run_dir.name

    # Check required files exist
    required = ["history.json", "predictions.npz", "subset_metrics.json"]
    missing  = [f for f in required if not (run_dir / f).exists()]
    if missing:
        print(f"ERROR: missing files in {run_dir}: {missing}")
        print("Run the full training first, or check the path.")
        sys.exit(1)

    history        = json.loads((run_dir / "history.json").read_text())
    subset_results = json.loads((run_dir / "subset_metrics.json").read_text())
    data           = np.load(run_dir / "predictions.npz")

    prob_outputs = None
    if "z_median" in data and "pit" in data:
        prob_outputs = {
            "z_mean":   data["z_mean"],
            "z_median": data["z_median"],
            "z_mode":   data["z_mode"],
            "pit":      data["pit"],
            "head_type": "probabilistic",
        }

    generate_all_plots(
        history        = history,
        z_true         = data["z_true"],
        z_pred         = data["z_pred"],
        subset_results = subset_results,
        out_dir        = run_dir,
        run_name       = run_name,
        prob_outputs   = prob_outputs,
    )


if __name__ == "__main__":
    main()
