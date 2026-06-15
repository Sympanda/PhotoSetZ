#!/usr/bin/env python3
"""
Train flat-vector benchmark baselines (MLP + MDN) for comparison with DeepSetZ.

Runs a progressive filter-subset ladder for each dataset:
  DECaLS 3-band → DECaLS 4-band → LSST → LSST+Roman → … → all surveys

Datasets (default): ellen, ts1 (PZDC Task Set 1), ts2 (PZDC Task Set 2)

Outputs are saved to benchmarks/<run_name>/ with the same plots/metrics as
main training runs.

Usage
-----
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --datasets ellen ts1
    python scripts/run_benchmarks.py --models mlp
    python scripts/run_benchmarks.py --subsets decals_3 lsst lsst_roman
    python scripts/run_benchmarks.py --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.benchmarks.subsets import BENCHMARK_DATASETS, get_subset_ladder
from src.benchmarks.train import (
    make_benchmark_config,
    resolve_subset_columns,
    train_benchmark,
)

MODEL_TYPES = {
    "mlp": "flat_mlp",
    "mdn": "flat_mdn",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train flat-vector benchmark baselines")
    p.add_argument(
        "--datasets", nargs="+", default=["ellen", "ts1", "ts2"],
        choices=list(BENCHMARK_DATASETS.keys()),
        help="Which datasets to benchmark (default: ellen ts1 ts2)",
    )
    p.add_argument(
        "--models", nargs="+", default=["mlp", "mdn"],
        choices=list(MODEL_TYPES.keys()),
        help="Benchmark model types (default: mlp mdn)",
    )
    p.add_argument(
        "--subsets", nargs="+", default=None,
        help="Only run these subset names (default: full ladder per dataset)",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs where benchmarks/<run_name>/best_model.pt already exists",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs (default: 80)",
    )
    p.add_argument(
        "--patience", type=int, default=None,
        help="Override early-stop patience (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jobs = []

    for ds_key in args.datasets:
        spec   = BENCHMARK_DATASETS[ds_key]
        ladder = get_subset_ladder(ds_key)

        for subset_name, requested_cols in ladder:
            if args.subsets and subset_name not in args.subsets:
                continue
            try:
                cols = resolve_subset_columns(spec, subset_name, requested_cols)
            except (FileNotFoundError, ValueError) as exc:
                print(f"SKIP {ds_key}/{subset_name}: {exc}")
                continue

            for model_key in args.models:
                model_type = MODEL_TYPES[model_key]
                cfg = make_benchmark_config(spec, subset_name, cols, model_type)
                if args.epochs is not None:
                    cfg.training.epochs = args.epochs
                if args.patience is not None:
                    cfg.training.early_stop_patience = args.patience
                jobs.append(cfg)

    if not jobs:
        print("No benchmark jobs to run.")
        return

    print(f"\nBenchmark queue: {len(jobs)} runs")
    print(f"  Datasets : {args.datasets}")
    print(f"  Models   : {args.models}")
    print(f"  Output   : benchmarks/\n")

    n_failed = 0
    for i, cfg in enumerate(jobs, 1):
        out_dir = ROOT / cfg.output_dir / cfg.run_name
        ckpt    = out_dir / "best_model.pt"

        if args.skip_existing and ckpt.exists():
            print(f"[{i}/{len(jobs)}] SKIP (exists) {cfg.run_name}")
            continue

        print(f"[{i}/{len(jobs)}] {cfg.run_name}")
        try:
            train_benchmark(cfg)
        except Exception as exc:
            n_failed += 1
            print(f"  FAILED {cfg.run_name}: {exc}")
            continue

    if n_failed:
        print(f"\nBenchmark queue finished with {n_failed} failure(s).")
    else:
        print("\nAll benchmark runs complete.")


if __name__ == "__main__":
    main()
