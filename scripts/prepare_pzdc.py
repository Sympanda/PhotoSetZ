"""
Prepare PZDC training/test splits for DeepSetZ.

The PZDC "test" parquets are the unlabelled challenge submission sets — they
have no redshift column.  This script takes the labelled training parquet for
each task set, does a stratified redshift split (80k train / 20k test), and
writes them into data/pzdc/ so DeepSetZ training can proceed.

Usage
-----
    # Prepare all available task sets automatically
    python scripts/prepare_pzdc.py

    # Or specify a single file
    python scripts/prepare_pzdc.py --file data/pzdc/pz_challenge_taskset_1_cardinal_training_10yr.parquet

Output files (example for TS1 cardinal 10yr)
---------------------------------------------
    data/pzdc/ts1_cardinal_10yr_train.parquet   (80k rows)
    data/pzdc/ts1_cardinal_10yr_test.parquet    (20k rows)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent
PZDC    = ROOT / "data" / "pzdc"
TARGET  = "redshift"
N_BINS  = 20          # redshift bins for stratification
TEST_FRAC = 0.20      # 20% held-out test


def stratified_split(df: pd.DataFrame, z_col: str = TARGET,
                     test_frac: float = TEST_FRAC,
                     n_bins: int = N_BINS,
                     seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/test with approximately uniform redshift coverage
    in both halves (same as scripts/prepare_data.py).
    """
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)

    bins = pd.qcut(df[z_col], q=n_bins, duplicates="drop")
    train_idx, test_idx = [], []

    for _, grp in df.groupby(bins, observed=True):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_df = df.loc[sorted(train_idx)].reset_index(drop=True)
    test_df  = df.loc[sorted(test_idx)].reset_index(drop=True)
    return train_df, test_df


def make_short_name(src: Path) -> str:
    """
    pz_challenge_taskset_1_cardinal_training_10yr → ts1_cardinal_10yr
    """
    m = re.search(r"taskset_(\d+)_(\w+?)_training_(\w+)", src.name)
    if m:
        return f"ts{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return src.stem


def prepare_file(src: Path) -> tuple[Path, Path]:
    df = pd.read_parquet(src)

    if TARGET not in df.columns:
        raise ValueError(
            f"{src.name} has no '{TARGET}' column — this looks like the "
            "unlabelled challenge submission file, not the training file."
        )

    n = len(df)
    short = make_short_name(src)
    n_test  = int(n * TEST_FRAC)
    n_train = n - n_test

    print(f"  {src.name}")
    print(f"    {n:,} rows  →  train: {n_train:,}  test: {n_test:,}")
    print(f"    z range [{df[TARGET].min():.3f}, {df[TARGET].max():.3f}]  "
          f"mean={df[TARGET].mean():.3f}")

    train_df, test_df = stratified_split(df, z_col=TARGET,
                                         test_frac=TEST_FRAC, n_bins=N_BINS)

    dst_train = PZDC / f"{short}_train.parquet"
    dst_test  = PZDC / f"{short}_test.parquet"
    train_df.to_parquet(dst_train, index=False)
    test_df.to_parquet(dst_test,  index=False)

    # Quick split summary
    for label, part_df in [("train", train_df), ("test", test_df)]:
        z = part_df[TARGET]
        print(f"    {label}: n={len(part_df):,}  mean={z.mean():.3f}  "
              f"std={z.std():.3f}  [{z.min():.3f}, {z.max():.3f}]")

    return dst_train, dst_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PZDC train/test splits")
    parser.add_argument("--file", default=None,
                        help="Specific training parquet to split (default: all in data/pzdc/)")
    args = parser.parse_args()

    print("=" * 65)
    print("  PZDC — stratified train / test split")
    print(f"  Test fraction: {TEST_FRAC:.0%}  |  Redshift bins: {N_BINS}")
    print("=" * 65 + "\n")

    if args.file:
        sources = [Path(args.file)]
    else:
        # Find all labelled training parquets (skip already-split files)
        sources = sorted(
            p for p in PZDC.glob("pz_challenge_taskset_*_training_*.parquet")
            if "_train.parquet" not in p.name and "_test.parquet" not in p.name
        )

    if not sources:
        print(f"No training parquets found in {PZDC}.")
        print("Run scripts/download_pzdc.py first.")
        return

    created = []
    for src in sources:
        try:
            tr, te = prepare_file(src)
            created.extend([tr, te])
            print()
        except ValueError as e:
            print(f"  SKIP: {e}\n")

    if not created:
        return

    print("=" * 65)
    print("  Done.  Files created:")
    for f in created:
        print(f"    {f.relative_to(ROOT)}")
    print()
    print("Update your config:")
    print("  data:")
    # Print first pair as example
    train_files = [f for f in created if "_train.parquet" in f.name]
    test_files  = [f for f in created if "_test.parquet"  in f.name]
    if train_files:
        print(f"    train_path: data/pzdc/{train_files[0].name}")
    if test_files:
        print(f"    test_path:  data/pzdc/{test_files[0].name}")


if __name__ == "__main__":
    main()
