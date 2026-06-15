"""
Prepare stratified train/test parquet files for DeepSetZ.

Combines the two existing 100k catalogues into one 200k pool, then
performs a stratified split (equal-frequency redshift bins) to produce:

    data/ellen/train_175k.parquet   —  175,000 galaxies
    data/ellen/test_25k.parquet     —   25,000 galaxies

Stratification ensures each redshift stratum is represented in the test
set at roughly the same proportion as in the full catalogue, so tail
performance statistics are reliable.

Usage
-----
    python scripts/prepare_data.py
    python scripts/prepare_data.py --n_train 160000 --n_test 40000
    python scripts/prepare_data.py --n_bins 50 --seed 123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root
ROOT = Path(__file__).parent.parent


def stratified_split(
    df: pd.DataFrame,
    z_col: str,
    n_test: int,
    n_bins: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into (train, test) with proportional representation across
    *n_bins* equal-frequency redshift bins.

    The exact test size is guaranteed to equal *n_test* by distributing
    any rounding remainder to/from the largest bins.
    """
    rng = np.random.default_rng(seed)
    n_total = len(df)
    test_frac = n_test / n_total

    # Equal-frequency binning (handles skewed z distributions)
    df = df.copy()
    df["_zbin"] = pd.qcut(df[z_col], q=n_bins, labels=False, duplicates="drop")

    # How many test samples from each bin?
    bin_sizes   = df.groupby("_zbin", observed=True).size()
    raw_counts  = bin_sizes * test_frac
    floor_counts = np.floor(raw_counts).astype(int)

    # Distribute the rounding remainder to the largest bins first
    remainder = n_test - floor_counts.sum()
    if remainder > 0:
        top_bins = (raw_counts - floor_counts).nlargest(int(remainder)).index
        floor_counts[top_bins] += 1

    test_indices: list[int] = []
    for bin_id, n_take in floor_counts.items():
        bin_idx = df.index[df["_zbin"] == bin_id].tolist()
        chosen  = rng.choice(bin_idx, size=min(n_take, len(bin_idx)), replace=False)
        test_indices.extend(chosen.tolist())

    test_df  = df.loc[test_indices].drop(columns=["_zbin"])
    train_df = df.drop(index=test_indices).drop(columns=["_zbin"])

    # Shuffle both
    test_df  = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, test_df


def print_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    z_col: str,
    n_bins: int = 5,
) -> None:
    train_tagged = train_df[[z_col]].copy(); train_tagged["_split"] = "train"
    test_tagged  = test_df[[z_col]].copy();  test_tagged["_split"]  = "test"
    full = pd.concat([train_tagged, test_tagged], ignore_index=True)
    full["_zbin"] = pd.qcut(full[z_col], q=n_bins, duplicates="drop")

    print(f"\n{'Redshift range':<22} {'Total':>8} {'Train':>8} {'Test':>8} {'Test%':>7}")
    print("─" * 56)
    for label, group in full.groupby("_zbin", observed=True):
        n_tot = len(group)
        n_tr  = (group["_split"] == "train").sum()
        n_te  = (group["_split"] == "test").sum()
        print(f"  {str(label):<20} {n_tot:>8,} {n_tr:>8,} {n_te:>8,} {n_te/n_tot*100:>6.1f}%")

    print("─" * 56)
    print(f"  {'TOTAL':<20} {len(full):>8,} {len(train_df):>8,} {len(test_df):>8,} "
          f"{len(test_df)/len(full)*100:>6.1f}%")

    print(f"\n  z range  — full:  [{full[z_col].min():.3f}, {full[z_col].max():.3f}]")
    print(f"  z range  — train: [{train_df[z_col].min():.3f}, {train_df[z_col].max():.3f}]")
    print(f"  z range  — test:  [{test_df[z_col].min():.3f}, {test_df[z_col].max():.3f}]")
    print(f"  z median — train: {train_df[z_col].median():.4f}")
    print(f"  z median — test:  {test_df[z_col].median():.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified train/test split of DeepSetZ galaxy catalogues."
    )
    parser.add_argument("--train_a",  default="data/ellen/train_100k.parquet",
                        help="First input parquet (default: train_100k)")
    parser.add_argument("--train_b",  default="data/ellen/test_100k.parquet",
                        help="Second input parquet (default: test_100k)")
    parser.add_argument("--out_dir",  default="data/ellen",
                        help="Output directory")
    parser.add_argument("--n_train",  type=int, default=175_000,
                        help="Number of training galaxies (default: 175000)")
    parser.add_argument("--n_test",   type=int, default=25_000,
                        help="Number of test galaxies (default: 25000)")
    parser.add_argument("--n_bins",   type=int, default=20,
                        help="Number of redshift bins for stratification (default: 20)")
    parser.add_argument("--z_col",    default="true_redshift",
                        help="Redshift column name (default: true_redshift)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    path_a = ROOT / args.train_a
    path_b = ROOT / args.train_b
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = args.n_train + args.n_test

    print(f"Loading {path_a.name} …")
    df_a = pd.read_parquet(path_a)
    print(f"  {len(df_a):,} rows")

    print(f"Loading {path_b.name} …")
    df_b = pd.read_parquet(path_b)
    print(f"  {len(df_b):,} rows")

    df = pd.concat([df_a, df_b], ignore_index=True)
    print(f"\nCombined: {len(df):,} rows")

    if len(df) < n_total:
        print(f"ERROR: combined catalogue ({len(df):,}) is smaller than "
              f"n_train+n_test ({n_total:,}). Reduce --n_train or --n_test.")
        sys.exit(1)

    # Subsample to exactly n_total if the pool is larger
    if len(df) > n_total:
        df = df.sample(n=n_total, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {n_total:,} rows for the split.")

    print(f"\nStratified split → train={args.n_train:,}  test={args.n_test:,}  "
          f"bins={args.n_bins}  seed={args.seed}")

    train_df, test_df = stratified_split(
        df, z_col=args.z_col, n_test=args.n_test,
        n_bins=args.n_bins, seed=args.seed,
    )

    print_split_summary(train_df, test_df, z_col=args.z_col, n_bins=5)

    # Save
    n_tr = len(train_df)
    n_te = len(test_df)
    train_name = f"train_{n_tr//1000}k.parquet"
    test_name  = f"test_{n_te//1000}k.parquet"
    train_path = out_dir / train_name
    test_path  = out_dir / test_name

    print(f"Saving {train_path} …")
    train_df.to_parquet(train_path, index=False)
    print(f"Saving {test_path} …")
    test_df.to_parquet(test_path, index=False)
    print("\nDone.")
    print(f"\nUpdate your config:\n"
          f"  data:\n"
          f"    train_path: {args.out_dir}/{train_name}\n"
          f"    test_path:  {args.out_dir}/{test_name}")


if __name__ == "__main__":
    main()
