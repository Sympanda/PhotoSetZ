"""
Download and convert the DESC PZ Data Challenge datasets.

This script:
  1. Downloads public.tgz directly from NERSC (no pz_data_challenge install needed)
  2. Extracts HDF5 files into data/pzdc/hdf5/
  3. Converts each HDF5 to parquet in data/pzdc/
  4. Prints a column summary so you can verify filter names

Usage
-----
    python scripts/download_pzdc.py [--taskset 1]

After running, train with:
    python src/train.py configs/pzdc_taskset1.yaml
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Only Task Sets 1 and 2 are currently in the public archive.
# Task Sets 3 and 4 are documented but have not yet been released publicly.
PUBLIC_URL = "https://portal.nersc.gov/cfs/lsst/PZ/data_challenge/public.tgz"
_AVAILABLE_TASKSETS = {1, 2}

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "data" / "pzdc"
HDF_DIR  = OUT_DIR / "hdf5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HDF_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Download + extract ────────────────────────────────────────────────────

def download_and_extract(taskset: int | str) -> list[Path]:
    """
    Download public.tgz (once) and extract HDF5 files to HDF_DIR.

    Parameters
    ----------
    taskset : int or "all"
        Which task set(s) to extract.  Use "all" to extract every task set
        in a single download pass — much faster than running separately.
    """
    # Build the list of prefixes to extract
    if taskset == "all":
        prefixes = [f"pz_challenge_taskset_{n}_" for n in range(1, 5)]
        label = "all task sets"
    else:
        prefixes = [f"pz_challenge_taskset_{taskset}_"]
        label = f"task set {taskset}"

    # Check if all files already present (skip download)
    all_present = all(
        any(HDF_DIR.glob(f"{pfx}*.hdf5"))
        for pfx in prefixes
    )
    if all_present:
        print(f"All {label} HDF5 files already present in {HDF_DIR} — skipping download.")
        extracted = sorted(
            f for pfx in prefixes
            for f in HDF_DIR.glob(f"{pfx}*.hdf5")
        )
        return extracted

    print(f"Downloading {PUBLIC_URL} …")
    print("(~147 MB — may take a minute or two)\n")

    def _progress(count, block_size, total_size):
        pct = min(count * block_size / total_size * 100, 100)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urllib.request.urlretrieve(PUBLIC_URL, tmp_path, reporthook=_progress)
        print()  # newline after progress bar
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"\nERROR: Download failed — {exc}")
        print("\nTry downloading manually:")
        print(f"  curl -L -O {PUBLIC_URL}")
        print(f"  tar -xzf public.tgz -C {HDF_DIR} --strip-components=1")
        sys.exit(1)

    print(f"Extracting {label} HDF5 files …")
    extracted: list[Path] = []

    with tarfile.open(tmp_path, "r:gz") as tar:
        for member in tar.getmembers():
            fname = Path(member.name).name
            if not fname.endswith(".hdf5"):
                continue
            if not any(fname.startswith(pfx) for pfx in prefixes):
                continue
            dst = HDF_DIR / fname
            if dst.exists():
                print(f"  Skipping (exists): {fname}")
                extracted.append(dst)
                continue
            print(f"  Extracting: {fname}")
            fobj = tar.extractfile(member)
            if fobj:
                dst.write_bytes(fobj.read())
                extracted.append(dst)

    tmp_path.unlink(missing_ok=True)

    if not extracted:
        print(f"ERROR: No matching HDF5 files found in the archive for {label}.")
        print("Check that the taskset number is correct (1–4).")
        sys.exit(1)

    print(f"Extracted {len(extracted)} file(s) to {HDF_DIR}\n")
    return sorted(extracted)


# ── 2. Convert HDF5 → parquet ────────────────────────────────────────────────

def convert_hdf5(src: Path) -> Path:
    """Convert a single HDF5 file to parquet and return the parquet path."""
    try:
        import h5py
        import pandas as pd
    except ImportError:
        print("h5py is required for conversion.  Install with:")
        print("  pip install h5py")
        sys.exit(1)

    dst = OUT_DIR / src.with_suffix(".parquet").name
    if dst.exists():
        print(f"  Skipping (already exists): {dst.name}")
        return dst

    print(f"  {src.name}  →  {dst.name}")
    with h5py.File(src, "r") as f:
        # Data may be at root level or inside a group
        groups = [k for k in f.keys() if isinstance(f[k], h5py.Group)]
        grp = f[groups[0]] if groups else f

        cols: dict = {}
        for key in grp.keys():
            try:
                arr = grp[key][:]
                if arr.ndim == 1:
                    cols[key] = arr
            except Exception:
                pass

    import pandas as pd
    df = pd.DataFrame(cols)
    df.to_parquet(dst, index=False)
    print(f"    {len(df):,} rows  ×  {len(df.columns)} columns")
    return dst


# ── 3. Print column summary ──────────────────────────────────────────────────

def print_column_summary(parquet_files: list[Path]) -> None:
    import pandas as pd

    training = [f for f in parquet_files if "training" in f.name]
    sample   = training[0] if training else parquet_files[0]
    df       = pd.read_parquet(sample)

    mag_cols  = sorted(c for c in df.columns if "mag_" in c and not c.endswith("_err"))
    err_cols  = sorted(c for c in df.columns if c.endswith("_err"))
    other     = sorted(c for c in df.columns if c not in mag_cols + err_cols)

    print("=" * 65)
    print(f"  Column summary: {sample.name}  ({len(df):,} rows)")
    print("=" * 65)
    print(f"\nMagnitude columns ({len(mag_cols)}):")
    for c in mag_cols:
        null_pct = df[c].isna().mean() * 100
        print(f"  {c:<35}  {null_pct:.1f}% NaN")
    print(f"\nError columns ({len(err_cols)}):")
    for c in err_cols:
        print(f"  {c}")
    print(f"\nOther columns: {other}")

    # Suggested config
    train_files = [f for f in parquet_files if "training" in f.name]
    test_files  = [f for f in parquet_files if "test" in f.name]
    print("\n" + "=" * 65)
    print("  Suggested config  (configs/pzdc_taskset1.yaml)")
    print("=" * 65)
    if train_files:
        print(f"\n  train_path: data/pzdc/{train_files[0].name}")
    if test_files:
        print(f"  test_path:  data/pzdc/{test_files[0].name}")

    # Detect which Roman column naming convention is used
    roman_cols = [c for c in mag_cols if "roman" in c.lower()]
    lsst_cols  = [c for c in mag_cols if "lsst"  in c.lower()]
    surveys = []
    if lsst_cols:
        surveys.append("lsst")
    if roman_cols:
        surveys.append("roman")
    print(f"  target_col: redshift")
    print(f"  active_surveys: {surveys}")
    print(f"  include_errors: {'true' if err_cols else 'false'}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download DESC PZ Data Challenge data")
    parser.add_argument("--taskset", default="1",
                        help="Which task set to download: 1, 2, 3, 4, or 'all' (default: 1)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download; convert any existing HDF5 files in data/pzdc/hdf5/")
    args = parser.parse_args()

    taskset = args.taskset if args.taskset == "all" else int(args.taskset)
    label   = "All Task Sets" if taskset == "all" else f"Task Set {taskset}"

    # Warn if requesting a task set not yet in the public archive
    requested = set(range(1, 5)) if taskset == "all" else {taskset}
    unavailable = requested - _AVAILABLE_TASKSETS
    if unavailable:
        print(f"NOTE: Task Set(s) {sorted(unavailable)} are not yet in the public archive.")
        print(f"      Only Task Sets {sorted(_AVAILABLE_TASKSETS)} are currently available.")
        print(f"      TS3/4 configs are kept as placeholders for when data is released.\n")
        # Filter to only available task sets
        requested = requested & _AVAILABLE_TASKSETS
        if not requested:
            print("Nothing to download.")
            return
        taskset = sorted(requested)[0] if len(requested) == 1 else "all"
        label   = "All available Task Sets" if taskset == "all" else f"Task Set {taskset}"

    print("=" * 65)
    print(f"  DESC PZ Data Challenge — {label}")
    print("=" * 65 + "\n")

    if args.skip_download:
        glob = "pz_challenge_taskset_*.hdf5" if taskset == "all" \
               else f"pz_challenge_taskset_{taskset}_*.hdf5"
        hdf5_files = sorted(HDF_DIR.glob(glob))
        if not hdf5_files:
            print(f"No HDF5 files found in {HDF_DIR}. Remove --skip-download to fetch them.")
            sys.exit(1)
    else:
        hdf5_files = download_and_extract(taskset)

    print(f"\nConverting {len(hdf5_files)} HDF5 file(s) to parquet …")
    parquets = [convert_hdf5(f) for f in sorted(hdf5_files)]

    print()
    print_column_summary(parquets)
    print(f"Done.  Parquet files are in  {OUT_DIR}\n")
    print("Run training with:")
    print("  python src/train.py configs/pzdc_taskset1.yaml\n")


if __name__ == "__main__":
    main()
