# DeepSetZ — Set-Based Photometric Redshift Estimation

A flexible neural framework for photometric redshift inference from variable-length sets of filter measurements, inspired by SED fitting but replacing explicit template matching with a learned posterior model.

Two encoder architectures are provided — a classic **DeepSets** model and a **Set Transformer** — paired with four swappable prediction heads. All models treat each filter measurement as a token, so they naturally handle missing filters, heterogeneous survey combinations, and evolving filter coverage without architectural changes.

---

## Background

Each galaxy is represented as a variable-length set of photometric tokens, one per available filter:

```
x_i = [ m_i, λ_eff_i, Δλ_i, survey_id_i ]           # token_dim = 4
x_i = [ m_i, λ_eff_i, Δλ_i, survey_id_i, σ_m_i ]    # token_dim = 5, with errors
```

where `m_i` is the normalised AB magnitude, `λ_eff` and `Δλ` are the filter's effective wavelength and RMS bandwidth (computed from transmission curves), and `survey_id` identifies the instrument. An optional fifth feature `σ_m` carries the per-filter magnitude error. The model learns:

```
p(z | X_g) = p(z | {m_i, λ_eff_i, Δλ_i}_{i ∈ S_g})
```

without requiring a fixed input vector — absent filters are simply not in the set. By default, non-detections (NaN magnitudes) are dropped; opt-in SED mode can keep them as flagged tokens (see [SED-like filter handling](#sed-like-filter-handling-opt-in)).

---

## Project Structure

```
DeepSetZ/
├── configs/
│   ├── deepsets.yaml             # DeepSets + MLP head (Ellen data)
│   ├── deepsets_nsf.yaml         # DeepSets + Neural Spline Flow head
│   ├── set_transformer.yaml      # Set Transformer + MLP head
│   ├── ts1_10yr.yaml             # PZDC Task Set 1, 10-year depth (DeepSets)
│   ├── ts1_10yr_st.yaml          # TS1 10yr, Set Transformer + MDN (end-to-end)
│   ├── ts1_10yr_st_2part.yaml     # TS1 10yr, two-stage MLP → MDN
│   ├── ts1_10yr_st_nsf.yaml       # TS1 10yr, Set Transformer + NSF (end-to-end)
│   ├── ts1_10yr_st_nsf_2part.yaml # TS1 10yr, two-stage MLP → NSF
│   ├── ts1_10yr_st_nsf_sed.yaml   # TS1 10yr, NSF + SED-like tokens / coverage / bottleneck
│   ├── ts1_1yr.yaml / ts1_1yr_st.yaml
│   ├── ts2_10yr.yaml / ts2_10yr_st.yaml   # TS2 (spectroscopic bias)
│   ├── ts2_1yr.yaml / ts2_1yr_st.yaml
│   ├── ts3_10yr.yaml / ts3_10yr_st.yaml   # TS3 — data not yet public
│   └── ts4_1yr.yaml / ts4_1yr_st.yaml     # TS4 — data not yet public
├── data/
│   ├── ellen/
│   │   ├── train_175k.parquet    # Stratified training split (175k galaxies)
│   │   ├── test_25k.parquet      # Held-out test split (25k galaxies)
│   │   ├── DC2LSST_u/g/r/i/z/y.res   # LSST filter transmission curves
│   │   └── roman_Y106/J129/H158.res  # Roman filter transmission curves
│   └── pzdc/
│       ├── pz_challenge_taskset_*.parquet   # PZ Data Challenge datasets
│       └── hdf5/                            # Raw HDF5 downloads
├── notebooks/
│   └── evaluate_interactive.ipynb   # Interactive filter/model selection UI
├── scripts/
│   ├── download_pzdc.py       # Download PZ Data Challenge data from NERSC
│   ├── prepare_data.py        # Build stratified Ellen train/test splits
│   ├── prepare_pzdc.py        # Build stratified PZDC train/test splits
│   ├── run_benchmarks.py      # Train flat MLP/MDN baselines for comparison
│   ├── calibrate_posterior.py # Post-hoc calibration on an existing run (no retrain)
│   ├── debug_nsf_conditioning.py  # Verify NSF uses context (log p sensitivity)
│   ├── export_qp.py           # Export p(z) posteriors to qp HDF5 (submission format)
│   └── submission.py          # PZDC challenge submission wrapper functions
├── src/
│   ├── filters.py          # Filter registry: .res parsing, λ_eff / Δλ
│   ├── dataset.py          # Parquet → variable-length token sets + collate_fn
│   ├── config.py           # Dataclass configs + YAML loader/saver
│   ├── train.py            # Training loop, checkpointing, metrics, plotting
│   ├── checkpoint_loader.py # Load old/new checkpoints (infers layout from weights)
│   ├── model_dims.py       # Bottleneck / representation dim helpers
│   ├── coverage_summary.py # Fixed-size wavelength/coverage summaries (SED mode)
│   ├── density_context.py  # Optional NSF density-context MLP
│   ├── data_options.py     # Token schema + legacy config alias resolution
│   ├── run_artifacts.py    # Checkpoint roles, run discovery, post_hoc.json paths
│   ├── calibration.py      # Post-hoc: MDN σ-scale; NSF grid temperature
│   ├── training_stages.py  # Split-training stage overrides + encoder freeze/load
│   ├── platform_fix.py     # macOS OpenMP workaround (auto-imported before torch)
│   ├── evaluate.py         # Photoz metrics, CDE loss, PIT statistics
│   ├── plot.py             # Scatter, Δz, survey metrics, calibration plots
│   ├── benchmarks/         # Flat-vector MLP/MDN baseline models + training
│   └── models/
│       ├── deepsets.py         # φ MLP → masked pool → ρ MLP
│       ├── set_transformer.py  # Token embed → multi-head self-attention → PMA pool
│       ├── bottleneck.py       # Optional encoder bottleneck MLP (384 → latent)
│       └── heads/
│           ├── mlp_regressor.py  # Point estimate, Huber loss
│           ├── binned_pdf.py     # Softmax over K z-bins, NLL loss
│           ├── mdn.py            # Gaussian mixture, NLL loss
│           └── nsf.py            # Neural Spline Flow, NLL loss
├── ideas.md            # Future improvement ideas and notes
├── outputs/            # DeepSetZ runs; one subdirectory per run
├── benchmarks/         # Flat baseline runs (MLP/MDN); same layout as outputs/
├── environment.yml
└── requirements.txt
```

---

## Filter Coverage

Filters are automatically sorted by effective wavelength. Metadata is read from `.res` transmission curves where available; otherwise hardcoded approximate values are used.

| Filter | Survey | λ_eff (Å) | Source |
|---|---|---|---|
| LSST u/g/r/i/z/y | LSST | 3671–9710 | `.res` files |
| Roman Y106/J129/H158 | Roman | 10595–15791 | `.res` files |
| Roman F184/K213 | Roman | 18400–21300 | Hardcoded |
| Euclid Y/J/H | Euclid | 10640–17700 | Hardcoded |
| WISE W1/W2 | WISE | 33526–46028 | Hardcoded |

**16 filters total**, covering 0.37–4.6 µm.

---

## Installation

### Conda environment (recommended, M2/M3 Mac with MPS support)

```bash
conda env create -f environment.yml
conda activate deepset-z
```

### Pip (alternative)

```bash
pip install -r requirements.txt
```

The `qp-prob` package is required for PZDC submission export:

```bash
pip install qp-prob
```

### macOS note: `torch_shm_manager` processes

Configs default to `num_workers: 4`, but on macOS the code automatically forces this to **0** to avoid lingering `torch_shm_manager` subprocesses that can accumulate after training and slow the machine down. You may see this message once per run:

```
[info] num_workers forced to 0 on macOS (avoids lingering torch_shm_manager processes)
```

If you still have orphaned processes from older runs: `pkill -f torch_shm_manager`

On Linux with CUDA, `num_workers: 4` is used as configured.

### macOS note: OpenMP / `libomp` abort

On Apple Silicon, conda PyTorch and other packages can each link their own copy of `libomp`. Importing both can abort with:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

Training and calibration scripts apply the standard workaround automatically (`KMP_DUPLICATE_LIB_OK=TRUE`) via `src/platform_fix.py` before PyTorch is imported. You should not need to set this manually. If a different entry point hits the same error, import `src.platform_fix` before `torch`.

---

## Data

### Ellen noiseless mock data

Pre-split stratified datasets are in `data/ellen/`:

| File | Rows | Purpose |
|---|---|---|
| `train_175k.parquet` | 175,000 | Training + validation |
| `test_25k.parquet` | 25,000 | Held-out test evaluation |

To regenerate the splits from the original 100k catalogues:

```bash
python scripts/prepare_data.py
```

### PZ Data Challenge data

Download the PZDC datasets from NERSC (requires an internet connection):

```bash
# Task Set 1 only (publicly available)
python scripts/download_pzdc.py --taskset 1 --output data/pzdc

# All available task sets
python scripts/download_pzdc.py --taskset all --output data/pzdc
```

> **Note:** Task Sets 3 and 4 are not yet in the public archive. Only Task Sets 1 and 2 are downloadable at this time.

After downloading, generate labelled train/test splits from the PZDC training files:

```bash
python scripts/prepare_pzdc.py --data-dir data/pzdc
```

This creates `_train.parquet` and `_test.parquet` files alongside the originals with a stratified 80/20 split.

---

## Usage

### Training

```bash
# DeepSets + MLP regressor on Ellen data
python src/train.py configs/deepsets.yaml

# Set Transformer + MLP regressor
python src/train.py configs/set_transformer.yaml

# DeepSets + Neural Spline Flow head
python src/train.py configs/deepsets_nsf.yaml

# Train on PZDC Task Set 1 (10-year depth, Cardinal simulation)
python src/train.py configs/ts1_10yr.yaml

# Set Transformer variant
python src/train.py configs/ts1_10yr_st.yaml

# NSF + SED extension (end-to-end)
python src/train.py configs/ts1_10yr_st_nsf_sed.yaml --run_name ts1_10yr_nsf_sed_01_st

# Override the run name
python src/train.py configs/deepsets.yaml --run_name my_experiment
```

Checkpoints, a config copy, metrics JSON, and diagnostic plots are all saved to `outputs/<run_name>/`.

**Final evaluation and plots use the best validation checkpoint** (`best_model.pt` / `best_posterior.pt`), not the last-epoch weights in `final_model.pt`. Training logs report `train_nll` / `val_nll` (lower is better).

### Post-hoc posterior calibration (no retrain)

**MDN:** fit a single σ scale factor on the validation split (MACE grid search).  
**NSF:** fit **grid temperature** `T` on evaluated PDF grids — spline widths are **not** scaled (invalid geometry).

```bash
# Auto-detect checkpoint role
python scripts/calibrate_posterior.py outputs/ts1_10yr_08_st

# Split-training posterior
python scripts/calibrate_posterior.py outputs/ts1_10yr_10_st --role posterior
```

Enable in config:

```yaml
# MDN
training:
  post_hoc_calibration:
    enabled: true
    sigma_min: 0.2
    sigma_max: 1.0

# NSF — use grid temperature, not σ-width scaling
head:
  nsf:
    use_grid_temperature_scaling: true
    disable_spline_width_posthoc_scaling: true
training:
  post_hoc_calibration:
    enabled: true
    temperature_min: 0.5
    temperature_max: 2.0
    n_grid_pdf: 256
```

Post-hoc also runs at end of training when `post_hoc_calibration.enabled: true`. Writes `calibration/post_hoc.json` into the existing run directory.

| File | Contents |
|---|---|
| `test_metrics_post_hoc.json` | MACE / KS / PIT RMSE before vs after; point metrics (should be unchanged) |
| `plots/calibration_post_hoc.png` | PIT + coverage for the scaled posterior |
| `plots/calibration_comparison.png` | Side-by-side before vs after on the test set |

**Interactive notebook:** open `notebooks/evaluate_interactive.ipynb`, select your run, set **View → Posterior + post-hoc σ**, press **Run** for scatter/Δz on the test set, then **Calibration** for live PIT plots. Switch back to **End-to-end** to compare raw vs calibrated.

### Training modes (one config, three pipelines)

All modes use the same YAML shape; these flags select the pipeline:

| Mode | Config | Output checkpoints |
|------|--------|-------------------|
| **End-to-end** (default) | `split_training: false` | `best_model.pt` |
| **Two-stage** | `split_training: true` | `best_point.pt` + `best_posterior.pt` |
| **Stage 2 only** | `split_training: true` + `stage1_checkpoint: outputs/…/best_point.pt` | copies stage 1 + trains new `best_posterior.pt` |

Post-hoc calibration (`post_hoc_calibration.enabled: true`) runs after any mode that ends with an MDN/NSF head (MDN: σ scale; NSF: grid temperature).

### Split training (encoder → point head → posterior head)

Optional two-stage pipeline: train an MLP point head first, freeze the encoder, then train an MDN or NSF on top. Both checkpoints live under the same `outputs/<run_name>/`:

```bash
# Enable split_training in the config (see example below), then:
python src/train.py configs/ts1_10yr_st.yaml --run_name ts1_10yr_10_st
```

Stage 1 writes `best_point.pt`; stage 2 writes `best_posterior.pt`. The interactive notebook exposes these as separate **View** options alongside end-to-end and post-hoc-calibrated posteriors.

**Stage 2 only** — reuse a finished stage-1 encoder without retraining the MLP:

```bash
# MDN two-stage
python src/train.py configs/ts1_10yr_st_2part.yaml \
  --run_name ts1_10yr_11_st \
  --stage1_checkpoint outputs/ts1_10yr_10_st/best_point.pt

# NSF two-stage (same pipeline; stage 2 head is NSF)
python src/train.py configs/ts1_10yr_st_nsf_2part.yaml \
  --run_name ts1_10yr_nsf_2part_01_st \
  --stage1_checkpoint outputs/ts1_10yr_11_st/best_point.pt
```

Both configs share the same structure: stage 1 = MLP point map; stage 2 = `freeze_encoder: true` + PDF head only. Stage-2 `spread_lambda` differs by head (MDN often `5.0`, NSF typically `0.05`–`1.0`).

### Benchmark baselines

Flat-vector MLP and MDN baselines (6-layer trunk: 64→128→256→128→64→32) train on fixed filter subsets for direct comparison with DeepSetZ. Outputs go to `benchmarks/` instead of `outputs/`.

```bash
# Full suite: Ellen + TS1 + TS2, all filter ladders, MLP + MDN (34 runs)
python scripts/run_benchmarks.py

# Ellen only
python scripts/run_benchmarks.py --datasets ellen

# Specific subsets or model types
python scripts/run_benchmarks.py --subsets decals_3 lsst lsst_roman
python scripts/run_benchmarks.py --models mlp

# Resume a partial run
python scripts/run_benchmarks.py --skip-existing
```

**Filter-subset ladder** (not every permutation — a progressive coverage ladder):

| Step | Subset | Bands |
|---|---|---|
| 1 | `decals_3` | g, r, z |
| 2 | `decals_4` | g, r, i, z |
| 3 | `lsst` | 6 LSST bands |
| 4 | `lsst_roman` | LSST + Roman |
| 5+ | mixtures | + WISE, + Euclid, … |
| last | `all` | all surveys (Ellen only; 16 filters) |

PZDC (TS1/TS2) uses the first four steps only (no Euclid/WISE in the challenge data).

Run names follow `benchmarks/<dataset>_<subset>_<model>/`, e.g. `benchmarks/ellen_decals_3_mlp/`, `benchmarks/ts1_lsst_roman_mdn/`.

### Evaluating a saved checkpoint

```bash
python src/evaluate.py outputs/deepsets_mlp/best_model.pt \
                       outputs/deepsets_mlp/config.yaml
```

### Interactive notebook

Open the Jupyter notebook for post-training exploration:

```bash
jupyter notebook notebooks/evaluate_interactive.ipynb
```

The notebook uses `load_model_from_checkpoint()` so **old runs load correctly** even when saved YAML omits newer fields (e.g. `use_coverage` defaults vs weights on disk).

- **Model selector** — pick from all trained runs in `outputs/` and `benchmarks/` (benchmarks prefixed `[bench]`)
- **View dropdown** — switch between model roles in the same run directory:
  - *Point (MLP)* — `best_point.pt` (split training only)
  - *Posterior (PDF)* — `best_posterior.pt` (split training only)
  - *End-to-end* — `best_model.pt` (standard training)
  - *Posterior + post-hoc σ* — same checkpoint as above + `calibration/post_hoc.json` (MDN: scales σ; NSF: grid temperature; μ unchanged)
- **Filter checkboxes** — select any subset of filters for DeepSetZ models (human-readable names)
- **Fixed filters for benchmarks** — benchmark runs show their training subset as read-only checkboxes
- **Point estimate toggle** — switch between mean / median / mode for probabilistic heads
- **Live scatter plot and Δz histogram** on the test set
- **Calibration plot** (PIT histogram + coverage curve) for MDN, BinnedPDF, and NSF heads

### Exporting predictions for PZDC submission

```bash
python scripts/export_qp.py \
  --config  outputs/ts1_10yr/config.yaml \
  --ckpt    outputs/ts1_10yr/best_model.pt \
  --test    data/pzdc/pz_challenge_taskset_1_cardinal_test_10yr.parquet \
  --output  submissions/pz_challenge_taskset_1_cardinal_pz_estimate_10yr.hdf5
```

This writes a `qp`-format HDF5 file containing the full p(z) posterior for every object, with `zmode` and `object_id` as ancillary data — the format required by the challenge evaluator.

Alternatively, use the Python wrapper functions directly:

```python
from scripts.submission import run_taskset1_estimation_only

run_taskset1_estimation_only(
    model_file  = "outputs/ts1_10yr/best_model.pt",
    test_file   = "data/pzdc/pz_challenge_taskset_1_cardinal_test_10yr.parquet",
    output_file = "submissions/pz_challenge_taskset_1_cardinal_pz_estimate_10yr.hdf5",
)
```

---

## Configuration

Configs are YAML files that override the Python dataclass defaults in `src/config.py`. Only fields you want to change need to be specified.

### Swapping datasets

```yaml
data:
  train_path: data/ellen/train_175k.parquet  # Ellen noiseless mocks
  test_path:  data/ellen/test_25k.parquet
  target_col: true_redshift

  # --- or for PZDC (after running prepare_pzdc.py) ---
  train_path: data/pzdc/ts1_cardinal_10yr_train.parquet
  test_path:  data/pzdc/ts1_cardinal_10yr_test.parquet
  target_col: redshift
  include_errors: true   # use σ_m as a 5th token feature; set model.token_dim: 5
```

All PZDC configs ship with calibration defaults enabled:

```yaml
model:
  use_coverage: true

training:
  fisher_lambda: 0.0
  spread_lambda: 0.01
  val_dropout:   true
```

### Swapping prediction heads

```yaml
head:
  type: mlp_regressor   # point estimate, Huber loss
  # type: binned_pdf    # discrete posterior over K z-bins, NLL loss
  # type: mdn           # Gaussian mixture posterior, NLL loss
  # type: nsf           # Neural Spline Flow posterior, NLL loss  ← recommended

  nsf:
    n_bins:  48
    z_min:   0.0
    z_max:   1.40      # use log(1+z_max) when log_target: true  (e.g. log(4) ≈ 1.40)
    hidden_dims: [128, 64]
```

### Redshift transformation

```yaml
data:
  log_target: true   # train in log(1+z) space to reduce tail bias
                     # all metrics and plots are always shown in real z
```

When `log_target: true` and using the NSF head, set `nsf.z_max` to `log(1 + z_true_max)` — typically `1.40` for `z ∈ [0, 3]`.

### Restricting active surveys or filters

```yaml
data:
  active_surveys: [lsst, roman]        # only use these surveys
  # active_surveys: [mag_g_lsst, mag_r_lsst, mag_i_lsst]   # individual filters
  # active_surveys: []                 # use all available filters (default)
```

### Stratified filter dropout

The dropout strategy trains the model to handle variable filter coverage. Four modes are sampled stochastically each batch:

```yaml
training:
  dropout:
    p_complete:    0.15   # full filter set
    p_preset:      0.25   # realistic survey combos (DECaLS, LSST-only, etc.)
    p_survey_drop: 0.25   # drop 1–2 entire surveys
    p_aggressive:  0.35   # random per-filter drops (most aggressive)
    min_filters:   3      # never drop below this many
```

### Posterior calibration (probabilistic heads)

Two optional regularisation terms penalise posterior width in opposite directions. Both default to `0.0` in all configs; enable as needed after inspecting the PIT histogram.

```yaml
training:
  fisher_lambda: 0.0      # penalises over-confident (sharp) posteriors — U-shaped PIT
  spread_lambda: 0.01    # penalises over-dispersed (broad) posteriors — n-shaped PIT
  val_dropout:   true     # log a second val pass with dropout (monitoring only)
```

| Knob | Effect | Use when |
|---|---|---|
| `fisher_lambda` | Encourages wider posteriors | PIT is U-shaped (under-dispersed) |
| `spread_lambda` | Encourages tighter posteriors | PIT is n-shaped (over-dispersed) |
| `val_dropout` | Logs `val_drop_loss` each epoch | Want to track performance under partial coverage |

Early stopping and checkpointing use the **clean val loss** (full filters, no dropout). The dropout val pass is logged and plotted but does not affect gradients or model selection.

### Split training and calibration

Decouple accurate point estimates from posterior width: train location (MLP) and dispersion (MDN/NSF) in separate stages, then optionally apply post-hoc σ scaling without retraining.

```yaml
training:
  # Standard end-to-end (default) → best_model.pt
  split_training: false

  # Two-stage pipeline → best_point.pt + best_posterior.pt
  # split_training: true
  # stage1:
  #   head: mlp_regressor
  #   epochs: 150
  #   huber_lambda: 0.2
  # stage2:
  #   head: mdn              # or nsf — inherits head.type if omitted
  #   epochs: 80
  #   freeze_encoder: true
  #   huber_lambda: 0.0
  #   spread_lambda: 10.0

  # Post-hoc calibration (MDN / NSF) — writes calibration/post_hoc.json
  post_hoc_calibration:
    enabled:    true
    sigma_min:  0.2       # MDN only
    sigma_max:  1.0
    temperature_min: 0.5  # NSF only (with use_grid_temperature_scaling)
    temperature_max: 2.0
```

| Setting | Effect |
|---|---|
| `split_training: true` | Stage 1: encoder + MLP → `best_point.pt`. Stage 2: frozen encoder + PDF head → `best_posterior.pt`. |
| `stage1_checkpoint` | Skip stage 1; load encoder from `best_point.pt` (split run) or `best_model.pt` (encoder only). Copies stage-1 artefacts into the new run dir. |
| `stage1` / `stage2` | Per-stage overrides — unset fields inherit from top-level `training`. See table below. |

**Per-stage override fields** (`training.stage1` / `training.stage2`):

| Field | Typical stage 1 (MLP) | Typical stage 2 (MDN/NSF) |
|---|---|---|
| `head` | `mlp_regressor` | `mdn` / `nsf` |
| `epochs`, `lr`, `weight_decay`, `warmup_epochs` | Higher LR (e.g. `2e-4`) | Often lower LR (NSF e.g. `8e-6`) |
| `lr_scheduler`, `clip_grad_norm`, `batch_size` | As needed | As needed |
| `huber_lambda`, `spread_lambda`, `huber_delta` | Point loss weight | Posterior regularisation |
| `early_stop_patience`, `early_stop_min_epoch` | Per stage | Per stage |
| `val_dropout`, `full_filter_epochs`, `dropout_resume_lr_mult` | Optional per stage | Optional per stage |
| `freeze_encoder` | — | `true` (default in stage 2) |
| `use_coverage` | Optional | Optional |

```yaml
training:
  lr: 2.0e-4              # default for both stages unless overridden
  weight_decay: 2.0e-4
  batch_size: 512
  split_training: true
  stage1:
    head: mlp_regressor
    epochs: 150
    lr: 2.0e-4
    warmup_epochs: 25
    huber_lambda: 0.2
  stage2:
    head: nsf
    epochs: 80
    lr: 8.0e-6              # NSF needs a much lower LR than the MLP stage
    weight_decay: 1.0e-4
    warmup_epochs: 5
    clip_grad_norm: 0.5
    freeze_encoder: true
    spread_lambda: 0.05
```

Each stage rebuilds its optimiser, scheduler, and dataloaders from the merged config, so stage 2 can use a different batch size or val-dropout setting without affecting stage 1.
| `post_hoc_calibration.enabled` | After training (or via `calibrate_posterior.py`), fit calibration on val. **MDN:** σ scale. **NSF:** grid temperature (requires `head.nsf.use_grid_temperature_scaling: true`). |

All artefacts share one `outputs/<run_name>/` directory so the notebook can compare views without hunting across folders.

### Encoder bottleneck (optional)

Compress pooled encoder features before the head with a 3-layer MLP. **Default `bottleneck: false` leaves behaviour unchanged** (full `embed_dim`, e.g. 384-d, fed to the head).

```yaml
model:
  bottleneck: false      # default — no bottleneck (backwards compatible)
  bottleneck: 64          # 384 → … → 64-d latent fed to head / density context
  bottleneck_dropout: 0.1
```

In split training, the bottleneck is trained in stage 1 and **frozen with the encoder** in stage 2. Checkpoints store `bottleneck.*` weights when enabled.

### SED-like filter handling (opt-in)

Extended token schema and NSF conditioning — **all defaults preserve legacy behaviour** (NaN bands dropped, no detection flags, no coverage summaries).

```yaml
data:
  encode_nondetections: false       # true + keep_token → NaN bands as flagged tokens
  nondetection_policy: drop        # drop | keep_token
  add_detection_flags: false       # +2 token dims: is_detected, is_nondetected
  strict_error_columns: false        # false: zero-fill missing _err cols + warn once

model:
  use_coverage_summary: false      # fixed-size wavelength/coverage scalars
  density_context_branch: false      # separate MLP context path for NSF
```

Example config: `configs/ts1_10yr_st_nsf_sed.yaml` (NSF end-to-end with non-detection tokens, coverage summaries, density context, bottleneck, grid-T post-hoc).

Debug NSF context usage:

```bash
python scripts/debug_nsf_conditioning.py outputs/ts1_10yr_nsf_01_st
python scripts/debug_nsf_conditioning.py --random   # untrained NSF sanity check
```

Unit tests: `python -m unittest tests.test_sed_extension tests.test_code_fixes -v`

### Coverage conditioning

Legacy scalar (default on many PZDC configs):

```yaml
model:
  use_coverage: true   # append n_active / n_total_filters to latent before head
```

Uses a **fixed denominator** (`n_total_filters` from the dataset registry), not batch max token count — so the same galaxy gets the same coverage feature regardless of batch padding.

Split training can set this per stage:

```yaml
training:
  split_training: true
  stage1:
    use_coverage: true   # 384-d latent + 1 → MLP
  stage2:
    use_coverage: true   # same concat before MDN/NSF
```

This lets the head learn "more filters → tighter posterior" without inferring coverage from the embedding alone. Automatically increases the head input dimension by 1.

### Model architecture

```yaml
model:
  type: deepsets        # deepsets | set_transformer
  token_dim: 4          # 4 (magnitudes only) or 5 (with magnitude errors)
  use_coverage: true    # optional; see above

  deepsets:
    phi_hidden: [256, 256]   # per-filter embedding MLP
    latent_dim: 256
    rho_hidden: [512, 512, 256]   # aggregation MLP
    embed_dim:  256
    pooling: attention      # mean | sum | max | attention
    activation: gelu        # gelu | relu | silu | leaky_relu | tanh

  set_transformer:
    embed_dim: 128
    n_heads: 4
    n_attn_layers: 2
```

---

## Prediction Heads

| Head | Type | Loss | Point estimate | Posterior | Recommended for |
|---|---|---|---|---|---|
| `mlp_regressor` | Deterministic | Huber | z_pred | — | Baseline, fast training |
| `binned_pdf` | Probabilistic | NLL | mode | Discrete histogram | All-purpose |
| `mdn` | Probabilistic | NLL | mode | Gaussian mixture | Small datasets |
| `nsf` | Probabilistic | NLL | mode | Spline CDF | Best calibration |

All probabilistic heads expose:
- `.point_estimates_from_params(...)` — returns `z_mean`, `z_median`, `z_mode`
- `.pit_values(...)` — Probability Integral Transform for calibration diagnostics
- `.cdf_at(...)` — evaluates the CDF at arbitrary redshift values
- `.fisher_penalty(...)` / `.spread_penalty(...)` — optional width regularisers (controlled via `fisher_lambda` / `spread_lambda` in config)

---

## Benchmark Baselines

Flat-vector baselines live in `src/benchmarks/` and provide a lower bound for DeepSetZ performance on identical filter subsets.

| Property | DeepSetZ | Benchmark |
|---|---|---|
| Input | Variable-length token set | Fixed-length magnitude vector |
| Missing data | Per-galaxy masking | Training-set median imputation |
| Architecture | DeepSets / Set Transformer + head | 6-layer MLP trunk + scalar / MDN head |
| Output dir | `outputs/` | `benchmarks/` |
| Filter selection | Any subset at inference | Fixed at training time |

The benchmark MLP uses Huber loss; the benchmark MDN uses a 5-component Gaussian mixture with NLL loss. PZDC benchmarks optionally include magnitude errors as extra input features (`include_mag_errors: true`).

Compare against DeepSetZ by matching the filter subset — e.g. `benchmarks/ellen_lsst_mdn/` vs. an interactive notebook run with only LSST filters selected on a DeepSetZ model.

---

## Evaluation Metrics

Standard photoz metrics computed on `Δz = (z_phot − z_spec) / (1 + z_spec)`:

| Metric | Definition |
|---|---|
| bias | median(Δz) |
| σ_NMAD | 1.4826 × median(\|Δz − bias\|) |
| outlier rate | fraction with \|Δz\| > 0.15 |
| outlier rate [PZDC] | fraction with \|Δz\| > max(0.06, 3 × σ_IQR) |
| RMSE | √mean(Δz²) |
| MAE | mean(\|Δz\|) |

Probabilistic metrics (for MDN, BinnedPDF, NSF heads):

| Metric | Definition |
|---|---|
| CDE loss | Conditional density estimation loss (Izbicki & Lee 2017) |
| PIT KS | KS statistic of PIT histogram vs. Uniform[0,1] |
| PIT RMSE | RMSE of empirical PIT CDF vs. ideal diagonal |
| Coverage | Fraction of true z within predicted credible intervals |

---

## Output Files

After training, `outputs/<run_name>/` contains one directory per config name. Multiple model **roles** (end-to-end, split-training stages, post-hoc calibration) are stored as separate files in that directory — not as separate output folders.

**Split training** (`split_training: true`) writes separate plot folders per stage:

```
outputs/<run_name>/plots/
  stage1/          # MLP point head — scatter, Δz, training curves, survey metrics
  stage2/          # MDN/NSF posterior (raw) — incl. calibration.png
  final/           # Post-hoc σ-scaled posterior — calibration + before/after comparison
```

```
outputs/<run_name>/
├── config.yaml
├── best_model.pt              # End-to-end training (split_training: false)
├── best_point.pt              # Split training — stage 1 (encoder + MLP)
├── best_posterior.pt          # Split training — stage 2 (encoder + MDN/NSF)
├── calibration/
│   └── post_hoc.json          # Fitted σ scale (val split; no duplicate checkpoint)
├── test_metrics_post_hoc.json # Test-set calibration before/after (from calibrate_posterior.py)
├── final_model.pt             # Last epoch (debugging); plots/metrics use best_*.pt
├── history.json               # End-to-end or stage 2 history
├── history_stage1.json        # Split training — stage 1 only
├── test_metrics.json          # End-to-end or stage 2 test metrics
├── test_metrics_point.json    # Split training — stage 1
├── test_metrics_posterior.json
├── subset_metrics.json        # Per-survey-combination metrics
├── predictions.npz            # z_true, z_pred (and z_mean/median/mode/pit for prob. heads)
└── plots/
    ├── training_curves.png
    ├── scatter.png
    ├── delta_z.png
    ├── survey_metrics.png
    ├── survey_metrics.csv
    ├── scatter_mean_median.png
    ├── calibration.png            # Raw posterior at end of training
    ├── calibration_post_hoc.png   # After post-hoc σ scaling (test set)
    └── calibration_comparison.png # Before vs after side-by-side (test set)
```

`post_hoc.json` records the fitted scale, MACE/KS before and after (val split), and which checkpoint role was calibrated. `test_metrics_post_hoc.json` holds the same comparison on the held-out test set. The evaluation notebook loads the same weights and applies the scale only when **View → Posterior + post-hoc σ** is selected.

Benchmark runs use the same layout under `benchmarks/<run_name>/`. The config includes a `benchmark:` section recording the model type (`flat_mlp` / `flat_mdn`), subset name, and fixed filter columns.

---

## PZ Data Challenge

DeepSetZ supports entry into the [PZ Data Challenge](https://pz-data-challenge.readthedocs.io/en/latest/) across all four task sets.

### Task sets

| Task Set | Description | Key challenge | Config |
|---|---|---|---|
| TS1 | IID train/test split | Baseline performance | `ts1_*.yaml` |
| TS2 | Spectroscopic selection bias | Domain shift | `ts2_*.yaml` |
| TS3 | Blended sources | Non-standard SEDs | `ts3_*.yaml` |
| TS4 | Variable noise levels | Heterogeneous errors | `ts4_*.yaml` |

### Submission workflow

```bash
# 1. Download and prepare data
python scripts/download_pzdc.py --taskset 1 --output data/pzdc
python scripts/prepare_pzdc.py --data-dir data/pzdc

# 2. Train
python src/train.py configs/ts1_10yr.yaml

# 3. Export submission file
python scripts/export_qp.py \
  --config  outputs/ts1_10yr/config.yaml \
  --ckpt    outputs/ts1_10yr/best_model.pt \
  --test    data/pzdc/pz_challenge_taskset_1_cardinal_test_10yr.parquet \
  --output  submissions/pz_challenge_taskset_1_cardinal_pz_estimate_10yr.hdf5
```

---

## Extending to New Filters

1. Add the new `.res` file to `data/` (or add hardcoded metadata to `_HARDCODED` in `src/filters.py`).
2. Ensure the corresponding magnitude column exists in your parquet catalogue.
3. Re-run — the model architecture does not change; only the number of possible tokens increases.

---

## Ideas and Future Work

See [`ideas.md`](ideas.md) for a structured list of potential improvements, including:
- Colour tokens and flux representations
- Uncertainty-weighted attention
- Domain adaptation for spectroscopic selection bias
- Quantile regression and evidential deep learning heads
- Contrastive pre-training and multi-task learning
- Physical SED augmentation
- Posterior calibration extras not yet implemented: attention entropy conditioning, temperature scaling, isotonic regression (post-hoc σ scaling for MDN/NSF is implemented — see above)
