# DeepSetZ — run commands

Activate the environment first:

```bash
conda activate deepset-z
cd /path/to/DeepSetZ
```

---

## 1. End-to-end (default)

Train encoder + head together in one pass.

**Config:** `configs/ts1_10yr_st.yaml` (MDN) or `configs/ts1_10yr_st_nsf.yaml` (NSF)

**Required params:**

```yaml
training:
  split_training: false   # default — can omit

head:
  type: mdn               # or mlp_regressor | nsf | binned_pdf
```

**Command:**

```bash
python src/train.py configs/ts1_10yr_st.yaml --run_name ts1_10yr_08_st
```

**Outputs:** `outputs/<run_name>/best_model.pt` (final test plots use this, not `final_model.pt`)

**NSF (end-to-end):**

```bash
python src/train.py configs/ts1_10yr_st_nsf.yaml --run_name ts1_10yr_nsf_01_st
```

**NSF + SED-like tokens / coverage summaries / bottleneck:**

```bash
python src/train.py configs/ts1_10yr_st_nsf_sed.yaml --run_name ts1_10yr_nsf_sed_01_st
```

Key opt-in flags in `ts1_10yr_st_nsf_sed.yaml`:

```yaml
data:
  encode_nondetections: true
  nondetection_policy: keep_token
  add_detection_flags: true
model:
  bottleneck: 64                    # false = unchanged (384-d latent)
  use_coverage_summary: true
  density_context_branch: true
head:
  nsf:
    use_grid_temperature_scaling: true
    disable_spline_width_posthoc_scaling: true
training:
  post_hoc_calibration:
    enabled: true
    temperature_min: 0.5
    temperature_max: 2.0
```

---

## 2. Two-stage (stage 1 + stage 2)

Stage 1: encoder (+ optional bottleneck) + MLP point head.  
Stage 2: frozen encoder + MDN/NSF posterior head only.

**Config:** `configs/ts1_10yr_st_2part.yaml` (MDN) or `configs/ts1_10yr_st_nsf_2part.yaml` (NSF)

**Required params:**

```yaml
training:
  split_training: true
  lr: 2.0e-4              # inherited by both stages unless overridden below
  stage1:
    head: mlp_regressor
    epochs: 150
    lr: 2.0e-4            # encoder + MLP — typically higher LR
  stage2:
    head: mdn              # or nsf
    epochs: 80
    lr: 1.0e-4            # MDN; use ~8e-6 for NSF (see nsf_2part config)
    freeze_encoder: true
```

Per-stage overrides also support `weight_decay`, `warmup_epochs`, `clip_grad_norm`, `batch_size`, `lr_scheduler`, `huber_lambda`, `spread_lambda`, `val_dropout`, and early-stopping fields. Unset keys inherit from top-level `training`.

**Command:**

```bash
python src/train.py configs/ts1_10yr_st_2part.yaml --run_name ts1_10yr_10_st
```

**Outputs:**

```
outputs/<run_name>/
  best_point.pt
  best_posterior.pt
  plots/stage1/
  plots/stage2/
  plots/final/          # if post_hoc_calibration.enabled: true
```

**NSF (two-stage):**

```bash
python src/train.py configs/ts1_10yr_st_nsf_2part.yaml --run_name ts1_10yr_nsf_2part_01_st
```

NSF stage-2 tuning (vs MDN in `ts1_10yr_st_2part.yaml`):

| Field | MDN 2-part | NSF 2-part |
|-------|------------|------------|
| `stage2.spread_lambda` | `5.0` | `0.05` |
| `stage2.huber_lambda` | `0.1` | `0.1` |
| Post-hoc | σ scale | grid temperature |

---

## 3. Stage 2 only (reuse existing stage 1)

Skip stage 1; load encoder from a finished run.

**Config:** same as two-stage, plus `stage1_checkpoint` (YAML or CLI).

**Command (CLI flag — overrides YAML):**

```bash
# MDN
python src/train.py configs/ts1_10yr_st_2part.yaml \
  --run_name ts1_10yr_11_st \
  --stage1_checkpoint outputs/ts1_10yr_10_st/best_point.pt

# NSF
python src/train.py configs/ts1_10yr_st_nsf_2part.yaml \
  --run_name ts1_10yr_nsf_2part_02_st \
  --stage1_checkpoint outputs/ts1_10yr_11_st/best_point.pt
```

**Checkpoint can be:**

- `best_point.pt` — from a two-stage run (recommended)
- `best_model.pt` — from end-to-end (encoder + bottleneck weights only)

Encoder architecture in the config must match the source run (`embed_dim`, `token_dim`, `bottleneck`, etc.).

**Outputs:** copies stage-1 artefacts + new `best_posterior.pt`, `plots/stage2/`, `plots/final/`

---

## Optional: post-hoc calibration (no retrain)

**MDN:** σ scaling on mixture widths.

```bash
python scripts/calibrate_posterior.py outputs/ts1_10yr_08_st
python scripts/calibrate_posterior.py outputs/ts1_10yr_10_st --role posterior
```

**NSF:** grid temperature (requires `use_grid_temperature_scaling: true` in config).

```bash
python scripts/calibrate_posterior.py outputs/ts1_10yr_nsf_01_st
python scripts/calibrate_posterior.py outputs/ts1_10yr_nsf_2part_01_st --role posterior
```

Writes `calibration/post_hoc.json` + test plots into the same run folder.

---

## Diagnostics and tests

**NSF context sensitivity** (log p should change when context is shuffled):

```bash
python scripts/debug_nsf_conditioning.py outputs/ts1_10yr_nsf_01_st
python scripts/debug_nsf_conditioning.py --random
```

**Unit tests** (SED extension + recent bug fixes):

```bash
python -m unittest tests.test_sed_extension tests.test_code_fixes -v
```

**Interactive evaluation** (checkpoint-aware reload for old runs):

```bash
jupyter notebook notebooks/evaluate_interactive.ipynb
```

---

## Quick reference

| Mode | Key flag | Main checkpoint(s) |
|------|----------|-------------------|
| End-to-end | `split_training: false` | `best_model.pt` |
| Two-stage | `split_training: true` | `best_point.pt` + `best_posterior.pt` |
| Stage 2 only | `split_training: true` + `stage1_checkpoint` | `best_posterior.pt` (new) |

| Config | Purpose |
|--------|---------|
| `ts1_10yr_st.yaml` | ST + MDN end-to-end |
| `ts1_10yr_st_2part.yaml` | ST + MLP → MDN |
| `ts1_10yr_st_nsf.yaml` | ST + NSF end-to-end |
| `ts1_10yr_st_nsf_2part.yaml` | ST + MLP → NSF |
| `ts1_10yr_st_nsf_sed.yaml` | ST + NSF + SED tokens / coverage / bottleneck |

**CLI overrides (any mode):**

```bash
python src/train.py <config.yaml> --run_name my_run
python src/train.py <config.yaml> --stage1_checkpoint outputs/foo/best_point.pt
```
