# DeepSetZ — run commands

Activate the environment first:

```bash
conda activate deepset-z
cd /path/to/DeepSetZ
```

---

## 1. End-to-end (default)

Train encoder + head together in one pass.

**Config:** `configs/ts1_10yr_st.yaml` (or any config without split training)

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

**Outputs:** `outputs/<run_name>/best_model.pt`

---

## 2. Two-stage (stage 1 + stage 2)

Stage 1: encoder + MLP point head.  
Stage 2: frozen encoder + MDN/NSF posterior head.

**Config:** `configs/ts1_10yr_st_2part.yaml`

**Required params:**

```yaml
training:
  split_training: true
  stage1:
    head: mlp_regressor
    epochs: 150
  stage2:
    head: mdn              # or nsf
    epochs: 80
    freeze_encoder: true
```

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

---

## 3. Stage 2 only (reuse existing stage 1)

Skip stage 1; load encoder from a finished run.

**Config:** same as two-stage (`configs/ts1_10yr_st_2part.yaml`), plus:

```yaml
training:
  split_training: true
  stage1_checkpoint: outputs/ts1_10yr_10_st/best_point.pt
  stage2:
    head: mdn
    epochs: 80
    freeze_encoder: true
```

**Command (CLI flag — overrides YAML):**

```bash
python src/train.py configs/ts1_10yr_st_2part.yaml \
  --run_name ts1_10yr_11_st \
  --stage1_checkpoint outputs/ts1_10yr_10_st/best_point.pt
```

**Checkpoint can be:**

- `best_point.pt` — from a two-stage run (recommended)
- `best_model.pt` — from end-to-end (encoder weights only)

Encoder architecture in the config must match the source run (`embed_dim`, `token_dim`, etc.).

**Outputs:** copies stage-1 artefacts + new `best_posterior.pt`, `plots/stage2/`, `plots/final/`

---

## Optional: post-hoc calibration (no retrain)

Fit σ scaling on an existing MDN/NSF run:

```bash
python scripts/calibrate_posterior.py outputs/ts1_10yr_08_st
python scripts/calibrate_posterior.py outputs/ts1_10yr_10_st --role posterior
```

Writes `calibration/post_hoc.json` + test plots into the same run folder.

---

## Quick reference

| Mode | Key flag | Main checkpoint(s) |
|------|----------|-------------------|
| End-to-end | `split_training: false` | `best_model.pt` |
| Two-stage | `split_training: true` | `best_point.pt` + `best_posterior.pt` |
| Stage 2 only | `split_training: true` + `stage1_checkpoint` | `best_posterior.pt` (new) |

**CLI overrides (any mode):**

```bash
python src/train.py <config.yaml> --run_name my_run
python src/train.py <config.yaml> --stage1_checkpoint outputs/foo/best_point.pt
```
