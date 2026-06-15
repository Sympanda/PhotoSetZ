# Ideas for Improving DeepSetZ

A working note of potential improvements discussed in the development of DeepSetZ.
Items are roughly grouped by theme and annotated with expected complexity and gain.

---

## 1. Richer Token Representations

### 1a. Colour tokens
Instead of raw magnitudes, represent each filter pair as a colour index:
`token = [λ_eff_1, Δλ_1, λ_eff_2, Δλ_2, m_1 - m_2]`
Colours are insensitive to dust normalisation and overall SED scaling, so the
network can focus on shape rather than amplitude. Suitable when apertures are
matched across bands.

**Effort**: medium  |  **Expected gain**: high

### 1b. Magnitude + colour hybrid
Keep the raw magnitude token and add extra colour tokens as additional set
elements. This gives the network both absolute and differential information.

**Effort**: medium  |  **Expected gain**: high

### 1c. Flux tokens instead of magnitudes
Replace magnitudes with fluxes (or log-fluxes) so that non-detections can
be represented as an upper-limit token rather than masked out entirely.
Requires upper-limit-aware loss or a separate learned upper-limit embedding.

**Effort**: high  |  **Expected gain**: medium (primarily for sparse data)

---

## 2. Uncertainty-Weighted Attention / Weighting

### 2a. Uncertainty as attention prior
Pass σ_m (magnitude error) into the multi-head self-attention mechanism as a
key-scaling factor: `Attention_ij ∝ exp(-σ_m_i) · softmax(QK^T/√d)`.
This biases the model to rely more on precise measurements.

**Effort**: medium  |  **Expected gain**: medium

### 2b. Heteroscedastic input noise augmentation
During training, corrupt magnitudes by sampling `m̃ = m + N(0, σ_m)`.
Forces the model to be robust to photometric noise without changing
the architecture.

**Effort**: low  |  **Expected gain**: medium

---

## 3. Domain Adaptation (for PZDC tasks 2–4)

### 3a. Spectroscopic selection bias correction (Task Set 2)
The spectroscopic training set is magnitude-biased (brighter galaxies selected).
Techniques:
- **Importance weighting**: Estimate density ratio P_test / P_train and use
  as sample weights during training.
- **Adversarial domain adaptation**: Add a domain classifier head that cannot
  distinguish training vs. photometric-only galaxies after gradient reversal.
- **Self-training / pseudo-labels**: Use the trained model to assign soft
  labels to photometric-only galaxies, then retrain on the full set.

**Effort**: high  |  **Expected gain**: high for TS2

### 3b. Test-time augmentation (TTA)
At inference, evaluate each galaxy at several random filter-dropout masks and
ensemble the resulting posteriors. This is free if the model already handles
dropout at test time, and reduces variance.

**Effort**: low  |  **Expected gain**: low–medium

---

## 4. Improved Posterior Heads

### 4a. Neural Spline Flow (NSF) — already implemented ✓
A monotone rational-quadratic spline CDF whose knot parameters are predicted
by the encoder. Analytically exact CDF and its inverse → exact PIT, median,
and CDF-based sampling without numerical integration.

### 4b. Normalising flow conditioned on the embedding
A full normalising flow (e.g. Real NVP or MAF) conditioned on the encoder
embedding transforms a standard Gaussian into p(z|X). More expressive than
MDN or NSF but requires many more parameters and is harder to train.

**Effort**: high  |  **Expected gain**: medium–high

### 4c. Quantile regression
Directly predict a set of quantiles {τ_1, …, τ_K} of p(z|X) using the
pinball loss. Naturally calibrated by construction; easily recovers the median
(τ=0.5) and credible intervals.

**Effort**: medium  |  **Expected gain**: medium

### 4d. Evidential deep learning
Predict the parameters of a Normal-Inverse-Gamma prior over (z, σ²) using
the evidential regression framework (Amini et al. 2020). Provides principled
epistemic uncertainty alongside aleatoric uncertainty.

**Effort**: medium  |  **Expected gain**: medium (uncertainty decomposition)

---

## 5. Training Strategies

### 5a. Photometric redshift as a multi-task problem
Add auxiliary regression tasks that share the encoder:
- Stellar mass log(M_*)
- Rest-frame colour (u-r)
- Star-galaxy morphology score
These tasks constrain the embedding to encode physically meaningful features
beyond just redshift, which tends to reduce outlier rates.

**Effort**: high (requires additional labels)  |  **Expected gain**: medium

### 5b. Contrastive pre-training
Pre-train the set encoder with a contrastive objective: galaxies with similar
true redshifts should have similar embeddings. Fine-tune the head afterwards.
Particularly useful if labels are scarce (TS2, TS3).

**Effort**: high  |  **Expected gain**: medium–high

### 5c. Cosine learning rate warmup + restarts (SGDR)
Already using cosine annealing; extending to multiple restarts (SGDR) can
help escape local minima when training probabilistic heads on complex
multi-modal posteriors.

**Effort**: low  |  **Expected gain**: low–medium

### 5d. Larger ensembles instead of single models
Train N=5 independent models with different random seeds. Ensemble their
predicted PDFs as a mixture. Empirically gives large gains in calibration
and σ_NMAD at very little implementation cost.

**Effort**: low  |  **Expected gain**: high

---

## 6. Architecture Search

### 6a. Deeper / wider φ-network
The per-filter embedding network `φ(f_i)` is currently shallow (2 layers).
Increasing depth to 3–4 layers with residual connections might allow richer
per-filter representations before aggregation.

**Effort**: low  |  **Expected gain**: low–medium

### 6b. Cross-attention between filters and a learnable reference set
Replace self-attention with cross-attention where the keys/values are a
small learnable reference set (like an Induced Set Attention Block, ISAB).
Reduces O(N²) attention to O(N·M) where M is the reference set size.

**Effort**: medium  |  **Expected gain**: low (only matters for large N)

### 6c. Perceiver-style architecture
A Perceiver encoder (Jaegle et al. 2021) maps a variable-length set to a
fixed-length latent array via cross-attention. Could be a natural fit for
the variable filter sets here, while scaling to many more filters (e.g.
future photometric surveys).

**Effort**: high  |  **Expected gain**: medium (mainly for scalability)

---

## 7. Data

### 7a. Physical SED augmentation
Use a fast SED code (e.g. CIGALE, Le Phare, or even a learned emulator) to
synthesise additional training galaxies with rare SEDs (strong emission lines,
unusual dust geometries). Addresses the tail bias noted in current training.

**Effort**: high  |  **Expected gain**: high for tails

### 7b. Mixing real spectroscopic data
Mix a fraction of real spectroscopic observations (e.g. from DESI, zCOSMOS)
into the training set to partially bridge the domain gap between simulated
and real photometry.

**Effort**: medium  |  **Expected gain**: high for real-data deployment

### 7c. Photometric zero-point offsets as training noise
Randomly perturb the per-band zero-point offsets by ±0.01–0.05 mag during
training. Forces the network to be robust to systematic calibration errors,
which are common in real surveys.

**Effort**: low  |  **Expected gain**: medium for real-data deployment

---

## 8. Calibration

*Context: MDN runs on TS1 10yr showed an n-shaped (inverted-U) PIT histogram,
indicating over-dispersed posteriors — the model outputs widths that are too
broad relative to the true residuals. This is suspected to be caused by a
tension between the stratified filter dropout (which requires broad posteriors
during sparse training steps) and the tighter posteriors warranted at test time
when most filters are present.*

### 8a. Fisher regularisation (training-time) ✓ implemented
Add a Fisher information penalty to the NLL loss during training:

```
loss = NLL  +  λ · E[ (d/dz log p(z|x))² ]
```

This directly penalises posteriors with high curvature (sharp, narrow peaks)
and prevents the model from being overconfident. Analytically tractable for
MDN (weighted sum of Gaussian derivative terms), NSF (spline PDF is smooth),
and approximately for BinnedPDF (finite differences). Controlled by a single
`fisher_lambda` hyperparameter (set to 0 to disable).

Based on prior experience on a separate redshift task, this was found to
improve calibration significantly.

**Effort**: medium  |  **Expected gain**: high

### 8b. Explicit token-count conditioning ✓ implemented
Pass the number of active filters `n_filters / n_max` as an explicit scalar
feature into the prediction head alongside the encoder embedding. This gives
the head direct access to coverage information rather than relying on it being
implicitly encoded in the embedding. Cheap to implement, directly addresses
the root cause of the over-dispersion under dropout.

Controlled via `model.use_coverage: true` in config.

**Effort**: low  |  **Expected gain**: medium–high

### 8b-ii. Spread regularisation (training-time) ✓ implemented
Inverse of Fisher: penalises broad/over-dispersed posteriors via a weighted
variance term `E[Σ_k π_k σ_k²]` (MDN) or posterior variance (BinnedPDF/NSF).
Use when PIT is n-shaped (our case). Controlled by `spread_lambda` in config.
`fisher_lambda` and `spread_lambda` are opposing knobs — typically only one
is non-zero.

**Effort**: low  |  **Expected gain**: medium–high for n-shaped PIT

### 8b-iii. Validation with dropout ✓ implemented
Second val pass each epoch with the same filter dropout as training. Seed is
fixed per epoch (`seed + epoch`) for a stable signal. Logged as `val_drop_*`
in `history.json` and plotted on `training_curves.png`. Early stopping still
uses clean (no-dropout) val loss.

Controlled via `training.val_dropout: true`.

**Effort**: low  |  **Expected gain**: diagnostic (does not affect gradients)

### 8c. Attention entropy as an uncertainty feature
Compute the entropy of the PMA attention weights and concatenate it to the
embedding before the head. High entropy (attention spread evenly across many
tokens) ≈ uncertain; low entropy (attention peaked on a few tokens) ≈ the
model found a clear signal. A proxy for information content that the head
can exploit.

**Effort**: low  |  **Expected gain**: medium

### 8d. Temperature scaling (post-hoc)
After training, fit a single temperature parameter T on a validation split
to scale all posterior widths: divide σ_k by T (MDN) or rescale the spline
(NSF). One-parameter fit, no retraining required. Appropriate for PZDC
submission as long as T is fitted on val labels, not test labels.
Diagnose need from PIT: n-shape → T > 1 (sharpen), U-shape → T < 1 (widen).

**Effort**: very low  |  **Expected gain**: medium for calibration

### 8e. Isotonic regression / Platt scaling on the PIT histogram
Fit a monotone recalibration function to the predicted CDFs using held-out
validation data. More flexible than temperature scaling but still post-hoc.

**Effort**: low  |  **Expected gain**: medium for calibration

### 8f. Prior-matching redshift prior
Multiply predicted p(z|X) by a population-level prior N(z) and renormalise.
Implicitly matches the overall redshift distribution to known population
statistics, reducing systematic bias in surveys with known N(z).

**Effort**: low  |  **Expected gain**: medium

---

## 9. Neural Spline Flow vs MDN: A Quick Comparison

| Aspect             | MDN                           | NSF                            |
|--------------------|-------------------------------|--------------------------------|
| Expressiveness     | Multi-modal (C components)    | Fully flexible (K-bin spline)  |
| CDF computation    | Numerical (grid search)       | Exact (analytic spline)        |
| CDF inverse        | Numerical (binary search)     | Exact (quadratic formula)      |
| Parameters per galaxy | 3C (π, μ, σ)              | 3K+1 (w, h, d)                 |
| Typical K or C     | 5–10                          | 16–64                          |
| Training stability | Moderate (mixture collapse)   | High (softmax/softplus outputs)|
| Calibration        | Good                          | Very good                      |
| Implementation     | Simple                        | Moderate                       |
| Added in DeepSetZ  | ✓ (head = "mdn")              | ✓ (head = "nsf")               |

For the PZDC, **NSF** is recommended as the default probabilistic head when
labels are plentiful (TS1, TS2). **MDN** may be preferable for TS3 and TS4
where training data is smaller and the mixture-component regularisation
reduces overfitting risk.

---

*Last updated: June 2026*
