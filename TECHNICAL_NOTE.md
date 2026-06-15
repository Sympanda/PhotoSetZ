# DeepSetZ: Set-Based Photometric Redshift Estimation

*James et al. — internal technical note*

---

## Motivation

Standard photometric redshift codes require a fixed-length input vector, forcing catalogue builders to decide upfront which filters constitute the "input space." This is increasingly limiting given heterogeneous multi-survey datasets where different galaxies have wildly different filter coverage (DECaLS g/r/z vs. full LSST+Roman+Euclid+WISE).

DeepSetZ replaces the fixed-vector paradigm with a **set-based representation** in which each galaxy is described by however many filter measurements it actually has. The model learns:

$$p(z \mid X_g) = p\!\left(z \;\Big|\; \{m_i,\, \lambda_{\mathrm{eff},i},\, \Delta\lambda_i,\, s_i\}_{i \in S_g}\right)$$

where $S_g \subseteq \mathcal{F}$ is the subset of filters available for galaxy $g$. Absent filters are simply not in the set — there is no imputation, masking with sentinel values, or fixed-dimension assumption.

---

## Method

### Galaxy representation

A **filter registry** $\mathcal{F}$ stores static per-filter metadata, computed from transmission curves where available:

$$F_j = \left[\lambda_{\mathrm{eff},j},\; \Delta\lambda_j,\; s_j\right]$$

with $\lambda_{\mathrm{eff}}$ the flux-weighted mean wavelength, $\Delta\lambda$ the RMS bandwidth, and $s_j \in \{0,1,2,3\}$ a survey identifier. Each observed filter measurement becomes a **token**:

$$x_i = \left[\frac{m_i - 25}{3},\;\; \frac{\log_{10}\lambda_{\mathrm{eff},i} - 4.0}{0.7},\;\; \frac{\log_{10}\Delta\lambda_i - 3.0}{0.5},\;\; \frac{s_i}{3}\right] \in \mathbb{R}^4$$

The magnitude is normalised around a reference of 25 AB mag. Wavelengths are in log-space to compress the large dynamic range from UV to mid-IR. Tokens are ordered by $\lambda_{\mathrm{eff}}$ for consistency, though the architectures are permutation-invariant by construction.

**Current token dimension: 4.** When realistic noise is available the token will be extended to include $\sigma_{m,i}$, SNR$_i$, a detection flag $d_i$, and limiting magnitude $m_{\mathrm{lim},i}$, giving:

$$x_i = \left[m_i,\; \sigma_{m,i},\; \mathrm{SNR}_i,\; d_i,\; m_{\mathrm{lim},i},\; \lambda_{\mathrm{eff},i},\; \Delta\lambda_i,\; s_i\right]$$

This extension requires no architectural changes — only the first linear layer of the encoder needs updating.

### Filter coverage

16 filters spanning 0.37–4.6 µm, covering four surveys:

| Survey | Filters | $\lambda_{\mathrm{eff}}$ range |
|---|---|---|
| LSST | u g r i z y | 3671–9710 Å |
| Roman | Y106 J129 H158 F184 K213 | 1.06–2.13 µm |
| Euclid | Y J H | 1.06–1.77 µm |
| WISE | W1 W2 | 3.35–4.60 µm |

### Encoder architectures

**DeepSets** applies a shared per-token MLP $\phi$ independently to each token, pools, then decodes with a second MLP $\rho$:

$$h_g = \rho\!\left(\mathrm{Pool}_{i \in S_g}\{\phi(x_i)\}\right)$$

The pooling operation is either a masked mean or a learned attention-weighted sum over token encodings. The attention variant uses a lightweight score network ($\phi\text{-dim} \to 1$) and adds negligible parameters while allowing the model to dynamically weight filter importance.

**Set Transformer** instead applies multi-head self-attention across all tokens before pooling, allowing token representations to condition on the full available filter set:

$$H_g = \mathrm{TransformerEncoder}\!\left(\{W_e\, x_i\}_{i \in S_g}\right), \qquad h_g = \mathrm{PMA}(H_g)$$

where PMA (Pooling by Multi-head Attention) uses a learned seed vector to aggregate $H_g$ into a fixed-size embedding. This lets the model learn cross-filter structure — colour ratios, SED breaks, Lyman dropout — directly in attention space, rather than inferring it post-pooling.

### Prediction head

A small MLP regressor maps $h_g \to \hat{z}$, trained with Huber loss on $\log(1+z)$ targets:

$$\mathcal{L} = \mathrm{Huber}_\delta\!\left(\log(1+\hat{z}),\; \log(1+z_{\mathrm{spec}})\right)$$

Training in log-redshift space reduces tail bias by re-weighting the loss: a prediction error at $z=2$ is penalised proportionally more than the same absolute error at $z=0.2$. All reported metrics are in linear $z$.

Alternative heads (binned softmax posterior, Gaussian mixture network) are implemented and swappable via config.

### Training strategy

The model is trained on 175k galaxies (stratified 90/10 split from a combined 200k catalogue). The key augmentation is **stratified filter dropout**, which exposes each training step to a randomly sampled filter coverage scenario:

| Mode | Probability | Description |
|---|---|---|
| Complete | 15% | All 16 filters |
| Survey preset | 25% | Named realistic combinations (DECaLS g/r/z, LSST-only, LSST+Roman, etc.) |
| Survey drop | 25% | Drop 1–2 entire surveys; optionally thin further |
| Aggressive | 35% | Per-filter dropout rate drawn from $\mathcal{U}(0.3, 0.8)$ |

A hard floor of 3 filters is always enforced. This ensures the model is simultaneously competitive at full depth and robust down to DECaLS-like sparse coverage.

---

## Experiments

### Dataset

Noiseless mock galaxy catalogues generated with DESC DC2 / Roman simulation (courtesy E. Gall). Redshift range $z \in [0.006, 2.29]$, median $z \approx 0.77$. The train/test split is stratified in 20 equal-frequency redshift bins to ensure representative tail statistics.

| Split | N |
|---|---|
| Train | 157,500 |
| Validation | 17,500 |
| Test | 25,000 |

### Results

All models use an MLP regressor head with Huber loss. Metrics are on the held-out test set, evaluated with all 16 filters present.

| Run | Encoder | Pooling | log$(1+z)$ | $\sigma_\mathrm{NMAD}$ | Bias | Outlier% | RMSE |
|---|---|---|---|---|---|---|---|
| DeepSets (mean) | DeepSets | Mean | — | 0.0216 | +0.0006 | 0.66% | 0.0365 |
| DeepSets (attention) | DeepSets | Attention | — | 0.0183 | +0.0025 | 0.50% | 0.0322 |
| DeepSets (attention + log$z$) | DeepSets | Attention | ✓ | 0.0194 | +0.0020 | 0.26% | 0.0296 |
| DeepSets (scaled, log$z$) | DeepSets | Attention | ✓ | **0.0172** | +0.0026 | **0.19%** | **0.0260** |

Key observations:
- Switching from mean to learned attention pooling reduces $\sigma_\mathrm{NMAD}$ by ~15% with negligible parameter cost.
- Training in log$(1+z)$ space more than halves the outlier rate (0.50% → 0.26%) with a modest impact on $\sigma_\mathrm{NMAD}$.
- A modestly scaled model (~500k parameters) achieves $\sigma_\mathrm{NMAD} = 0.017$ and an outlier rate of 0.19%.
- The Set Transformer reaches equivalent DeepSets performance in roughly $5\times$ fewer epochs, consistent with cross-filter attention providing a richer gradient signal from the first epoch.

### Survey-subset robustness

Each model is evaluated on all $2^4 - 1 = 15$ non-empty subsets of the four survey groups, plus named presets. As expected, performance degrades gracefully with fewer filters. Key findings:

- **LSST-only (6 bands):** $\sigma_\mathrm{NMAD} \approx 0.020$–$0.025$, outlier $\sim 0.3$–$0.5\%$
- **DECaLS-like (g/r/z only):** $\sigma_\mathrm{NMAD} \approx 0.030$–$0.040$, outlier $\sim 1$–$2\%$
- **LSST + Roman:** best single-combination performance
- **WISE-only:** model recovers a coarse redshift estimate but scatter is large, as expected

---

## Limitations and next steps

**Current data:** All training is on noiseless mocks. Real photometry will add per-band magnitude errors, SNR, detection flags, and limiting magnitudes to each token — the full token schema from the design. This is expected to be the largest lever on real-data performance.

**Planned:**

1. **Noisy data** — extend token to 8 dimensions; retrain. The set architecture is unchanged.
2. **Set Transformer scaling** — current small model (128-dim, 2 layers) is clearly not at capacity; scale to 256–512-dim with 4–6 layers.
3. **Posterior head** — swap MLP regressor for binned softmax or MDN to output full $p(z)$ rather than a point estimate.
4. **Learned filter embeddings** — replace the scalar $s_i$ survey ID with a trainable per-filter embedding, allowing the model to learn instrument-specific systematics.

---

*Code: `github.com/[repo]`. Contact: [email].*
