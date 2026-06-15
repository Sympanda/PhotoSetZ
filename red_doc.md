# Flexible Set-Based Photometric Redshift Framework

The proposed framework treats photometric redshift estimation as inference from a variable set of photometric constraints, rather than from a fixed catalogue vector. Each filter is viewed as a known measurement operator on the galaxy SED, while each galaxy is represented by the subset of filter measurements available for that object.

## Filter Registry

We define a global filter registry:

$$
\mathcal{F} = \{F_1, F_2, \ldots, F_M\},
$$

where each filter \(F_j\) has static metadata:

$$
F_j =
\left[
\lambda_{\mathrm{eff},j},
\Delta\lambda_j,
s_j,
e_{\mathrm{filter},j},
e_{\mathrm{survey},j}
\right].
$$

Here \(\lambda_{\mathrm{eff},j}\) is the effective wavelength, \(\Delta\lambda_j\) is a bandwidth summary, \(s_j\) denotes optional static survey or instrumental metadata, and \(e_{\mathrm{filter},j}\) and \(e_{\mathrm{survey},j}\) are optional learned embeddings for the filter and survey/instrument.

The registry can be extended when new filters are introduced. If learned filter embeddings are used, new filters require new embedding entries, which can be initialised from nearby filters in wavelength and fine-tuned using overlapping data.

## Galaxy Photometry as a Set

For a given galaxy \(g\), only a subset of filters may be available:

$$
S_g \subseteq \mathcal{F}.
$$

The galaxy is represented as a variable-length set of observed photometric tokens:

$$
X_g = \{x_i : i \in S_g\}.
$$

Each token corresponds to one object-specific measurement through one filter:

$$
x_i =
\left[
m_i,
\sigma_{m,i},
\mathrm{SNR}_i,
d_i,
m_{\mathrm{lim},i},
\lambda_{\mathrm{eff},i},
\Delta\lambda_i,
e_{\mathrm{filter},i},
e_{\mathrm{survey},i}
\right].
$$

Here \(m_i\) is the measured AB magnitude, \(\sigma_{m,i}\) is the magnitude uncertainty, \(\mathrm{SNR}_i\) is the signal-to-noise ratio, \(d_i\) is a detection or upper-limit flag, and \(m_{\mathrm{lim},i}\) is the limiting magnitude or local depth.

Filters that were not observed are absent from the set. Filters that were observed but yielded weak or non-detections remain in the set, with \(d_i=0\) and the corresponding limiting magnitude and uncertainty information included. This distinction is important because non-detections provide useful constraints on SED breaks and dropout behaviour.

Tokens are ordered by effective wavelength for convenience:

$$
\lambda_{\mathrm{eff},1} < \lambda_{\mathrm{eff},2} < \cdots < \lambda_{\mathrm{eff},N_g},
$$

where \(N_g = |S_g|\) is the number of available measurements for galaxy \(g\). However, filter identity is not encoded by token position alone; it is provided explicitly through wavelength metadata and optional learned embeddings.

## Set Encoder

The model learns a mapping:

$$
f_\theta : X_g \mapsto h_g,
$$

where \(h_g\) is a learned representation of the galaxy's available photometric constraints.

A simple DeepSets encoder applies a shared token network \(\phi\) to each measurement and pools over the available tokens:

$$
h_i = \phi(x_i),
$$

$$
h_g = \rho\left(\mathrm{Pool}\{h_i : i \in S_g\}\right),
$$

where \(\mathrm{Pool}\) may be a mean, sum, max, or attention-weighted pooling operation. This formulation is naturally robust to missing filters because it does not require a fixed input vector.

A Set Transformer variant instead allows filter tokens to interact before pooling:

$$
H_g = \mathrm{TransformerEncoder}\left(\{h_i : i \in S_g\}\right),
$$

$$
h_g = \mathrm{AttentionPool}(H_g).
$$

This allows the model to learn colour-like relations, breaks, and cross-filter dependencies directly from the available filter set, without requiring hand-defined colour indices.

## Redshift Posterior Head

The final model predicts a redshift posterior:

$$
p_\theta(z \mid X_g).
$$

A stable first choice is a binned posterior over \(K\) redshift bins:

$$
p_\theta(z_k \mid X_g)
=
\mathrm{softmax}_k(W h_g + b),
\qquad
k = 1,\ldots,K.
$$

For example, for \(0 \leq z \leq 2.5\), one may use \(K=100\) bins.

A compact continuous alternative is a mixture density network:

$$
p_\theta(z \mid X_g)
=
\sum_{c=1}^{C}
\pi_c(X_g)
\mathcal{N}
\left(
z \mid \mu_c(X_g), \sigma_c^2(X_g)
\right),
$$

where the mixture weights satisfy:

$$
\sum_{c=1}^{C} \pi_c(X_g) = 1.
$$

The MDN head provides a continuous multimodal posterior, while the binned PDF head is usually simpler to train and calibrate.

## Training Strategy

The model is trained to infer redshift posteriors from incomplete and heterogeneous photometry:

$$
\mathcal{L}
=
-\log p_\theta(z_{\mathrm{spec}} \mid X_g).
$$

To encourage robustness to missing filters and survey variation, training can include random filter dropout, survey-like filter subsets, depth/error perturbations, rare-filter oversampling, and optional filter-embedding dropout. The latter forces the model to rely on physical filter metadata rather than memorising filter identities alone.

## Extending to New Filter Systems

New filters can be incorporated by adding entries to the filter registry. If a new filter \(F_{\mathrm{new}}\) is introduced, its known metadata are added:

$$
F_{\mathrm{new}}
=
\left[
\lambda_{\mathrm{eff,new}},
\Delta\lambda_{\mathrm{new}},
s_{\mathrm{new}},
e_{\mathrm{filter,new}},
e_{\mathrm{survey,new}}
\right].
$$

The learned embedding \(e_{\mathrm{filter,new}}\) can be initialised from filters with similar \(\lambda_{\mathrm{eff}}\) or from a metadata-based encoder, then fine-tuned on galaxies observed with the new system.

The main network does not require architectural redesign as long as the token schema is unchanged. Adding a filter changes the number of possible tokens, not the per-token feature dimension. This makes the framework naturally extensible to new surveys, narrow-band additions, or heterogeneous multi-instrument photometry.

## Motivation

This formulation is inspired by template fitting, but replaces explicit template matching with a learned posterior model. Instead of requiring a fixed vector of magnitudes, the model conditions on the available photometric constraints, their uncertainties, their wavelength coverage, and their observing depth:

$$
p(z \mid X_g)
=
p\left(
z \mid
\{m_i, \sigma_{m,i}, \mathrm{SNR}_i, d_i,
m_{\mathrm{lim},i}, \lambda_{\mathrm{eff},i}, \Delta\lambda_i\}_{i=1}^{N_g}
\right).
$$

This provides a flexible neural analogue of SED fitting for photometric redshift estimation under missing, heterogeneous, and evolving filter coverage.
