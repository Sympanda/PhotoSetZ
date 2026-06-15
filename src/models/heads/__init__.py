"""
Prediction heads for DeepSetZ.  Each head takes the encoder embedding
(B, embed_dim) and produces a redshift prediction.

All heads implement a common interface:
    forward(embedding) → output_dict

    output_dict always contains:
        'z_pred'  : FloatTensor (B,)  — point estimate (mean or mode)
        'loss'    : FloatTensor ()    — scalar loss given 'z_true' kwarg
                                        (computed internally if 'z_true' passed)

Heads may additionally include 'z_sigma', 'log_prob', 'bins', 'probs', etc.

Usage in a training loop:
    out = head(embedding, z_true=batch_z)
    loss = out['loss']
    z_pred = out['z_pred']
"""

from .mlp_regressor import MLPRegressor
from .binned_pdf import BinnedPDF
from .mdn import MDN
from .nsf import NeuralSplineFlow

HEAD_REGISTRY = {
    "mlp_regressor": MLPRegressor,
    "binned_pdf":    BinnedPDF,
    "mdn":           MDN,
    "nsf":           NeuralSplineFlow,
}

__all__ = ["MLPRegressor", "BinnedPDF", "MDN", "NeuralSplineFlow", "HEAD_REGISTRY"]
