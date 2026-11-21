
"""
VANTA-RN v0.1
Normalized Residual Feature Attention.

Changes relative to VANTA-R:
    1. Smooth L2 normalization of Q and K.
    2. Attention aggregation scaled by sqrt(F), not sqrt(d).

No additional trainable parameters.

Mathematics:
    q_hat_i = q_i / sqrt(1 + ||q_i||_2^2)
    k_hat_i = k_i / sqrt(1 + ||k_i||_2^2)

    h = Q_hat @ (K_hat.T @ values) / sqrt(F)
    z = values + gamma * h
    prediction = readout(z)
"""

import importlib.util
import math
import sys
from pathlib import Path

import torch


# Load the actual VANTA-R model rather than duplicating it.
ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILE = ROOT / "VANTA_R" / "model.py"

if not REFERENCE_FILE.is_file():
    raise FileNotFoundError(
        f"Cannot locate VANTA-R reference: {REFERENCE_FILE}"
    )

spec = importlib.util.spec_from_file_location(
    "_vanta_r_reference_model",
    REFERENCE_FILE,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import {REFERENCE_FILE}")

reference_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference_module
spec.loader.exec_module(reference_module)

OriginalVANTA = (
    reference_module.ResidualFeatureAttentionRegressor
)


class NormalizedFeatureAttentionRegressor(OriginalVANTA):
    """
    VANTA-RN: Smooth L2-normalized Q/K attention.

    All model parameters are inherited from VANTA-R.
    """

    MODEL_VERSION = "VANTA-RN-v0.1"

    @staticmethod
    def smooth_l2_normalize(tensor):
        """
        Apply normalization independently to each
        feature token along the latent dimension.

        tensor: (batch, F, d)
        """
        squared_norm = tensor.square().sum(
            dim=-1,
            keepdim=True,
        )

        return tensor * torch.rsqrt(1.0 + squared_norm)

    def project_raw(self, x_numeric, x_categorical):
        """
        Compute exactly the original affine projections
        before normalization.

        Returns:
            q_raw:  (batch, F, d)
            k_raw:  (batch, F, d)
            values: (batch, F)
        """

        tokens, values = self.tokenize(
            x_numeric,
            x_categorical,
        )

        q_raw = (
            tokens * self.query_weight
            + self.query_bias
        )

        k_raw = (
            tokens * self.key_weight
            + self.key_bias
        )

        return q_raw, k_raw, values

    def project(self, x_numeric, x_categorical):
        """
        Return normalized query/key vectors.
        """

        q_raw, k_raw, values = self.project_raw(
            x_numeric,
            x_categorical,
        )

        q = self.smooth_l2_normalize(q_raw)
        k = self.smooth_l2_normalize(k_raw)

        return q, k, values

    def attention_matrix(self, x_numeric, x_categorical):
        """
        Explicit attention matrix for diagnostics.

        A = Q_hat @ K_hat.T / sqrt(F)

        Returns:
            attention: (batch, F, F)
        """

        q, k, _ = self.project(
            x_numeric,
            x_categorical,
        )

        num_features = q.shape[1]

        return (
            q @ k.transpose(-1, -2)
        ) / math.sqrt(num_features)

    def forward(
        self,
        x_numeric,
        x_categorical,
        return_attention=False,
    ):
        q, k, values = self.project(
            x_numeric,
            x_categorical,
        )

        num_features = values.shape[1]

        # Associative interaction aggregation:
        #
        # h = Q_hat @ (K_hat.T @ values) / sqrt(F)
        #
        # No F x F matrix is constructed here.

        key_summary = (
            k * values.unsqueeze(-1)
        ).sum(dim=1)

        h = (
            q * key_summary.unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(num_features)

        # Original scalar-gated residual.
        z = values + self.gamma * h

        prediction = self.readout(z).squeeze(-1)

        if return_attention:
            attention = (
                q @ k.transpose(-1, -2)
            ) / math.sqrt(num_features)

            return prediction, attention

        return prediction

