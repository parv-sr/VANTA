
"""
VANTA-G: Gated Residual Feature Attention for Regression.

One token and one scalar value per original input feature.

Numerical features:
    token_i = x_i * ones(d)
    value_i = x_i

Categorical features:
    token_i = learned embedding (d dimensions)
    value_i = learned scalar embedding

Attention:
    q_i = token_i * Wq[i] + bq[i]
    k_i = token_i * Wk[i] + bk[i]

    A = Q @ K.T / sqrt(d)
    h = A @ values

Feature-wise gated residual:
    z = values + gamma * h

Regression:
    y_hat = Linear(z)

No softmax, FFN, MLP, or learned dense value projection.
"""

import math
from collections.abc import Sequence

import torch
from torch import nn


class GatedFeatureAttentionRegressor(nn.Module):

    def __init__(
        self,
        num_features: int,
        numeric_positions: Sequence[int],
        categorical_positions: Sequence[int],
        categorical_cardinalities: Sequence[int],
        latent_dim: int = 24,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        if num_features < 1 or latent_dim < 1:
            raise ValueError("Feature count and latent dimension must be positive.")

        if len(categorical_positions) != len(categorical_cardinalities):
            raise ValueError("Every categorical feature needs a cardinality.")

        positions = list(numeric_positions) + list(categorical_positions)

        if sorted(positions) != list(range(num_features)):
            raise ValueError("Feature positions must partition the input features.")

        self.num_features = num_features
        self.latent_dim = latent_dim

        self.register_buffer(
            "numeric_index",
            torch.tensor(list(numeric_positions), dtype=torch.long)
        )

        self.register_buffer(
            "categorical_index",
            torch.tensor(list(categorical_positions), dtype=torch.long)
        )

        # Separate affine Q/K parameters for every feature.
        self.query_weight = nn.Parameter(
            torch.empty(num_features, latent_dim)
        )
        self.query_bias = nn.Parameter(
            torch.zeros(num_features, latent_dim)
        )

        self.key_weight = nn.Parameter(
            torch.empty(num_features, latent_dim)
        )
        self.key_bias = nn.Parameter(
            torch.zeros(num_features, latent_dim)
        )

        # Learned categorical token representations.
        self.category_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, latent_dim, padding_idx=0)
            for cardinality in categorical_cardinalities
        ])

        # Learned scalar values for categorical features.
        self.category_value_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, 1, padding_idx=0)
            for cardinality in categorical_cardinalities
        ])

        # VANTA-G's distinguishing component:
        # One learnable gate per original feature.
        self.gamma = nn.Parameter(
            torch.full((num_features,), float(gamma_init))
        )

        # No hidden layers: simple regression readout.
        self.readout = nn.Linear(num_features, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_weight, mean=0.0, std=0.2)
        nn.init.normal_(self.key_weight, mean=0.0, std=0.2)

        nn.init.zeros_(self.query_bias)
        nn.init.zeros_(self.key_bias)

        for embedding in self.category_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.5)
            with torch.no_grad():
                embedding.weight[0].zero_()

        for embedding in self.category_value_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.5)
            with torch.no_grad():
                embedding.weight[0].zero_()

        self.readout.reset_parameters()

    def tokenize(self, x_numeric, x_categorical):
        """
        Convert inputs into:
            tokens: (batch, F, d)
            values: (batch, F)
        """

        if x_numeric.ndim != 2 or x_categorical.ndim != 2:
            raise ValueError("Expected two-dimensional input tensors.")

        batch_size = x_numeric.size(0)

        if x_categorical.size(0) != batch_size:
            raise ValueError("Numerical and categorical batch sizes differ.")

        if x_numeric.size(1) != self.numeric_index.numel():
            raise ValueError("Incorrect number of numerical columns.")

        if x_categorical.size(1) != self.categorical_index.numel():
            raise ValueError("Incorrect number of categorical columns.")

        tokens = x_numeric.new_zeros(
            batch_size, self.num_features, self.latent_dim
        )

        values = x_numeric.new_zeros(
            batch_size, self.num_features
        )

        # Numerical features: repeat each scalar across d dimensions.
        if self.numeric_index.numel() > 0:
            num_tokens = x_numeric.unsqueeze(-1).expand(
                -1, -1, self.latent_dim
            )

            tokens = tokens.index_copy(
                1, self.numeric_index, num_tokens
            )

            values = values.index_copy(
                1, self.numeric_index, x_numeric
            )

        # Categorical features: trainable embedding lookups.
        if self.categorical_index.numel() > 0:
            cat_tokens = torch.stack([
                embedding(x_categorical[:, i])
                for i, embedding in enumerate(self.category_embeddings)
            ], dim=1)

            cat_values = torch.stack([
                embedding(x_categorical[:, i]).squeeze(-1)
                for i, embedding in enumerate(
                    self.category_value_embeddings
                )
            ], dim=1)

            tokens = tokens.index_copy(
                1, self.categorical_index, cat_tokens
            )

            values = values.index_copy(
                1, self.categorical_index, cat_values
            )

        return tokens, values

    def project(self, x_numeric, x_categorical):
        tokens, values = self.tokenize(x_numeric, x_categorical)

        q = tokens * self.query_weight + self.query_bias
        k = tokens * self.key_weight + self.key_bias

        return q, k, values

    def attention_matrix(self, x_numeric, x_categorical):
        """
        Explicit attention matrix for analysis.

        Shape: (batch, F, F)
        """
        q, k, _ = self.project(x_numeric, x_categorical)

        return (
            q @ k.transpose(-1, -2)
        ) / math.sqrt(self.latent_dim)

    def forward(
        self,
        x_numeric,
        x_categorical,
        return_attention=False,
    ):
        q, k, values = self.project(x_numeric, x_categorical)

        # Linear attention:
        # h = Q @ (K.T @ values) / sqrt(d)
        #
        # Avoid constructing the F x F attention matrix.

        key_summary = (
            k * values.unsqueeze(-1)
        ).sum(dim=1)

        h = (
            q * key_summary.unsqueeze(1)
        ).sum(dim=-1) / math.sqrt(self.latent_dim)

        # Feature-wise residual gating.
        # gamma has shape (F,) and broadcasts across the batch.
        z = values + self.gamma * h

        prediction = self.readout(z).squeeze(-1)

        if return_attention:
            attention = (
                q @ k.transpose(-1, -2)
            ) / math.sqrt(self.latent_dim)

            return prediction, attention

        return prediction

