"""Approach A: residual raw-attention regression with learned category embeddings.

The model keeps ONE token and ONE value per original spreadsheet column.

Numerical column i:
    e_i = x_i * ones(d)
    v_i = x_i                     (after training-set standardization)

Categorical column i:
    e_i = Embedding_i(category)   (a trainable d-dimensional vector)
    v_i = ValueEmbedding_i(category)  (a trainable scalar)

For either kind of feature:
    q_i = e_i * Wq[i] + bq[i]     (elementwise, separate affine Q/K maps)
    k_i = e_i * Wk[i] + bk[i]

    A = Q @ K.T / sqrt(d)        (raw asymmetric attention, NO softmax)
    h = A @ v                    (input values, NO learned dense V projection)
    z = v + gamma * h            (feature-space residual)
    y = Linear(z)                (one output, NO MLP/FFN)

The forward pass uses associativity: h = Q @ (K.T @ v) / sqrt(d), avoiding
materializing the F-by-F attention matrix. attention_matrix() is available
for inspection.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class ResidualFeatureAttentionRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        numeric_positions: Sequence[int],
        categorical_positions: Sequence[int],
        categorical_cardinalities: Sequence[int],
        latent_dim: int = 4,
        gamma_init: float = 0.1,
        learnable_gamma: bool = True,
    ) -> None:
        super().__init__()
        if num_features < 1 or latent_dim < 1:
            raise ValueError("num_features and latent_dim must be positive")
        if len(categorical_positions) != len(categorical_cardinalities):
            raise ValueError("Provide one cardinality per categorical feature")
        if any(card < 2 for card in categorical_cardinalities):
            raise ValueError("Each category embedding requires >=2 indices (including unknown 0)")
        all_positions = list(numeric_positions) + list(categorical_positions)
        if sorted(all_positions) != list(range(num_features)):
            raise ValueError("Numeric and categorical positions must partition 0..num_features-1")

        self.num_features = num_features
        self.latent_dim = latent_dim
        # Register indices as buffers, so .to(device) moves them with the module.
        self.register_buffer("numeric_index", torch.tensor(list(numeric_positions), dtype=torch.long))
        self.register_buffer("categorical_index", torch.tensor(list(categorical_positions), dtype=torch.long))

        # A distinct affine Q and K map for EVERY feature, including bias terms.
        self.query_weight = nn.Parameter(torch.empty(num_features, latent_dim))
        self.query_bias = nn.Parameter(torch.zeros(num_features, latent_dim))
        self.key_weight = nn.Parameter(torch.empty(num_features, latent_dim))
        self.key_bias = nn.Parameter(torch.zeros(num_features, latent_dim))

        # Category ID 0 means unseen at training time. Its embedding stays zero.
        # Q/K receive a learned VECTOR; values receive a learned SCALAR.
        self.category_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, latent_dim, padding_idx=0)
            for cardinality in categorical_cardinalities
        ])
        self.category_value_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, 1, padding_idx=0)
            for cardinality in categorical_cardinalities
        ])

        gamma = torch.tensor(float(gamma_init))
        if learnable_gamma:
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer("gamma", gamma)
        self.readout = nn.Linear(num_features, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_weight, mean=0.0, std=0.2)
        nn.init.normal_(self.key_weight, mean=0.0, std=0.2)
        nn.init.zeros_(self.query_bias)
        nn.init.zeros_(self.key_bias)
        for emb in self.category_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.5)
            with torch.no_grad():
                emb.weight[0].zero_()
        for emb in self.category_value_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.5)
            with torch.no_grad():
                emb.weight[0].zero_()
        self.readout.reset_parameters()

    def tokenize(
        self, x_numeric: torch.Tensor, x_categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feature tokens (B,F,d) and corresponding values (B,F)."""
        if x_numeric.ndim != 2 or x_categorical.ndim != 2:
            raise ValueError("Expected numeric and categorical tensors of shape (batch, columns)")
        batch = x_numeric.shape[0]
        if x_categorical.shape[0] != batch:
            raise ValueError("Numerical and categorical batch sizes must match")
        if x_numeric.shape[1] != self.numeric_index.numel():
            raise ValueError("Incorrect number of numerical columns")
        if x_categorical.shape[1] != self.categorical_index.numel():
            raise ValueError("Incorrect number of categorical columns")

        tokens = x_numeric.new_zeros(batch, self.num_features, self.latent_dim)
        values = x_numeric.new_zeros(batch, self.num_features)

        if self.numeric_index.numel():
            num_tokens = x_numeric.unsqueeze(-1).expand(-1, -1, self.latent_dim)
            tokens = tokens.index_copy(1, self.numeric_index, num_tokens)
            values = values.index_copy(1, self.numeric_index, x_numeric)

        if self.categorical_index.numel():
            cat_tokens = torch.stack([
                emb(x_categorical[:, j])
                for j, emb in enumerate(self.category_embeddings)
            ], dim=1)
            cat_values = torch.stack([
                emb(x_categorical[:, j]).squeeze(-1)
                for j, emb in enumerate(self.category_value_embeddings)
            ], dim=1)
            tokens = tokens.index_copy(1, self.categorical_index, cat_tokens)
            values = values.index_copy(1, self.categorical_index, cat_values)

        return tokens, values

    def project(
        self, x_numeric: torch.Tensor, x_categorical: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, values = self.tokenize(x_numeric, x_categorical)
        q = tokens * self.query_weight + self.query_bias
        k = tokens * self.key_weight + self.key_bias
        return q, k, values

    def attention_matrix(
        self, x_numeric: torch.Tensor, x_categorical: torch.Tensor
    ) -> torch.Tensor:
        """Raw (not softmaxed) attention scores of shape (batch,F,F)."""
        q, k, _ = self.project(x_numeric, x_categorical)
        return (q @ k.transpose(-1, -2)) / math.sqrt(self.latent_dim)

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q, k, values = self.project(x_numeric, x_categorical)
        # A @ values = Q @ (K.T @ values) / sqrt(d)
        key_summary = (k * values.unsqueeze(-1)).sum(dim=1)  # (batch,d)
        h = (q * key_summary.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.latent_dim)
        z = values + self.gamma * h
        predictions = self.readout(z).squeeze(-1)
        if return_attention:
            return predictions, (q @ k.transpose(-1, -2)) / math.sqrt(self.latent_dim)
        return predictions

