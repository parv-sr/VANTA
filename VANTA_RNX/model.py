"""VANTA-RNX v0.1: separable nonlinear Q/K maps and numeric value mappings.

Pure feature-wise maps retain the associative O(F*d) interaction aggregation.
The direct readout receives the unmodified numeric/categorical feature values.
The reference VANTA-RN implementation is imported without modifying it.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as Fnn

_REFERENCE = Path(__file__).resolve().parents[1] / "VANTA_RN" / "model.py"
_spec = importlib.util.spec_from_file_location("_vanta_rnx_rn_reference", _REFERENCE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load VANTA-RN reference {_REFERENCE}")
_ref = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ref
_spec.loader.exec_module(_ref)
ReferenceRN = _ref.NormalizedFeatureAttentionRegressor


class NonlinearFeatureAttentionRegressor(ReferenceRN):
    """Feature-map nonlinear attention with independently testable value mapping.

    kernel_map: identity | tanh | cubic
    value_map: identity | asinh | learned_asinh | gelu2

    Q/K maps are applied *after* RN's smooth L2 normalization. A norm-preserving
    rescaling preserves RN's original per-score bound after every map.
    Value maps affect *numeric* contributions to attention, not the direct path
    or categorical scalar embeddings.
    """

    MODEL_VERSION = "VANTA-RNX-v0.1"
    KERNEL_MAPS = ("identity", "tanh", "cubic")
    VALUE_MAPS = ("identity", "asinh", "learned_asinh", "gelu2")

    def __init__(self, *args, kernel_map="identity", value_map="identity", **kwargs):
        super().__init__(*args, **kwargs)
        if kernel_map not in self.KERNEL_MAPS:
            raise ValueError(f"Unsupported kernel_map={kernel_map!r}")
        if value_map not in self.VALUE_MAPS:
            raise ValueError(f"Unsupported value_map={value_map!r}")
        self.kernel_map = kernel_map
        self.value_map = value_map
        if kernel_map == "cubic":
            self.query_cubic = nn.Parameter(torch.zeros(self.latent_dim))
            self.key_cubic = nn.Parameter(torch.zeros(self.latent_dim))
        n_numeric = self.numeric_index.numel()
        if value_map == "learned_asinh":
            self.value_compression = nn.Parameter(torch.zeros(n_numeric))
        elif value_map == "gelu2":
            self.value_scale = nn.Parameter(
                torch.tensor([-1.0, 1.0]).expand(n_numeric, 2).clone()
            )
            self.value_bias = nn.Parameter(torch.zeros(n_numeric, 2))
            self.value_out = nn.Parameter(torch.zeros(n_numeric, 2))

    @staticmethod
    def _preserve_token_norm(original: torch.Tensor, mapped: torch.Tensor):
        """Retain each token's RN norm (including zero norm), without F² buffers."""
        eps = torch.finfo(original.dtype).eps
        numer = original.square().sum(dim=-1, keepdim=True) + eps
        denom = mapped.square().sum(dim=-1, keepdim=True) + eps
        return mapped * torch.sqrt(numer / denom)

    def _map_qk(self, x: torch.Tensor, *, query: bool):
        if self.kernel_map == "identity":
            return x
        if self.kernel_map == "tanh":
            mapped = torch.tanh(x)
        else:
            param = self.query_cubic if query else self.key_cubic
            alpha = 0.5 * torch.tanh(param)
            mapped = x + alpha * x.pow(3)
        return self._preserve_token_norm(x, mapped)

    def _map_values(self, values: torch.Tensor):
        if self.value_map == "identity" or self.numeric_index.numel() == 0:
            return values
        numeric = values.index_select(1, self.numeric_index)
        if self.value_map == "asinh":
            modified = torch.asinh(numeric)
        elif self.value_map == "learned_asinh":
            strength = torch.tanh(self.value_compression).unsqueeze(0)
            modified = numeric + strength * (torch.asinh(numeric) - numeric)
        else:
            hidden = Fnn.gelu(numeric.unsqueeze(-1) * self.value_scale + self.value_bias)
            modified = numeric + (hidden * self.value_out).sum(dim=-1)
        return values.index_copy(1, self.numeric_index, modified)

    def mapped_components(self, x_numeric, x_categorical):
        """Return phi(Q), psi(K), attention values U, and original direct values V."""
        q, k, v = super().project(x_numeric, x_categorical)
        return self._map_qk(q, query=True), self._map_qk(k, query=False), self._map_values(v), v

    def attention_matrix(self, x_numeric, x_categorical):
        """Diagnostic-only explicit attention; not used by the fast forward path."""
        q, k, _, _ = self.mapped_components(x_numeric, x_categorical)
        return (q @ k.transpose(-1, -2)) / math.sqrt(self.num_features)

    def forward(self, x_numeric, x_categorical, return_attention=False):
        q, k, u, v = self.mapped_components(x_numeric, x_categorical)
        summary = (k * u.unsqueeze(-1)).sum(dim=1)
        h = (q * summary.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.num_features)
        prediction = self.readout(v + self.gamma * h).squeeze(-1)
        if return_attention:
            return prediction, (q @ k.transpose(-1, -2)) / math.sqrt(self.num_features)
        return prediction
