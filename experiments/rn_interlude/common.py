
"""Shared builders for VANTA-RN diagnostics and tuning."""

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EX03 = ROOT / "experiments" / "datasets_ex03"
EX04 = ROOT / "experiments" / "normalization_ex04"
RN_DIR = ROOT / "VANTA_RN"
OUT = Path(__file__).resolve().parent / "results"

for path in (EX03, RN_DIR):
    sys.path.insert(0, str(path))

from datasets import load_dataset, preprocess
from model import NormalizedFeatureAttentionRegressor


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(data, latent_dim=24, gamma_init=0.1):
    n_numeric = len(data["numeric"])
    n_features = len(data["features"])

    return NormalizedFeatureAttentionRegressor(
        num_features=n_features,
        numeric_positions=list(range(n_numeric)),
        categorical_positions=list(
            range(n_numeric, n_features)
        ),
        categorical_cardinalities=data["cardinalities"],
        latent_dim=latent_dim,
        gamma_init=gamma_init,
        learnable_gamma=True,
    )


def model_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def load_ex04_model(data, dataset, seed, device):
    path = (
        EX04 / "results" / dataset
        / f"seed_{seed}" / "vanta_rn" / "checkpoint.pt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing VANTA-RN checkpoint: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    for key in ("cardinalities", "vocabularies"):
        if checkpoint[key] != data[key]:
            raise ValueError(
                f"Checkpoint {key} differs from data split."
            )

    if not np.isclose(
        checkpoint["target_mean"],
        data["target_mean"],
    ):
        raise ValueError("Checkpoint target mean mismatch.")

    if not np.isclose(
        checkpoint["target_std"],
        data["target_std"],
    ):
        raise ValueError("Checkpoint target std mismatch.")

    model = make_model(
        data,
        latent_dim=checkpoint.get("latent_dim", 24),
    )

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )

    return model.to(device).eval(), path

