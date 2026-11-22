"""Numeric input transformations; fit policy on TRAIN-standardized data only."""
from __future__ import annotations
import numpy as np

VARIANTS = ("standard", "clip_z5", "winsor_01_99", "asinh")


def fit_policy(train_standardized: np.ndarray, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    x = np.asarray(train_standardized, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("expected finite 2D train-only standardized numeric data")
    policy = {"variant": variant, "number_of_features": x.shape[1]}
    if variant == "winsor_01_99":
        policy["lower"] = np.quantile(x, .01, axis=0).tolist()
        policy["upper"] = np.quantile(x, .99, axis=0).tolist()
    return policy


def apply_policy(standardized: np.ndarray, policy: dict) -> np.ndarray:
    x = np.asarray(standardized, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != policy["number_of_features"]:
        raise ValueError("numeric feature dimension mismatch")
    variant = policy["variant"]
    if variant == "standard":
        out = x
    elif variant == "clip_z5":
        out = np.clip(x, -5, 5)
    elif variant == "winsor_01_99":
        out = np.clip(x, policy["lower"], policy["upper"])
    elif variant == "asinh":
        out = np.arcsinh(x)
    else:
        raise ValueError(f"unknown variant: {variant}")
    return out.astype(np.float32)
