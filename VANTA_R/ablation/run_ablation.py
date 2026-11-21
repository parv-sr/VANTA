
"""
VANTA-R: Ablation studies.

Run from residual_attention/:

    uv run python3 ablation/run_ablation.py

Available variants:
    no_attention  - Retrain with attention disabled
    fixed_gamma   - Retrain with gamma fixed to 1
    frozen_qk     - Freeze randomly initialized Q/K
    all           - Run all three

The original checkpoint is loaded as a reference.
All ablation outputs are stored inside ablation/<variant>/.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# Import existing architecture and dataset pipeline
# ---------------------------------------------------------

ABLATION_DIR = Path(__file__).resolve().parent
ROOT = ABLATION_DIR.parent

sys.path.insert(0, str(ROOT))

import train as diamond_train
from model import ResidualFeatureAttentionRegressor


# ---------------------------------------------------------
# No-attention baseline
# ---------------------------------------------------------

class NoAttentionBaseline(ResidualFeatureAttentionRegressor):
    """
    Uses the same categorical VALUE embeddings and readout,
    but bypasses the attention computation entirely.

    y = Linear(values)

    Q/K parameters exist for compatibility, but are frozen.
    """

    def forward(self, x_numeric, x_categorical):
        _, values = self.tokenize(x_numeric, x_categorical)
        return self.readout(values).squeeze(-1)


def freeze_query_key(model):
    for name in [
        "query_weight",
        "query_bias",
        "key_weight",
        "key_bias",
    ]:
        getattr(model, name).requires_grad_(False)

    # These embeddings only feed the Q/K pathway.
    for param in model.category_embeddings.parameters():
        param.requires_grad_(False)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Model construction
# ---------------------------------------------------------

def build_model(variant, latent_dim, cardinalities):
    kwargs = {
        "num_features": len(diamond_train.FEATURES),
        "numeric_positions": [
            diamond_train.FEATURES.index(col)
            for col in diamond_train.NUMERIC
        ],
        "categorical_positions": [
            diamond_train.FEATURES.index(col)
            for col in diamond_train.CATEGORICAL
        ],
        "categorical_cardinalities": cardinalities,
        "latent_dim": latent_dim,
    }

    if variant == "no_attention":
        model = NoAttentionBaseline(
            **kwargs,
            gamma_init=0.0,
            learnable_gamma=False,
        )
        freeze_query_key(model)

    elif variant == "fixed_gamma":
        model = ResidualFeatureAttentionRegressor(
            **kwargs,
            gamma_init=1.0,
            learnable_gamma=False,
        )

    elif variant == "frozen_qk":
        model = ResidualFeatureAttentionRegressor(
            **kwargs,
            gamma_init=0.1,
            learnable_gamma=True,
        )
        freeze_query_key(model)

    elif variant == "full":
        model = ResidualFeatureAttentionRegressor(
            **kwargs,
            gamma_init=0.1,
            learnable_gamma=True,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")

    return model


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions = []

    for x_num, x_cat, _ in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)

        output = model(x_num, x_cat)
        predictions.append(output.cpu().numpy())

    return np.concatenate(predictions)


@torch.no_grad()
def normalized_mse(model, loader, device):
    model.eval()

    total = 0.0
    count = 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        output = model(x_num, x_cat)

        total += (output - y).square().sum().item()
        count += len(y)

    return total / count


def evaluate(model, loader, actual, data, device):
    predictions = predict(model, loader, device)

    predictions = (
        predictions * data["y_std"] + data["y_mean"]
    )

    mse = mean_squared_error(actual, predictions)

    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predictions)),
        "R2": float(r2_score(actual, predictions)),
    }


def print_metrics(name, metrics):
    print(
        f"{name:22s} | "
        f"R2: {metrics['R2']:.5f} | "
        f"RMSE: ${metrics['RMSE']:,.2f} | "
        f"MAE: ${metrics['MAE']:,.2f}"
    )


# ---------------------------------------------------------
# Training one ablation
# ---------------------------------------------------------

def run_experiment(
    variant,
    args,
    data,
    loaders,
    device,
    latent_dim,
    cardinalities,
):
    print(f"\n{'=' * 65}")
    print(f"ABLATION: {variant.upper()}")
    print("=" * 65)

    # Reset RNG for comparable independent runs.
    set_seed(args.seed)

    model = build_model(
        variant, latent_dim, cardinalities
    ).to(device)

    trainable = [
        p for p in model.parameters() if p.requires_grad
    ]

    parameter_count = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()

    # Create a fresh shuffled loader for each experiment.
    generator = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        data["train"][0],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    train_history = []
    val_history = []

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(x_num, x_cat)
            loss = criterion(prediction, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in {variant}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable, max_norm=1.0
            )

            optimizer.step()

        train_loss = normalized_mse(
            model, loaders["train"], device
        )

        val_loss = normalized_mse(
            model, loaders["val"], device
        )

        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train {train_loss:.5f} | "
                f"Val {val_loss:.5f}"
            )

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    results = {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_val_normalized_mse": best_loss,
        "trainable_parameters": parameter_count,
    }

    for split in ["train", "val", "test"]:
        dataset, actual = data[split]

        results[split] = evaluate(
            model, loaders[split], actual, data, device
        )

    print(f"\nBest epoch: {best_epoch}")

    for split in ["train", "val", "test"]:
        print_metrics(
            split.upper(), results[split]
        )

    if hasattr(model, "gamma"):
        results["gamma"] = float(model.gamma.item())
        print(f"Gamma: {results['gamma']:+.5f}")

    # -----------------------------------------------------
    # Save results separately for each variant
    # -----------------------------------------------------

    output_dir = ABLATION_DIR / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label="Training MSE")
    plt.plot(val_history, label="Validation MSE")
    plt.axvline(
        best_epoch - 1,
        linestyle="--",
        color="gray",
        label=f"Best epoch: {best_epoch}",
    )

    plt.xlabel("Epoch (zero-indexed on plot)")
    plt.ylabel("Normalized MSE")
    plt.title(f"VANTA-R Ablation: {variant}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=180,
    )
    plt.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": variant,
            "latent_dim": latent_dim,
            "cardinalities": cardinalities,
            "seed": args.seed,
            "best_epoch": best_epoch,
        },
        output_dir / "checkpoint.pt",
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        choices=[
            "no_attention",
            "fixed_gamma",
            "frozen_qk",
            "all",
        ],
        default="no_attention",
    )

    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    checkpoint_path = ROOT / "attention_checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Original checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    original_seed = int(checkpoint["seed"])

    if args.seed is None:
        args.seed = original_seed
    elif args.seed != original_seed:
        raise ValueError(
            "Use the original checkpoint seed for a "
            "fair comparison with the saved reference."
        )

    latent_dim = int(checkpoint["latent_dim"])

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Original latent dimension: {latent_dim}")
    print(f"Split seed: {args.seed}")

    # Reuse the exact dataset pipeline from train.py.
    df = diamond_train.load_dataset(args.csv)
    data = diamond_train.prepare_data(df, args.seed)

    cardinalities = [
        len(data["vocabularies"][col]) + 1
        for col in diamond_train.CATEGORICAL
    ]

    # Check that the preprocessed data matches the checkpoint.
    if checkpoint["features"] != diamond_train.FEATURES:
        raise ValueError("Feature ordering differs from checkpoint.")

    if checkpoint["cardinalities"] != cardinalities:
        raise ValueError("Categorical cardinalities differ.")

    if checkpoint["vocabularies"] != data["vocabularies"]:
        raise ValueError("Category vocabularies differ.")

    loaders = {
        split: DataLoader(
            data[split][0],
            batch_size=2048,
            shuffle=False,
        )
        for split in ["train", "val", "test"]
    }

    # -----------------------------------------------------
    # Original model reference
    # -----------------------------------------------------

    original = build_model(
        "full", latent_dim, cardinalities
    )

    original.load_state_dict(
        checkpoint["model_state_dict"]
    )
    original = original.to(device)

    print("\nORIGINAL TRAINED MODEL")

    reference = {}

    for split in ["train", "val", "test"]:
        actual = data[split][1]

        reference[split] = evaluate(
            original,
            loaders[split],
            actual,
            data,
            device,
        )

        print_metrics(
            split.upper(),
            reference[split],
        )

    # -----------------------------------------------------
    # Post-training attention removal
    # -----------------------------------------------------

    muted = copy.deepcopy(original)

    with torch.no_grad():
        muted.gamma.zero_()

    print("\nORIGINAL MODEL WITH ATTENTION REMOVED")

    for split in ["val", "test"]:
        actual = data[split][1]

        metrics = evaluate(
            muted,
            loaders[split],
            actual,
            data,
            device,
        )

        print_metrics(split.upper(), metrics)

    # -----------------------------------------------------
    # Retrained ablations
    # -----------------------------------------------------

    variants = (
        ["no_attention", "fixed_gamma", "frozen_qk"]
        if args.variant == "all"
        else [args.variant]
    )

    results = {}

    for variant in variants:
        results[variant] = run_experiment(
            variant,
            args,
            data,
            loaders,
            device,
            latent_dim,
            cardinalities,
        )

    print("\n" + "=" * 65)
    print("FINAL TEST COMPARISON")
    print("=" * 65)

    print_metrics("Original VANTA-R", reference["test"])

    for variant, result in results.items():
        print_metrics(variant, result["test"])

    print(f"\nResults saved in: {ABLATION_DIR}")


if __name__ == "__main__":
    main()

