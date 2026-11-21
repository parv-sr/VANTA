
"""
VANTA - Experiment 02: Multi-Seed Robustness

Models:
    VANTA-R
    VANTA-G
    Parameter-matched Embedding MLP

The dataset split is fixed at seed 42.
Model initialization and batch shuffling vary by seed.

Run from seeds_ex02/:
    uv run python3 run_seeds.py

Quick run:
    uv run python3 run_seeds.py --seeds 42 43 44
"""

import argparse
import copy
import importlib.util
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------
# 1. Project paths and model imports
# ---------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

R_DIR = ROOT / "VANTA_R"
G_DIR = ROOT / "VANTA_G"
OUT = HERE / "results"

OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(R_DIR))

import train as vanta_data

RModel = vanta_data.ResidualFeatureAttentionRegressor


def load_g_model():
    path = G_DIR / "model.py"

    spec = importlib.util.spec_from_file_location(
        "vanta_g_ex02", path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.GatedFeatureAttentionRegressor


GModel = load_g_model()

FEATURES = vanta_data.FEATURES
NUMERIC = vanta_data.NUMERIC
CATEGORICAL = vanta_data.CATEGORICAL

MODELS = ["VANTA-R", "VANTA-G", "Embedding MLP"]

EMBED_DIM = 8


# ---------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ---------------------------------------------------------
# 3. Parameter-matched MLP from Experiment 01
# ---------------------------------------------------------

class EmbeddingMLP(nn.Module):

    def __init__(self, cardinalities, hidden_dim):
        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(card, EMBED_DIM, padding_idx=0)
            for card in cardinalities
        ])

        input_dim = (
            len(NUMERIC) + len(cardinalities) * EMBED_DIM
        )

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_num, x_cat):
        embeddings = [
            embedding(x_cat[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]

        x = torch.cat([x_num] + embeddings, dim=1)

        return self.network(x).squeeze(-1)


def matched_mlp_width(target_params, cardinalities):

    embedding_params = sum(cardinalities) * EMBED_DIM

    input_dim = (
        len(NUMERIC) + len(cardinalities) * EMBED_DIM
    )

    width = round(
        (target_params - embedding_params - 1)
        / (input_dim + 2)
    )

    return max(1, width)


# ---------------------------------------------------------
# 4. Construct models
# ---------------------------------------------------------

def build_model(name, seed, cardinalities, latent_dim,
                gamma_init, mlp_width):

    set_seed(seed)

    kwargs = {
        "num_features": len(FEATURES),
        "numeric_positions": [
            FEATURES.index(col) for col in NUMERIC
        ],
        "categorical_positions": [
            FEATURES.index(col) for col in CATEGORICAL
        ],
        "categorical_cardinalities": cardinalities,
        "latent_dim": latent_dim,
        "gamma_init": gamma_init,
    }

    if name == "VANTA-R":
        return RModel(
            **kwargs,
            learnable_gamma=True,
        )

    if name == "VANTA-G":
        return GModel(**kwargs)

    if name == "Embedding MLP":
        return EmbeddingMLP(
            cardinalities, mlp_width
        )

    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_mse(model, loader, device):
    model.eval()

    total = 0.0
    count = 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        prediction = model(x_num, x_cat)

        total += (prediction - y).square().sum().item()
        count += len(y)

    return total / count


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    outputs = []

    for x_num, x_cat, _ in loader:
        output = model(
            x_num.to(device),
            x_cat.to(device),
        )
        outputs.append(output.cpu().numpy())

    return np.concatenate(outputs)


def calculate_metrics(pred_scaled, actual, data):
    predicted = (
        pred_scaled * data["y_std"] + data["y_mean"]
    )

    mse = mean_squared_error(actual, predicted)

    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
    }


# ---------------------------------------------------------
# 6. Plot individual loss curves
# ---------------------------------------------------------

def save_loss_curve(history, best_epoch, name, seed, directory):
    epochs = np.arange(1, len(history["train"]) + 1)

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history["train"],
        label="Training MSE",
    )

    plt.plot(
        epochs,
        history["val"],
        label="Validation MSE",
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        color="gray",
        label=f"Best epoch: {best_epoch}",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title(f"{name} | Seed {seed}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    filename = name.replace(" ", "_").replace("-", "_")

    plt.savefig(
        directory / f"{filename}_loss.png",
        dpi=180,
    )
    plt.close()


# ---------------------------------------------------------
# 7. Train a single model with a single seed
# ---------------------------------------------------------

def run_one(name, seed, args, data, loaders,
            cardinalities, latent_dim, mlp_width, device):

    print(f"\n{'=' * 65}")
    print(f"MODEL: {name} | SEED: {seed}")
    print("=" * 65)

    model = build_model(
        name,
        seed,
        cardinalities,
        latent_dim,
        args.gamma_init,
        mlp_width,
    ).to(device)

    params = count_parameters(model)
    print(f"Trainable parameters: {params:,}")

    # New generator per model ensures identical shuffle order
    # for a given seed, regardless of model architecture.
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        data["train"][0],
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    history = {
        "train": [],
        "val": [],
    }

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
                    f"Non-finite loss: {name}, seed {seed}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        train_mse = evaluate_mse(
            model, loaders["train"], device
        )

        val_mse = evaluate_mse(
            model, loaders["val"], device
        )

        history["train"].append(train_mse)
        history["val"].append(val_mse)

        if val_mse < best_loss - 1e-7:
            best_loss = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train MSE: {train_mse:.5f} | "
                f"Val MSE: {val_mse:.5f}"
            )

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")

    model.load_state_dict(best_state)

    seed_dir = OUT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    save_loss_curve(
        history, best_epoch, name, seed, seed_dir
    )

    # Save the epoch histories as numerical data too.
    history_frame = pd.DataFrame({
        "epoch": np.arange(1, len(history["train"]) + 1),
        "train_MSE": history["train"],
        "val_MSE": history["val"],
    })

    safe_name = name.replace(" ", "_").replace("-", "_")

    history_frame.to_csv(
        seed_dir / f"{safe_name}_history.csv",
        index=False,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": name,
            "seed": seed,
            "latent_dim": latent_dim,
            "best_epoch": best_epoch,
            "cardinalities": cardinalities,
        },
        seed_dir / f"{safe_name}_checkpoint.pt",
    )

    result = {
        "model": name,
        "seed": seed,
        "parameters": params,
        "best_epoch": best_epoch,
        "best_val_normalized_MSE": best_loss,
    }

    for split in ["train", "val", "test"]:
        actual = data[split][1]

        prediction = predict(
            model, loaders[split], device
        )

        split_metrics = calculate_metrics(
            prediction, actual, data
        )

        for metric, value in split_metrics.items():
            result[f"{split}_{metric}"] = value

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Test R2: {result['test_R2']:.5f} | "
        f"Test RMSE: ${result['test_RMSE']:,.2f}"
    )

    return result


# ---------------------------------------------------------
# 8. Aggregate statistics
# ---------------------------------------------------------

def make_summary(results):
    return results.groupby(
        "model", sort=False
    ).agg(
        parameters=("parameters", "first"),
        runs=("seed", "count"),
        R2_mean=("test_R2", "mean"),
        R2_std=("test_R2", "std"),
        RMSE_mean=("test_RMSE", "mean"),
        RMSE_std=("test_RMSE", "std"),
        MAE_mean=("test_MAE", "mean"),
        MAE_std=("test_MAE", "std"),
        best_epoch_mean=("best_epoch", "mean"),
    ).reset_index()


def paired_comparisons(results):
    r2 = results.pivot(
        index="seed", columns="model", values="test_R2"
    )

    rmse = results.pivot(
        index="seed", columns="model", values="test_RMSE"
    )

    rows = []

    for competitor in ["VANTA-G", "Embedding MLP"]:
        # Positive dR2 means VANTA-R performed better.
        delta_r2 = r2["VANTA-R"] - r2[competitor]

        # Positive RMSE reduction means VANTA-R performed better.
        delta_rmse = rmse[competitor] - rmse["VANTA-R"]

        rows.append({
            "comparison": f"VANTA-R vs {competitor}",
            "mean_delta_R2": delta_r2.mean(),
            "std_delta_R2": delta_r2.std(),
            "mean_RMSE_reduction": delta_rmse.mean(),
            "std_RMSE_reduction": delta_rmse.std(),
            "VANTA_R_wins": int((delta_rmse > 0).sum()),
            "seeds": len(delta_rmse),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 9. Comparison plots
# ---------------------------------------------------------

def save_comparison_plots(results):
    for metric, ylabel, filename in [
        ("test_R2", "Test R2", "r2_by_seed.png"),
        ("test_RMSE", "Test RMSE ($)", "rmse_by_seed.png"),
    ]:
        plt.figure(figsize=(9, 5))

        for name in MODELS:
            subset = results[
                results["model"] == name
            ].sort_values("seed")

            plt.plot(
                subset["seed"],
                subset[metric],
                marker="o",
                label=name,
            )

        plt.xlabel("Initialization seed")
        plt.ylabel(ylabel)
        plt.title("VANTA Experiment 02")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.savefig(OUT / filename, dpi=180)
        plt.close()


# ---------------------------------------------------------
# 10. Main experiment
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--gamma-init", type=float, default=0.1)
    parser.add_argument("--csv", type=str, default=None)

    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.patience) <= 0:
        parser.error("epochs, batch-size and patience must be positive")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds are not allowed")

    # Original checkpoint defines reference architecture.
    checkpoint = torch.load(
        R_DIR / "attention_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )

    if args.split_seed != int(checkpoint["seed"]):
        parser.error(
            "Use the original checkpoint split seed for EX02."
        )

    latent_dim = int(checkpoint["latent_dim"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("VANTA - EXPERIMENT 02: MULTI-SEED ROBUSTNESS")
    print("=" * 65)

    print(f"Device: {device}")
    print(f"Fixed split seed: {args.split_seed}")
    print(f"Initialization seeds: {args.seeds}")
    print(f"Latent dimension: {latent_dim}")

    # Load and preprocess dataset ONCE, using fixed split.
    df = vanta_data.load_dataset(args.csv)
    data = vanta_data.prepare_data(df, args.split_seed)

    cardinalities = [
        len(data["vocabularies"][col]) + 1
        for col in CATEGORICAL
    ]

    if checkpoint["features"] != FEATURES:
        raise ValueError("Feature order differs from VANTA-R.")

    if checkpoint["cardinalities"] != cardinalities:
        raise ValueError("Category cardinalities differ.")

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

    # Determine VANTA-R parameter budget.
    reference = build_model(
        "VANTA-R",
        args.seeds[0],
        cardinalities,
        latent_dim,
        args.gamma_init,
        1,
    )

    target_params = count_parameters(reference)

    mlp_width = matched_mlp_width(
        target_params, cardinalities
    )

    print(f"VANTA-R parameter budget: {target_params:,}")
    print(f"Matched MLP hidden width: {mlp_width}")

    rows = []

    # Sequential runs: one model at a time, for each seed.
    for seed in args.seeds:
        for name in MODELS:
            result = run_one(
                name,
                seed,
                args,
                data,
                loaders,
                cardinalities,
                latent_dim,
                mlp_width,
                device,
            )

            rows.append(result)

            # Save incrementally so completed runs survive
            # an interruption or later training failure.
            pd.DataFrame(rows).to_csv(
                OUT / "metrics.csv",
                index=False,
            )

    results = pd.DataFrame(rows)

    summary = make_summary(results)
    comparisons = paired_comparisons(results)

    summary.to_csv(
        OUT / "summary.csv", index=False
    )

    comparisons.to_csv(
        OUT / "paired_comparisons.csv", index=False
    )

    save_comparison_plots(results)

    print("\n" + "=" * 80)
    print("FINAL RESULTS: MEAN +/- SD ACROSS SEEDS")
    print("=" * 80)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 80)
    print("PAIRED COMPARISONS")
    print("=" * 80)

    print(
        comparisons.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print(f"\nResults saved to: {OUT}")


if __name__ == "__main__":
    main()

